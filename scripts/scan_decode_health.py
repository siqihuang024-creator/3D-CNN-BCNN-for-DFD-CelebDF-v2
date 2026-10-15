"""Decode every manifest video end to end and record what FFmpeg complains about.

``report_video_sizes.py`` reads container metadata only, so a file whose header
is intact but whose stream is truncated is reported ``ok``. This scan records
the number of frames FFmpeg actually outputs and can flag clips shorter than a
specified minimum. The first Stage A
run on DFD showed what that misses: ``moov atom not found``, ``partial file``,
``Invalid NAL unit size`` and 30-second read timeouts, all while the epoch line
reported ``skipped=0``.

The legacy, non-strict loader could hide these failures by repeating a frame.
The revised strict-clip protocol rejects missing frames at runtime, but an
up-front full-decode scan avoids discovering failures mid-training.

That is a correctness problem before it is a speed problem. If corruption is
not spread evenly across the classes, "is this clip frozen?" becomes a free
detector of exactly the kind the shortcut gate exists to catch, and the network
will find it. So this script measures the rate per class and reports the gap
with a two-proportion test.

Full decode is the only check that sees the problem: FFmpeg reports nothing
until it reaches the damaged bytes.

Usage:
    python scripts/scan_decode_health.py \
        --manifest artifacts/manifests/combined_manifest_stage_a.csv \
        --datasets DFD --workers 4 \
        --output artifacts/decode_health_dfd.csv

    python scripts/scan_decode_health.py \
        --manifest artifacts/manifests/celeb_phase_a_revised_1to2_merged.csv \
        --datasets CelebDFv3 --splits train val --min-frames 32 --workers 4 \
        --output artifacts/v2_revised/decode_health.csv
"""

import argparse
import csv
import math
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_bcnn.utils import DATASET_ROOT_ENV_VARS

# Lines FFmpeg emits that describe a damaged stream rather than a stylistic
# quibble. Anything outside this set is still recorded, just not counted as
# corruption, so a new failure mode shows up in the CSV instead of vanishing.
FATAL_MARKERS = (
    "moov atom not found",
    "partial file",
    "Invalid NAL unit size",
    "Error splitting the input into NAL units",
    "missing picture in access unit",
    "Invalid data found when processing input",
    "error while decoding",
    "corrupt",
    "truncat",
)


def resolve_roots(overrides):
    roots = {}
    for name, variable in DATASET_ROOT_ENV_VARS.items():
        value = os.environ.get(variable)
        if value:
            roots[name] = Path(value)
    for item in overrides or []:
        name, separator, path = item.partition("=")
        if not separator:
            raise ValueError("--dataset-root expects NAME=PATH, got {!r}".format(item))
        roots[name] = Path(path)
    return roots


def probe(video_path, timeout):
    """Decode the whole file and report errors and actual output frame count.

    ``-f null -`` decodes every frame and throws the output away, which is what
    makes the damaged bytes reachable. ``-v error`` keeps the banner out.
    """
    command = ["ffmpeg", "-v", "error", "-progress", "pipe:1",
               "-i", str(video_path), "-map", "0:v:0", "-an",
               "-vsync", "0", "-f", "null", "-"]
    started = time.time()
    try:
        finished = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=timeout)
    except subprocess.TimeoutExpired:
        return "timeout", -1, "decode exceeded {}s".format(timeout), time.time() - started, -1
    lines = [line.strip() for line in
             finished.stderr.decode("utf-8", "replace").splitlines() if line.strip()]
    elapsed = time.time() - started
    frame_lines = [line.partition("=")[2] for line in
                   finished.stdout.decode("utf-8", "replace").splitlines()
                   if line.startswith("frame=")]
    decoded_frames = int(frame_lines[-1]) if frame_lines else -1
    if finished.returncode != 0:
        return "corrupt", len(lines), (lines or ["ffmpeg exited unsuccessfully"])[0][:200], elapsed, decoded_frames
    if not lines:
        return "ok", 0, "", elapsed, decoded_frames
    fatal = [line for line in lines
             if any(marker.lower() in line.lower() for marker in FATAL_MARKERS)]
    verdict = "corrupt" if fatal else "warning"
    return verdict, len(lines), (fatal or lines)[0][:200], elapsed, decoded_frames


def two_proportion_z(bad_a, total_a, bad_b, total_b):
    """z for H0: the two rates are equal. Returns (z, pooled rate)."""
    if not total_a or not total_b:
        return float("nan"), float("nan")
    pooled = (bad_a + bad_b) / float(total_a + total_b)
    spread = pooled * (1 - pooled) * (1.0 / total_a + 1.0 / total_b)
    if spread <= 0:
        return float("nan"), pooled
    z = (bad_a / float(total_a) - bad_b / float(total_b)) / math.sqrt(spread)
    return z, pooled


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-root", action="append", default=None,
                        metavar="NAME=PATH")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Parallel ffmpeg processes. Keep this well under nproc when a "
             "training run is already competing for the same cores.")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Seconds before a single file is given up on.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-frames", type=int, default=None,
                        help="Flag videos with fewer than this many fully decoded frames.")
    parser.add_argument("--output", default="artifacts/decode_health.csv")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        print("ffmpeg is not on PATH. Install it with "
              "conda install -c conda-forge ffmpeg -- the copy bundled inside "
              "OpenCV cannot be driven from the shell.")
        return 2

    roots = resolve_roots(args.dataset_root)
    with open(args.manifest, "r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row["split"] in set(args.splits)]
    if args.datasets:
        rows = [row for row in rows if row["dataset"] in set(args.datasets)]
    missing = sorted({row["dataset"] for row in rows} - set(roots))
    if missing:
        print("No root for {}. Set {} or pass --dataset-root.".format(
            missing, [DATASET_ROOT_ENV_VARS.get(name) for name in missing]))
        return 2
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        print("The filters selected no videos.")
        return 2
    print("probing {} videos with {} parallel ffmpeg processes".format(
        len(rows), args.workers))

    def work(row):
        path = roots[row["dataset"]] / row["path"]
        if not path.exists():
            return row, ("missing", -1, "file not found", 0.0, -1)
        return row, probe(path, args.timeout)

    results, started = [], time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, (row, outcome) in enumerate(pool.map(work, rows), start=1):
            verdict, count, message, elapsed, decoded_frames = outcome
            if args.min_frames is not None and verdict in ("ok", "warning"):
                if decoded_frames < 0:
                    verdict = "unverified"
                    message = "ffmpeg did not report a decoded frame count"
                elif decoded_frames < args.min_frames:
                    verdict = "short"
                    message = "decoded {} frames; need {}".format(
                        decoded_frames, args.min_frames)
            results.append({
                "dataset": row["dataset"], "split": row["split"],
                "class_name": row["class_name"], "label": row["label"],
                "path": row["path"], "verdict": verdict,
                "ffmpeg_messages": count, "first_message": message,
                "decoded_frames": decoded_frames,
                "decode_seconds": round(elapsed, 2),
            })
            if index % 50 == 0 or index == len(rows):
                rate = index / max(time.time() - started, 1e-9)
                bad = sum(1 for item in results if item["verdict"] != "ok")
                print("  {}/{}  bad so far {}  {:.1f} videos/s  eta {:.0f} min"
                      .format(index, len(rows), bad, rate,
                              (len(rows) - index) / rate / 60))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print("\nwrote {}".format(output))

    tally = {}
    for item in results:
        key = (item["dataset"], item["class_name"])
        entry = tally.setdefault(key, {"videos": 0, "corrupt": 0, "stalled": 0})
        entry["videos"] += 1
        if item["verdict"] == "corrupt":
            entry["corrupt"] += 1
        elif item["verdict"] in ("timeout", "missing", "short", "unverified"):
            entry["stalled"] += 1

    print("\n{:<12} {:<10} {:>7} {:>9} {:>9} {:>8}".format(
        "dataset", "class", "videos", "corrupt", "stalled", "rate"))
    for key in sorted(tally):
        entry = tally[key]
        bad = entry["corrupt"] + entry["stalled"]
        print("{:<12} {:<10} {:>7} {:>9} {:>9} {:>7.1%}".format(
            key[0], key[1], entry["videos"], entry["corrupt"],
            entry["stalled"], bad / float(entry["videos"])))

    print("\nIs corruption a free detector?")
    for dataset in sorted({key[0] for key in tally}):
        real = tally.get((dataset, "real"))
        fake = tally.get((dataset, "fake"))
        if not real or not fake:
            continue
        bad_real = real["corrupt"] + real["stalled"]
        bad_fake = fake["corrupt"] + fake["stalled"]
        z, _ = two_proportion_z(bad_real, real["videos"], bad_fake, fake["videos"])
        # |z| > 1.96 is the 5% two-sided threshold. A gap that large means a
        # ranker reading "is this clip frozen?" beats chance, and the affected
        # videos have to leave the manifest before the number is trustworthy.
        flag = "SHORTCUT RISK" if abs(z) > 1.96 else "no measurable gap"
        print("  {}: real {:.1%} vs fake {:.1%}, z={:+.2f}  -> {}".format(
            dataset, bad_real / float(real["videos"]),
            bad_fake / float(fake["videos"]), z, flag))
    return 0


if __name__ == "__main__":
    sys.exit(main())
