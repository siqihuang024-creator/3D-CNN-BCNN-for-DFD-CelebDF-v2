"""Check whether seeking returns the frames the loader asked for.

``_decode_clips`` seeks to a random start with
``capture.set(cv2.CAP_PROP_POS_FRAMES, start)`` and then reads forward
(``data.py:350``). On H.264 that seek has to resume from the nearest keyframe,
and OpenCV's implementation is unreliable on long files with sparse keyframes:
it can land off by several frames, hand back a stalled read that the loader
then papers over by repeating the previous frame (``data.py:358-365``), or emit
exactly the `Invalid NAL unit size` / `Error splitting the input into NAL units`
messages that filled the DFD training log while the epoch line still said
``skipped=0``.

That failure mode is dataset-shaped, which is why it is worth measuring here:
DFD is a tens-of-seconds 1080p scene, CelebDFv3 a short clip. If the seek is
lossy on one and clean on the other, it explains a gap that identity count,
face detection and input mode have each already failed to explain.

Method: decode a video straight through once and keep a 32x32 grey signature
per frame -- cheap enough to hold a whole video in a few hundred kilobytes.
Then take the loader's own path for the same frame indices and, for each frame
that comes back, find which sequential frame it actually matches. The offset
between "asked for" and "got" is the measurement; a clip whose frames collapse
onto one index is the frozen-clip failure the loader hides.

Usage:
    python scripts/seek_fidelity.py \
        --manifest artifacts/manifests/combined_manifest_stage_a.csv \
        --datasets DFD CelebDFv3 --videos-per-dataset 20
"""

import argparse
import csv
import os
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from video_bcnn.utils import DATASET_ROOT_ENV_VARS  # noqa: E402

# A 32x32 grey thumbnail separates neighbouring frames of a talking head while
# staying small enough to keep one per frame for a whole video.
SIGNATURE_SIDE = 32
# Frames closer than this in mean absolute grey level are treated as the same
# picture. Consecutive frames of real footage sit far above it; a repeated
# frame sits at zero.
SAME_FRAME_TOLERANCE = 0.75


def signature(frame):
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(grey, (SIGNATURE_SIDE, SIGNATURE_SIDE),
                       interpolation=cv2.INTER_AREA)
    return small.astype(np.float32)


def sequential_signatures(path, limit):
    """Ground truth: decode from the start, no seeking anywhere."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return None
    signatures = []
    try:
        while len(signatures) < limit:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            signatures.append(signature(frame))
    finally:
        capture.release()
    return np.stack(signatures) if signatures else None


def seek_clip(capture, indices):
    """The loader's path: one seek to the start, then read forward.

    Takes an open capture rather than a path because `_decode_clips` issues
    every clip's seek on the **same** handle (`data.py:348-355`), and a handle
    that has already been seeked does not necessarily behave like a fresh one.
    """
    start, end = int(indices[0]), int(indices[-1])
    wanted = set(int(index) for index in indices)
    got = {}
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    for position in range(start, end + 1):
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        if position in wanted:
            got[position] = signature(frame)
    return [got.get(int(index)) for index in indices]


def nearest_index(truth, probe):
    """Which sequential frame does this returned frame actually match?"""
    distance = np.abs(truth - probe).mean(axis=(1, 2))
    best = int(np.argmin(distance))
    return best, float(distance[best])


def measure(path, clip_length, stride, clips, limit, generator):
    truth = sequential_signatures(path, limit)
    if truth is None or len(truth) < clip_length * stride + 2:
        return None
    span = clip_length * stride
    highest = len(truth) - span - 1
    if highest < 1:
        return None
    result = {"frames_decoded": len(truth), "requested": 0, "returned": 0,
              "exact": 0, "offset_sum": 0, "offset_max": 0,
              "frozen_clips": 0, "clips": 0}
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return None
    starts = [int(generator.randint(0, highest)) for _ in range(clips)]
    try:
        for start in starts:
            indices = [start + step * stride for step in range(clip_length)]
            frames = seek_clip(capture, indices)
            if frames is None:
                continue
            result["clips"] += 1
            matched = []
            for wanted, frame in zip(indices, frames):
                result["requested"] += 1
                if frame is None:
                    continue
                result["returned"] += 1
                actual, _ = nearest_index(truth, frame)
                offset = abs(actual - wanted)
                if offset == 0:
                    result["exact"] += 1
                result["offset_sum"] += offset
                result["offset_max"] = max(result["offset_max"], offset)
                matched.append(actual)
            # A clip whose frames land on fewer distinct pictures than it asked
            # for carries less motion than the model is told it does; in the
            # worst case every frame is the same and the clip has no motion.
            if matched and len(set(matched)) < len(matched):
                result["frozen_clips"] += 1
    finally:
        capture.release()
    return result


def resolve_roots():
    roots = {}
    for name, variable in DATASET_ROOT_ENV_VARS.items():
        value = os.environ.get(variable)
        if value:
            roots[name] = Path(value)
    return roots


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--datasets", nargs="+", default=["DFD", "CelebDFv3"])
    parser.add_argument("--splits", nargs="+", default=["train"])
    parser.add_argument("--videos-per-dataset", type=int, default=20)
    parser.add_argument("--clips-per-video", type=int, default=4)
    parser.add_argument("--clip-length", type=int, default=8)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=1500,
                        help="Cap on frames decoded per video for the ground "
                             "truth, so one long file cannot dominate the run. "
                             "DFD scenes run past 900 frames, and a seek is "
                             "likelier to go wrong deep into a file.")
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Videos measured concurrently. Training ran 12 DataLoader workers "
             "seeking at random into 23 GB of 1080p video at once, and the "
             "30-second read timeouts in its log are what I/O contention looks "
             "like -- a stalled read the loader then hides by repeating the "
             "previous frame. One process cannot reproduce that, so match the "
             "worker count the run actually used.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="artifacts/seek_fidelity.csv")
    args = parser.parse_args()

    roots = resolve_roots()
    with open(args.manifest, "r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row["split"] in set(args.splits)]

    generator = np.random.RandomState(args.seed)
    records, totals = [], {}
    for dataset in args.datasets:
        if dataset not in roots:
            print("no root for {}; set {}".format(
                dataset, DATASET_ROOT_ENV_VARS.get(dataset)))
            continue
        here = [row for row in rows if row["dataset"] == dataset]
        chosen = []
        for label in ("1", "0"):
            pool = [row for row in here if row["label"] == label]
            generator.shuffle(pool)
            chosen.extend(pool[:max(args.videos_per_dataset // 2, 1)])
        print("\n=== {}: {} videos, {} at a time ===".format(
            dataset, len(chosen), max(args.workers, 1)), flush=True)

        def work(row, dataset=dataset):
            path = roots[dataset] / row["path"]
            if not path.exists():
                return row, None
            # A generator per video, seeded from its path, so the clip starts
            # do not depend on the order threads happen to finish in.
            seed = zlib.crc32(row["path"].encode("utf-8")) ^ args.seed
            return row, measure(path, args.clip_length, args.stride,
                                args.clips_per_video, args.max_frames,
                                np.random.RandomState(seed % (2 ** 31)))

        with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as pool:
            for position, (row, outcome) in enumerate(pool.map(work, chosen),
                                                      start=1):
                if outcome is None:
                    continue
                records.append(dict(outcome, dataset=dataset,
                                    class_name=row["class_name"],
                                    path=row["path"]))
                bucket = totals.setdefault((dataset, row["class_name"]),
                                           {key: 0 for key in outcome})
                bucket["videos"] = bucket.get("videos", 0) + 1
                for key, value in outcome.items():
                    if key == "offset_max":
                        bucket[key] = max(bucket[key], value)
                    else:
                        bucket[key] += value
                if position % 5 == 0 or position == len(chosen):
                    print("  {}/{}".format(position, len(chosen)), flush=True)

    if not records:
        print("nothing measured")
        return 2

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print("\nwrote {}".format(output))

    print("\n{:<12} {:<7} {:>7} {:>9} {:>10} {:>10} {:>9} {:>8}".format(
        "dataset", "class", "videos", "returned", "exact hit", "mean off",
        "max off", "frozen"))
    for key in sorted(totals):
        item = totals[key]
        requested = max(item["requested"], 1)
        returned = max(item["returned"], 1)
        print("{:<12} {:<7} {:>7} {:>8.1%} {:>10.1%} {:>10.2f} {:>9} {:>7.1%}".format(
            key[0], key[1], item["videos"], item["returned"] / requested,
            item["exact"] / returned, item["offset_sum"] / returned,
            item["offset_max"], item["frozen_clips"] / max(item["clips"], 1)))

    print("\nHow to read this:")
    print("  exact hit near 100%  -> seeking returns the frames asked for")
    print("  exact hit well below -> the loader trains on frames it did not")
    print("                          request, and 'mean off' says by how far")
    print("  frozen               -> clips whose frames collapse onto fewer")
    print("                          pictures than requested: less motion than")
    print("                          the model is told it has")
    print("  A gap between the two datasets is the point: it would explain a")
    print("  DFD-specific failure that identity count, face detection and")
    print("  input mode have each failed to explain.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
