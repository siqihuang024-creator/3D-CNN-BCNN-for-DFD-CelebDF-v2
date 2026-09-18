"""Verify that a manifest resolves against this machine's dataset roots.

Run this first on a rented GPU. A wrong root, a case-mismatched folder, or a
missing video codec otherwise only shows up deep into a training run.
"""

import argparse
import collections
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_bcnn.data import VideoClipDataset
from video_bcnn.utils import load_config, override_dataset_roots, verify_dataset_roots


def sample_rows(rows, active, per_group):
    """Take a deterministic sample from every dataset/split/class group."""
    groups = collections.defaultdict(list)
    for row in rows:
        if active and row["dataset"] not in active:
            continue
        groups[(row["dataset"], row["split"], row["class_name"])].append(row)
    selected = []
    for key in sorted(groups):
        values = groups[key]
        rng = random.Random(hash(key) & 0xFFFFFFFF)
        selected.extend(rng.sample(values, min(per_group, len(values))))
    return groups, selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-root", action="append", default=None, metavar="NAME=PATH")
    parser.add_argument("--per-group", type=int, default=5,
                        help="Videos to check per dataset/split/class group.")
    parser.add_argument("--decode", action="store_true",
                        help="Also open each sampled video to confirm codec support.")
    parser.add_argument("--all", action="store_true",
                        help="Check every video the experiment reads, not a sample. "
                             "Implies --decode. ~5 minutes for DFD's 2401 videos, and "
                             "worth it before a multi-hour run.")
    parser.add_argument("--splits", nargs="+", default=None,
                        help="Restrict --all to these splits (default: train val test).")
    args = parser.parse_args()

    config = override_dataset_roots(load_config(args.config), args.dataset_root)
    roots = verify_dataset_roots(config)
    print("Dataset roots exist:")
    for name, root in sorted(roots.items()):
        print("  {:<12} {}".format(name, root))

    with open(args.manifest, "r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    active = set(config["data"].get("active_datasets") or [])
    groups, selected = sample_rows(rows, active, args.per_group)
    if args.all:
        splits = set(args.splits or ("train", "val", "test"))
        selected = [
            row for row in rows
            if row["split"] in splits and (not active or row["dataset"] in active)
        ]

    print("\nManifest groups (dataset / split / class -> videos):")
    for key in sorted(groups):
        print("  {:<10} {:<22} {:<5} {}".format(key[0], key[1], key[2], len(groups[key])))

    decode = args.decode or args.all
    if args.all:
        print("\nFull scan of {} videos: each is opened and its first and last frame "
              "read.".format(len(selected)))
    missing, unreadable, truncated = [], [], []
    for position, row in enumerate(selected):
        path = Path(roots[row["dataset"]]) / row["path"]
        if args.all and position and position % 500 == 0:
            print("  {}/{} checked... ({} problems so far)".format(
                position, len(selected), len(missing) + len(unreadable) + len(truncated)),
                flush=True)
        if not path.is_file():
            missing.append(path)
            # Reported as found, so an interrupted scan still yields its results.
            print("  MISSING     {}".format(path), flush=True)
            continue
        if not decode:
            continue
        import cv2
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened() or not capture.read()[0]:
            capture.release()
            unreadable.append(path)
            print("  UNREADABLE  {}".format(path), flush=True)
            continue
        # Metadata frequently overstates the length; the last frame is where a
        # truncated upload or an unusable seek index shows up.
        declared = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, declared - 1))
        if declared < 1 or not capture.read()[0]:
            measured = VideoClipDataset._measure_decodable_frames(capture)
            truncated.append((path, declared, measured))
            print("  MISMATCH    {}  declared={} decodable={}".format(
                path, declared, measured), flush=True)
        capture.release()

    print("\nChecked {} videos.".format(len(selected)))
    if missing:
        print("MISSING {} file(s). First few:".format(len(missing)))
        for path in missing[:5]:
            print("  {}".format(path))
        print("\nThe manifest stores paths relative to the root, so a root that exists but "
              "yields missing files usually means the folder layout or letter case differs "
              "from the Windows copy. Linux is case-sensitive; Windows is not.")
    if unreadable:
        print("UNREADABLE {} file(s) (codec/OpenCV problem). First few:".format(len(unreadable)))
        for path in unreadable[:5]:
            print("  {}".format(path))
    if truncated:
        print("SEEK/LENGTH MISMATCH on {} file(s): metadata disagrees with what the "
              "decoder yields. Training recovers from these automatically (it "
              "re-measures and decodes without seeking), but a large count usually "
              "means an incomplete upload. First few:".format(len(truncated)))
        for path, declared, measured in truncated[:5]:
            print("  {}  declared={} decodable={}".format(path, declared, measured))
    if not missing and not unreadable and not truncated:
        print("OK: every video resolved{}.".format(" and decoded" if decode else ""))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
