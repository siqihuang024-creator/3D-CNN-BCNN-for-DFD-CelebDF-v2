"""List exactly the videos an experiment reads, for a minimal dataset upload.

The manifests park a large share of each dataset in `unused`,
`excluded_donor_train`, or `excluded_donor_eval`; those files are never opened.
Uploading only the referenced videos cuts the transfer substantially.

    python scripts/list_required_videos.py \
      --config configs/dfd_supervised_3d.yaml \
      --manifest artifacts/manifests/combined_manifest_supervised.csv \
      --out-dir upload_lists

Each dataset gets one newline-separated file of paths relative to that
dataset's root, ready for `rsync --files-from` or `rclone --files-from`:

    rsync -av --files-from=upload_lists/DFD.txt \
      "/local/DFD-Kaggle/" remote:/root/autodl-tmp/datasets/DFD-Kaggle/
"""

import argparse
import collections
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_bcnn.utils import load_config, override_dataset_roots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--out-dir", default=None,
                        help="Write one <dataset>.txt per dataset here.")
    parser.add_argument("--measure", action="store_true",
                        help="Also total the bytes on this machine (slow but exact).")
    parser.add_argument("--dataset-root", action="append", default=None, metavar="NAME=PATH")
    args = parser.parse_args()

    config = override_dataset_roots(load_config(args.config), args.dataset_root)
    active = set(config["data"].get("active_datasets") or [])
    roots = config["data"]["dataset_roots"]

    with open(args.manifest, "r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    wanted = collections.defaultdict(list)
    skipped = collections.Counter()
    for row in rows:
        if active and row["dataset"] not in active:
            continue
        if row["split"] in args.splits:
            wanted[row["dataset"]].append(row["path"])
        else:
            skipped[(row["dataset"], row["split"])] += 1

    if not wanted:
        raise SystemExit("No videos matched. Check --splits and the config's active_datasets.")

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for dataset in sorted(wanted):
        paths = sorted(set(wanted[dataset]))
        line = "{}: {} videos needed".format(dataset, len(paths))
        if args.measure:
            root = Path(roots[dataset])
            total = sum((root / item).stat().st_size for item in paths if (root / item).is_file())
            line += "  ({:.1f} GB)".format(total / 1e9)
        print(line)
        if out_dir:
            target = out_dir / "{}.txt".format(dataset)
            target.write_text("\n".join(paths) + "\n", encoding="utf-8")
            print("  -> {}".format(target))

    if skipped:
        print("\nNot needed (never opened by this experiment):")
        for (dataset, split), count in sorted(skipped.items()):
            print("  {:<12} {:<22} {}".format(dataset, split, count))


if __name__ == "__main__":
    main()
