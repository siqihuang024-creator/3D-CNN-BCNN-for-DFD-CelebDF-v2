"""Rank the test videos by container metadata, and ask whether the model agrees.

Two separate questions, and they must not be conflated. The first is how far the
labels can be separated by information that never passed through the network --
file bitrate, frame geometry, duration -- which is a property of the benchmark,
not of the detector. The second is whether the detector's own scores move with
that information, which is a property of the detector.

The second question can only be answered as far as a rank correlation reaches: a
Spearman coefficient near zero rules out a monotonic association and nothing
more. A non-linear dependence, or one that runs through resolution or coding
artefacts, would not show up here. Say "no monotonic association", never "the
model does not use bitrate".

AUROC is symmetric in direction, so a control reading 0.2174 orders the classes
exactly as strongly as one reading 0.7826; what the tables report is the reading
itself, and the orientation-adjusted value beside it.

Usage:
    python scripts/metadata_association.py \
        --scores results/run_stage_a_celebdfv3_face_stageb_celeb/report_test_scores.csv \
        --video-sizes scripts/video_size_reports/video_sizes.csv \
        --dataset CelebDFv3 \
        --output results/diagnostics/metadata_association_celeb_test.json
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
BACKSLASH = chr(92)


def read_video_sizes(path, dataset):
    """(relative path) -> the scalars a container exposes without decoding."""
    table = {}
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "ok" or row["dataset"] != dataset:
                continue
            relative = row["relative_path"].replace(BACKSLASH, "/")
            parts = relative.split("/")
            if len(parts) > 1 and parts[0] == row["dataset"]:
                relative = "/".join(parts[1:])
            try:
                width, height = float(row["width"]), float(row["height"])
                frames, duration = float(row["frame_count"]), float(row["duration_seconds"])
                megabytes = float(row["file_size_mb"])
            except (KeyError, TypeError, ValueError):
                continue
            if width <= 0 or height <= 0 or duration <= 0 or frames <= 0:
                continue
            table[relative] = {
                "bitrate_mb_per_second": megabytes / duration,
                "bits_per_pixel": (megabytes * 8e6) / (frames * width * height),
                "frame_width": width,
                "frame_height": height,
                "frame_pixels": width * height,
                "aspect_ratio": width / height,
                "frame_count": frames,
            }
    return table


def spearman(a, b):
    """Rank correlation without a scipy dependency; ties are broken by order."""
    ranks_a = np.argsort(np.argsort(a)).astype(float)
    ranks_b = np.argsort(np.argsort(b)).astype(float)
    ranks_a -= ranks_a.mean()
    ranks_b -= ranks_b.mean()
    denominator = np.sqrt((ranks_a ** 2).sum() * (ranks_b ** 2).sum())
    return float((ranks_a * ranks_b).sum() / denominator) if denominator else float("nan")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", required=True,
                        help="A *_scores.csv written by evaluate_3d_bcnn.py.")
    parser.add_argument("--video-sizes", required=True)
    parser.add_argument("--dataset", required=True,
                        help="Which corpus in video_sizes.csv the scores belong to.")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    sizes = read_video_sizes(args.video_sizes, args.dataset)
    with open(args.scores, "r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    # Current reports carry the manifest-relative path explicitly.  Older
    # reports only have the decoded absolute path, so retain that fallback.
    scores, labels, controls, matched = [], [], {}, 0
    for row in rows:
        key = (row.get("relative_path") or row["path"]).replace(BACKSLASH, "/")
        key = key.rsplit(args.dataset + "/", 1)[-1]
        entry = sizes.get(key)
        if entry is None:
            continue
        matched += 1
        scores.append(float(row["anomaly_score"]))
        labels.append(int(row["label_real"]))
        for name, value in entry.items():
            controls.setdefault(name, []).append(value)

    if matched < 2:
        raise SystemExit("Only {} of {} scored videos matched the size table; check "
                         "--dataset and the paths.".format(matched, len(rows)))
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    fake = 1 - labels
    print("matched {} of {} scored videos ({} real / {} fake)".format(
        matched, len(rows), int(labels.sum()), int(fake.sum())))

    record = {
        "scores": str(Path(args.scores).as_posix()),
        "dataset": args.dataset,
        "videos": int(matched),
        "real": int(labels.sum()),
        "fake": int(fake.sum()),
        "model_auroc": float(roc_auc_score(fake, scores)),
        "controls": {},
    }
    print("\nmodel AUROC on this subset: {:.4f}\n".format(record["model_auroc"]))
    print("{:<24}{:>9}{:>12}{:>11}{:>11}{:>11}".format(
        "control", "auroc", "orientation", "rho all", "rho real", "rho fake"))
    for name in sorted(controls):
        values = np.asarray(controls[name], dtype=float)
        auroc = float(roc_auc_score(fake, values))
        entry = {
            "auroc": auroc,
            "orientation_adjusted": max(auroc, 1.0 - auroc),
            "spearman_with_score": {
                "all": spearman(scores, values),
                "real": spearman(scores[labels == 1], values[labels == 1]),
                "fake": spearman(scores[labels == 0], values[labels == 0]),
            },
        }
        record["controls"][name] = entry
        print("{:<24}{:>9.4f}{:>12.4f}{:>11.4f}{:>11.4f}{:>11.4f}".format(
            name, auroc, entry["orientation_adjusted"],
            entry["spearman_with_score"]["all"],
            entry["spearman_with_score"]["real"],
            entry["spearman_with_score"]["fake"]))

    destination = Path(args.output or (ROOT / "results/diagnostics"
                                       / "metadata_association.json"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    print("\nwrote {}".format(destination))
    print("A rank correlation near zero rules out a monotonic association only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
