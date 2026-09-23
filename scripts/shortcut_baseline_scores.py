"""Score a split with one metadata control, in the model's own score format.

A shortcut floor read beside a model's AUROC answers nothing on its own: the
two numbers carry no interval, and 0.802 against 0.795 is not a difference.
Writing the control as a video_scores.csv turns it into one more run that
compare_experiments.py can resample, so the question becomes a paired,
identity-clustered interval on

    AUROC(model) - AUROC(control)

measured on the same videos under the same resampling.

The control is oriented so that a larger value means "more fake", flipping it
when it ranks the other way. The flip reads the labels, so an oriented control
is an optimistic opponent -- which is the conservative direction here, because
it makes the model's advantage harder to claim, not easier.

Usage:
    python scripts/shortcut_baseline_scores.py \
        --manifest artifacts/manifests/combined_manifest_p05.csv \
        --video-sizes scripts/video_size_reports/video_sizes.csv \
        --box-cache artifacts/face_cache --dataset CelebDFv3 --split val \
        --control face_box_share_of_frame \
        --match artifacts/v2/<run>/reports/full_val_video_scores.csv \
        --output results/v2/shortcut_facebox_scores.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from metadata_association import add_face_controls, read_video_sizes  # noqa: E402

BACKSLASH = chr(92)
FIELDS = ["video_id", "video_path", "relative_path", "dataset", "split",
          "label_real", "label_fake", "forgery_method", "identity",
          "source_identity", "source_family_id", "video_score", "num_clips",
          "predictive_mean", "predictive_std", "embedding_norm", "checkpoint",
          "experiment", "seed"]


def video_id(dataset, relative):
    return "{}::{}".format(dataset, str(relative).replace(BACKSLASH, "/"))


def wanted_ids(path):
    """The video IDs a model run scored, so the two files line up exactly."""
    if path is None:
        return None
    with open(path, "r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {row["video_id"] for row in rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--video-sizes", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--control", default="face_box_share_of_frame",
                        help="Which metadata column becomes the detector.")
    parser.add_argument("--box-cache", default=None,
                        help="Detection-cache root; required for the crop controls.")
    parser.add_argument("--match", default=None,
                        help="A model's *_video_scores.csv. Only the videos it "
                             "scored are written, so the files are comparable.")
    parser.add_argument("--min-coverage", type=float, default=0.99,
                        help="Minimum fraction of --match videos with a valid control "
                             "(default: 0.99).")
    parser.add_argument("--orient", choices=["auto", "raw"], default="auto",
                        help="auto flips the control when it ranks real above "
                             "fake, so larger always means more fake.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 0 < args.min_coverage <= 1:
        parser.error("--min-coverage must be between 0 and 1.")

    metadata = read_video_sizes(args.video_sizes, args.dataset)
    if args.control.startswith("face_box_"):
        if not args.box_cache:
            parser.error("--box-cache is required for a face-box control.")
        add_face_controls(metadata, args.box_cache, args.dataset)
    with open(args.manifest, "r", newline="", encoding="utf-8") as handle:
        manifest = [row for row in csv.DictReader(handle)
                    if row["dataset"] == args.dataset and row["split"] == args.split]
    keep = wanted_ids(args.match)

    rows, missing = [], []
    for record in manifest:
        relative = record["path"].replace(BACKSLASH, "/")
        identifier = video_id(args.dataset, relative)
        if keep is not None and identifier not in keep:
            continue
        entry = metadata.get(relative)
        value = None if entry is None else entry.get(args.control)
        if value is None or not np.isfinite(float(value)):
            missing.append(identifier)
            continue
        label_real = int(record["label"])
        rows.append({
            "video_id": identifier, "video_path": "", "relative_path": relative,
            "dataset": args.dataset, "split": args.split,
            "label_real": label_real, "label_fake": 1 - label_real,
            "forgery_method": record.get("method", ""),
            "identity": record.get("target_id", ""),
            "source_identity": record.get("donor_id", ""),
            "source_family_id": record.get("source_clip", ""),
            "video_score": float(value), "num_clips": 0,
            "predictive_mean": float(value), "predictive_std": 0.0,
            "embedding_norm": "", "checkpoint": "",
            "experiment": "shortcut:{}".format(args.control), "seed": "",
        })
    if not rows:
        raise SystemExit("No video carried {!r}; check --control and --box-cache."
                         .format(args.control))
    expected = len(keep) if keep is not None else len(manifest)
    coverage = len(rows) / float(max(expected, 1))
    if keep is not None and coverage < args.min_coverage:
        raise SystemExit("Control coverage {:.1%} ({}/{}) is below --min-coverage "
                         "{:.1%}; inspect missing metadata/cache before comparing."
                         .format(coverage, len(rows), expected, args.min_coverage))

    labels = np.asarray([row["label_real"] for row in rows], dtype=np.int64)
    scores = np.asarray([row["video_score"] for row in rows], dtype=float)
    if len(np.unique(labels)) < 2:
        raise SystemExit("The scored control needs both real and fake videos.")
    auroc = float(roc_auc_score(1 - labels, scores))
    raw_auroc = auroc
    flipped = args.orient == "auto" and auroc < 0.5
    if flipped:
        for row in rows:
            row["video_score"] = -row["video_score"]
            row["predictive_mean"] = row["video_score"]
        auroc = 1.0 - auroc

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    provenance = {
        "control": args.control,
        "dataset": args.dataset,
        "split": args.split,
        "manifest": args.manifest,
        "video_sizes": args.video_sizes,
        "box_cache": args.box_cache,
        "matched_score_file": args.match,
        "orientation": args.orient,
        "orientation_selected_using_labels": args.orient == "auto",
        "sign_flipped": flipped,
        "raw_auroc": raw_auroc,
        "oriented_auroc": auroc,
        "videos_scored": len(rows),
        "videos_expected": expected,
        "coverage": coverage,
        "real_scored": int(labels.sum()),
        "fake_scored": int((1 - labels).sum()),
        "videos_missing_control": len(missing),
        "missing_video_ids_sample": missing[:10],
    }
    provenance_path = destination.with_name(destination.stem + "_provenance.json")
    with open(provenance_path, "w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2)

    print("{}: {} videos ({} real / {} fake), AUROC {:.4f}{}".format(
        args.control, len(rows), int(labels.sum()), int((1 - labels).sum()),
        auroc, ", sign flipped" if flipped else ""))
    if missing:
        print("{} video(s) carry no value for this control and were left out; "
              "compare_experiments.py then needs --intersect:".format(len(missing)))
        for item in missing[:5]:
            print("  {}".format(item))
    print("wrote {} and {}".format(destination, provenance_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
