"""Relate model scores to container and face-crop metadata within class strata.

Metadata-only AUROC diagnoses dataset shortcuts. Spearman rho, p-value and n
are reported separately for real videos, fake videos and every manipulation;
an all-video correlation can be induced by the label and is deliberately not
used as evidence of model reliance.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
BACKSLASH = chr(92)
DATASET_ALIASES = {"DFD-Kaggle": "DFD"}


def _normalise_relative(relative, dataset):
    relative = str(relative).replace(BACKSLASH, "/")
    parts = relative.split("/")
    if len(parts) > 1 and DATASET_ALIASES.get(parts[0], parts[0]) == dataset:
        relative = "/".join(parts[1:])
    return relative


def read_video_sizes(path, dataset):
    """Map manifest-relative path to visible/container scalar controls."""
    table = {}
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            row_dataset = DATASET_ALIASES.get(row.get("dataset"), row.get("dataset"))
            if row.get("status") != "ok" or row_dataset != dataset:
                continue
            try:
                width, height = float(row["width"]), float(row["height"])
                frames = float(row["frame_count"])
                duration = float(row["duration_seconds"])
                megabytes = float(row["file_size_mb"])
            except (KeyError, TypeError, ValueError):
                continue
            if min(width, height, frames, duration) <= 0:
                continue
            table[_normalise_relative(row["relative_path"], dataset)] = {
                "bitrate_mb_per_second": megabytes / duration,
                "bits_per_pixel": (megabytes * 8e6) / (frames * width * height),
                "frame_width": width, "frame_height": height,
                "frame_pixels": width * height, "aspect_ratio": width / height,
                "frame_count": frames,
            }
    return table


def add_face_controls(table, directory, dataset):
    """Add median face width and width/frame-width share from the cache."""
    if not directory:
        return
    root = Path(directory) / dataset
    if not root.is_dir():
        return
    for item in root.rglob("*.npz"):
        relative = item.relative_to(root).as_posix()
        if relative.endswith(".npz"):
            relative = relative[:-4]
        entry = table.get(relative)
        if entry is None:
            continue
        try:
            with np.load(item) as payload:
                boxes = np.asarray(payload["boxes"])
        except (OSError, KeyError, ValueError):
            continue
        if not len(boxes):
            continue
        box_width = float(np.median(boxes[:, 2]))
        if np.isfinite(box_width) and box_width > 0:
            entry["face_box_width_px"] = box_width
            entry["face_box_share_of_frame"] = box_width / entry["frame_width"]


def association(scores, values):
    scores = np.asarray(scores, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(scores) & np.isfinite(values)
    if (int(mask.sum()) < 3 or len(np.unique(scores[mask])) < 2 or
            len(np.unique(values[mask])) < 2):
        return {"rho": None, "p_value": None, "n": int(mask.sum())}
    # Unpack rather than read attributes: scipy 1.7 names the field
    # correlation, newer scipy statistic, and both results unpack as a pair.
    rho, p_value = spearmanr(scores[mask], values[mask])
    return {"rho": float(rho), "p_value": float(p_value),
            "n": int(mask.sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True,
                        help="A legacy *_scores.csv or new *_video_scores.csv file.")
    parser.add_argument("--video-sizes", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--box-cache", default=None,
                        help="Detection-cache root; enables face_box_share_of_frame.")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    metadata = read_video_sizes(args.video_sizes, args.dataset)
    add_face_controls(metadata, args.box_cache, args.dataset)
    with open(args.scores, "r", newline="", encoding="utf-8") as handle:
        score_rows = list(csv.DictReader(handle))

    records = []
    for row in score_rows:
        relative = (row.get("relative_path") or row.get("video_path") or
                    row.get("path") or "")
        relative = relative.replace(BACKSLASH, "/").rsplit(args.dataset + "/", 1)[-1]
        entry = metadata.get(relative)
        if entry is None:
            continue
        score = row.get("video_score", row.get("anomaly_score"))
        method = row.get("forgery_method", row.get("method", ""))
        records.append({"score": float(score), "label_real": int(row["label_real"]),
                        "method": method, "controls": entry})
    if len(records) < 2:
        raise SystemExit("Only {} of {} score rows matched metadata.".format(
            len(records), len(score_rows)))

    labels = np.asarray([row["label_real"] for row in records], dtype=np.int64)
    scores = np.asarray([row["score"] for row in records], dtype=float)
    fake = 1 - labels
    report = {"scores": str(Path(args.scores).as_posix()), "dataset": args.dataset,
              "videos": len(records), "real": int(labels.sum()), "fake": int(fake.sum()),
              "model_auroc": float(roc_auc_score(fake, scores)), "controls": {}}
    names = sorted({name for row in records for name in row["controls"]})
    print("matched {} of {} scored videos ({} real / {} fake)".format(
        len(records), len(score_rows), int(labels.sum()), int(fake.sum())))
    print("model AUROC on matched subset: {:.4f}".format(report["model_auroc"]))

    for name in names:
        usable = [row for row in records if name in row["controls"]]
        control = np.asarray([row["controls"][name] for row in usable], dtype=float)
        usable_fake = np.asarray([1 - row["label_real"] for row in usable], dtype=np.int64)
        usable_scores = np.asarray([row["score"] for row in usable], dtype=float)
        auroc = (float(roc_auc_score(usable_fake, control))
                 if len(np.unique(usable_fake)) == 2 else None)
        methods = np.asarray([row["method"] for row in usable])
        masks = {"real_only": usable_fake == 0, "fake_only": usable_fake == 1}
        for method in sorted(set(methods[usable_fake == 1])):
            masks["method:{}".format(method)] = ((usable_fake == 1) & (methods == method))
        strata = {stratum: association(usable_scores[mask], control[mask])
                  for stratum, mask in masks.items()}
        report["controls"][name] = {
            "metadata_auroc": auroc,
            "orientation_adjusted": (max(auroc, 1.0 - auroc)
                                     if auroc is not None else None),
            "associations": strata,
        }

    destination = Path(args.output or
                       (ROOT / "results/diagnostics/metadata_association.json"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print("wrote {}".format(destination))
    print("Interpret correlations as monotonic associations, not causal reliance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
