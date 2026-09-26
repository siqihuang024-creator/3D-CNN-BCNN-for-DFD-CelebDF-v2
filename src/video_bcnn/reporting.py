"""Persistence helpers for reproducible V2 experiments."""

import csv
import math
from pathlib import Path
import numpy as np
from .utils import ensure_dir, save_json


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _flatten(prefix, value, output):
    if isinstance(value, dict):
        for key, item in sorted(value.items()):
            _flatten("{}_{}".format(prefix, key) if prefix else str(key), item, output)
    elif not isinstance(value, (list, tuple)):
        output[prefix] = value


def save_history(history, output_dir):
    output_dir = ensure_dir(output_dir)
    safe = json_safe(history)
    save_json(output_dir / "history.json", safe)
    rows, fields = [], []
    for record in safe:
        flat = {}
        _flatten("", record, flat)
        rows.append(flat)
        for key in flat:
            if key not in fields:
                fields.append(key)
    with open(output_dir / "history.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _at(values, key, index, default=""):
    items = values.get(key, [])
    return items[index] if len(items) > index else default


def _video_id(values, index):
    return "{}::{}".format(_at(values, "datasets", index),
                           str(_at(values, "relative_paths", index,
                                   _at(values, "paths", index))).replace("\\", "/"))


def save_video_scores(values, output_path, context=None):
    """Write one stable, joinable row per scored video."""
    context = context or {}
    fields = ["video_id", "video_path", "relative_path", "dataset", "split",
              "label_real", "label_fake", "forgery_method", "identity",
              "source_identity", "source_family_id", "video_score", "num_clips",
              "predictive_mean", "predictive_std", "embedding_norm", "checkpoint",
              "experiment", "seed"]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        count = len(values["labels"])
        for index in range(count):
            clips = _at(values, "clip_scores", index, [])
            label_real = int(values["labels"][index])
            writer.writerow({
                "video_id": _video_id(values, index),
                "video_path": _at(values, "paths", index),
                "relative_path": _at(values, "relative_paths", index),
                "dataset": _at(values, "datasets", index),
                "split": context.get("split", ""),
                "label_real": label_real, "label_fake": 1 - label_real,
                "forgery_method": _at(values, "methods", index),
                "identity": _at(values, "target_ids", index),
                "source_identity": _at(values, "donor_ids", index),
                "source_family_id": _at(values, "source_clips", index),
                "video_score": float(values["scores"][index]),
                "num_clips": len(clips),
                "predictive_mean": float(values["means"][index]),
                "predictive_std": float(values["stds"][index]),
                "embedding_norm": _at(values, "embedding_norms", index),
                "checkpoint": context.get("checkpoint", ""),
                "experiment": context.get("experiment", ""),
                "seed": context.get("seed", ""),
            })


def save_clip_scores(values, output_path, context=None):
    """Write every clip prediction and its exact frame support."""
    context = context or {}
    fields = ["video_id", "video_path", "relative_path", "dataset", "split",
              "label_real", "label_fake", "forgery_method", "identity",
              "source_family_id", "clip_index", "clip_start_frame", "clip_end_frame",
              "clip_score", "predictive_mean", "predictive_std", "checkpoint",
              "experiment", "seed"]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for video_index in range(len(values["labels"])):
            clip_scores = np.asarray(_at(values, "clip_scores", video_index, []),
                                     dtype=float).reshape(-1)
            clip_means = np.asarray(_at(values, "clip_means", video_index,
                                        np.full(len(clip_scores), np.nan)),
                                    dtype=float).reshape(-1)
            clip_stds = np.asarray(_at(values, "clip_stds", video_index,
                                       np.zeros(len(clip_scores))),
                                   dtype=float).reshape(-1)
            frame_indices = np.asarray(_at(values, "clip_frame_indices", video_index,
                                           np.full((len(clip_scores), 0), -1)),
                                       dtype=np.int64)
            label_real = int(values["labels"][video_index])
            for clip_index, clip_score in enumerate(clip_scores):
                frames = (frame_indices[clip_index]
                          if frame_indices.ndim == 2 and clip_index < len(frame_indices)
                          else np.asarray([], dtype=np.int64))
                known = frames[frames >= 0]
                writer.writerow({
                    "video_id": _video_id(values, video_index),
                    "video_path": _at(values, "paths", video_index),
                    "relative_path": _at(values, "relative_paths", video_index),
                    "dataset": _at(values, "datasets", video_index),
                    "split": context.get("split", ""),
                    "label_real": label_real, "label_fake": 1 - label_real,
                    "forgery_method": _at(values, "methods", video_index),
                    "identity": _at(values, "target_ids", video_index),
                    "source_family_id": _at(values, "source_clips", video_index),
                    "clip_index": clip_index,
                    "clip_start_frame": int(known.min()) if len(known) else "",
                    "clip_end_frame": int(known.max()) if len(known) else "",
                    "clip_score": float(clip_score),
                    "predictive_mean": float(clip_means[clip_index]),
                    "predictive_std": float(clip_stds[clip_index]),
                    "checkpoint": context.get("checkpoint", ""),
                    "experiment": context.get("experiment", ""),
                    "seed": context.get("seed", ""),
                })


def save_scores(values, output_path):
    """Backward-compatible video score export used by older analysis scripts."""
    fields = ["dataset", "path", "relative_path", "method", "target_id", "donor_id",
              "source_clip", "label_real", "anomaly_score", "predictive_mean",
              "predictive_std", "embedding_norm"]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(len(values["labels"])):
            writer.writerow({"dataset": _at(values, "datasets", index),
                             "path": _at(values, "paths", index),
                             "relative_path": _at(values, "relative_paths", index),
                             "method": _at(values, "methods", index),
                             "target_id": _at(values, "target_ids", index),
                             "donor_id": _at(values, "donor_ids", index),
                             "source_clip": _at(values, "source_clips", index),
                             "label_real": int(values["labels"][index]),
                             "anomaly_score": float(values["scores"][index]),
                             "predictive_mean": float(values["means"][index]),
                             "predictive_std": float(values["stds"][index]),
                             "embedding_norm": _at(values, "embedding_norms", index)})


def save_evaluation_report(values, metrics, output_dir, split, report_name=None,
                           write_legacy=True):
    output_dir = ensure_dir(output_dir)
    report_name = str(report_name or split)
    path = output_dir / "{}.json".format(report_name)
    save_json(path, json_safe(metrics))
    context = {key: metrics.get(key, "") for key in
               ("split", "checkpoint", "experiment", "seed")}
    save_video_scores(values, output_dir / "{}_video_scores.csv".format(report_name),
                      context)
    save_clip_scores(values, output_dir / "{}_clip_scores.csv".format(report_name),
                     context)
    # Keep the legacy filename for select_aggregator.py and existing reports.
    if write_legacy:
        save_scores(values, output_dir / "{}_scores.csv".format(split))
    return path
