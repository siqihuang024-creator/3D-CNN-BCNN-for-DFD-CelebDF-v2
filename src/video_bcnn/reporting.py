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


def save_scores(values, output_path):
    fields = ["dataset", "path", "method", "target_id", "donor_id", "source_clip",
              "label_real", "anomaly_score", "predictive_mean", "predictive_std",
              "embedding_norm"]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        count = len(values["labels"])
        for index in range(count):
            def at(key, default=""):
                value = values.get(key, [])
                return value[index] if len(value) > index else default
            writer.writerow({"dataset": at("datasets"), "path": at("paths"),
                             "method": at("methods"), "target_id": at("target_ids"),
                             "donor_id": at("donor_ids"), "source_clip": at("source_clips"),
                             "label_real": int(values["labels"][index]),
                             "anomaly_score": float(values["scores"][index]),
                             "predictive_mean": float(values["means"][index]),
                             "predictive_std": float(values["stds"][index]),
                             "embedding_norm": at("embedding_norms")})


def save_evaluation_report(values, metrics, output_dir, split):
    output_dir = ensure_dir(output_dir)
    path = output_dir / "{}.json".format(split)
    save_json(path, json_safe(metrics))
    save_scores(values, output_dir / "{}_scores.csv".format(split))
    return path
