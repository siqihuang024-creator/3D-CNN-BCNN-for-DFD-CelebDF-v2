"""Strict paired comparison of full-validation video score files.

Every run must contain exactly the same video IDs.  Bootstrap resampling is
paired across runs and clustered by identity, the plan's primary unit (falling
back to source family), so an identity drawn twice contributes all of its
videos twice to every model.  Identity clusters are coarser than source
families -- one person has several source videos -- so they give the wider,
honest interval; pass --cluster-key source_family_id for the secondary one.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from video_bcnn.reporting import json_safe
from video_bcnn.utils import save_json


def _identifier(row):
    value = row.get("video_id")
    if value:
        return value
    relative = (row.get("relative_path") or row.get("video_path") or
                row.get("path") or "").replace("\\", "/")
    return "{}::{}".format(row.get("dataset", ""), relative)


def read_score_file(path):
    with open(path, "r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    table = {}
    for row in rows:
        video_id = _identifier(row)
        if not video_id or video_id in table:
            raise ValueError("Missing or duplicate video_id {!r} in {}.".format(
                video_id, path))
        row = dict(row)
        row["video_id"] = video_id
        row["label_real"] = int(row["label_real"])
        score = row.get("video_score", row.get("anomaly_score"))
        row["video_score"] = float(score)
        # Accept both the new explicit names and legacy score exports.
        row.setdefault("identity", row.get("target_id", ""))
        row.setdefault("source_family_id", row.get("source_clip", ""))
        row.setdefault("forgery_method", row.get("method", ""))
        table[video_id] = row
    if not table:
        raise ValueError("No score rows in {}.".format(path))
    return table


def align_tables(tables, intersect=False):
    """Return aligned metadata and scores over one identical set of videos.

    Differing ID sets are an error by default, because a model run that quietly
    skipped videos would otherwise be compared on a different population. The
    one legitimate exception is a metadata control that cannot score every
    video, so `intersect` drops the difference and says how much it dropped.
    """
    names = list(tables)
    reference = set(tables[names[0]])
    if intersect:
        shared = set.intersection(*(set(tables[name]) for name in names))
        dropped = {name: len(set(tables[name]) - shared) for name in names}
        if not shared:
            raise ValueError("The score files share no video IDs.")
        print("comparing on {} shared videos; dropped per file: {}".format(
            len(shared), dropped))
        reference = shared
    mismatches = {}
    for name in names[1:]:
        current = set(tables[name])
        if not intersect and current != reference:
            mismatches[name] = {
                "missing": sorted(reference - current)[:10],
                "extra": sorted(current - reference)[:10],
            }
    if mismatches:
        raise ValueError("Score files do not contain identical video IDs: {}".format(
            mismatches))
    video_ids = sorted(reference)
    first = tables[names[0]]
    for name in names[1:]:
        for video_id in video_ids:
            if tables[name][video_id]["label_real"] != first[video_id]["label_real"]:
                raise ValueError("Runs disagree on label for {}.".format(video_id))
    metadata = [first[video_id] for video_id in video_ids]
    scores = {name: np.asarray([tables[name][video_id]["video_score"]
                                for video_id in video_ids], dtype=float)
              for name in names}
    return video_ids, metadata, scores


def alignment_diagnostics(tables, video_ids):
    """Record how much of each original score file entered the comparison."""
    shared = set(video_ids)
    id_sets = {name: set(table) for name, table in tables.items()}
    reference = next(iter(id_sets.values()))
    files = {}
    for name, table in tables.items():
        dropped = id_sets[name] - shared
        files[name] = {
            "videos_original": len(table),
            "videos_dropped": len(dropped),
            "real_dropped": sum(table[video_id]["label_real"] == 1
                                for video_id in dropped),
            "fake_dropped": sum(table[video_id]["label_real"] == 0
                                for video_id in dropped),
            "dropped_video_ids_sample": sorted(dropped)[:10],
        }
    return {
        "video_ids_identical": all(ids == reference for ids in id_sets.values()),
        "videos_compared": len(shared),
        "files": files,
    }


def choose_clusters(rows, preferred="identity", fallback="source_family_id"):
    for key in (preferred, fallback):
        values = [str(row.get(key, "")).strip() for row in rows]
        if all(values):
            return key, np.asarray(values)
    raise ValueError("Neither cluster key {!r} nor fallback {!r} is complete.".format(
        preferred, fallback))


def ranking_metrics(label_real, scores):
    label_real = np.asarray(label_real, dtype=np.int64)
    fake = 1 - label_real
    if np.unique(fake).size < 2:
        return {"auroc": float("nan"), "ap_fake": float("nan"),
                "ap_real": float("nan"), "macro_ap_real_fake": float("nan")}
    ap_fake = float(average_precision_score(fake, scores))
    ap_real = float(average_precision_score(label_real, -np.asarray(scores)))
    return {"auroc": float(roc_auc_score(fake, scores)),
            "ap_fake": ap_fake, "ap_real": ap_real,
            "macro_ap_real_fake": 0.5 * (ap_fake + ap_real)}


def paired_cluster_bootstrap(label_real, scores, clusters, draws=2000, seed=42):
    groups = defaultdict(list)
    for index, cluster in enumerate(clusters):
        groups[str(cluster)].append(index)
    members = [np.asarray(groups[key], dtype=np.int64) for key in sorted(groups)]
    if not members:
        raise ValueError("No bootstrap clusters.")
    rng = np.random.default_rng(seed)
    samples = {name: {"auroc": [], "macro_ap_real_fake": []} for name in scores}
    pairs = [(left, right) for index, left in enumerate(scores)
             for right in list(scores)[index + 1:]]
    deltas = {"{}-{}".format(right, left):
              {"auroc": [], "macro_ap_real_fake": []}
              for left, right in pairs}
    valid = 0
    for _ in range(int(draws)):
        selected = rng.integers(0, len(members), size=len(members))
        indices = np.concatenate([members[index] for index in selected])
        if len(np.unique(np.asarray(label_real)[indices])) < 2:
            continue
        current = {name: ranking_metrics(np.asarray(label_real)[indices], values[indices])
                   for name, values in scores.items()}
        for name, metrics in current.items():
            for metric in samples[name]:
                samples[name][metric].append(metrics[metric])
        for left, right in pairs:
            key = "{}-{}".format(right, left)
            for metric in deltas[key]:
                deltas[key][metric].append(current[right][metric] - current[left][metric])
        valid += 1

    def interval(values):
        values = np.asarray(values, dtype=float)
        if not len(values):
            return {"mean": None, "low": None, "high": None}
        return {"mean": float(values.mean()),
                "low": float(np.quantile(values, 0.025)),
                "high": float(np.quantile(values, 0.975))}

    return {
        "clusters": len(members), "draws_requested": int(draws),
        "draws_valid": valid,
        "models": {name: {metric: interval(values)
                           for metric, values in metrics.items()}
                   for name, metrics in samples.items()},
        "deltas": {pair: {metric: interval(values)
                           for metric, values in metrics.items()}
                   for pair, metrics in deltas.items()},
    }


def manipulation_rows(metadata, scores):
    labels = np.asarray([row["label_real"] for row in metadata], dtype=np.int64)
    methods = np.asarray([row.get("forgery_method", "") for row in metadata])
    real = labels == 1
    rows = []
    for method in sorted(set(methods[labels == 0])):
        mask = real | ((labels == 0) & (methods == method))
        row = {"method": method, "n_real": int(real.sum()),
               "n_fake": int(((labels == 0) & (methods == method)).sum())}
        for name, values in scores.items():
            metrics = ranking_metrics(labels[mask], values[mask])
            row["{}_auroc".format(name)] = metrics["auroc"]
            row["{}_macro_ap".format(name)] = metrics["macro_ap_real_fake"]
        names = list(scores)
        for left, right in zip(names, names[1:]):
            row["delta_{}_{}_auroc".format(right, left)] = (
                row["{}_auroc".format(right)] - row["{}_auroc".format(left)])
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("score_files", nargs="+",
                        help="full_val_video_scores.csv files in comparison order")
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--cluster-key", default="identity")
    parser.add_argument("--fallback-cluster-key", default="source_family_id")
    parser.add_argument("--sensitivity-cluster-key", default=None,
                        help="Second paired bootstrap clustering. Defaults to "
                             "source_family_id when identity is primary, or "
                             "identity otherwise.")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-videos", type=int, default=None,
                        help="Require this many shared scored videos (e.g. 8205 for "
                             "CelebDFv3 full validation).")
    parser.add_argument("--intersect", action="store_true",
                        help="Compare on the shared videos instead of requiring "
                             "identical ID sets. For a metadata control that "
                             "cannot score every video, not for model runs.")
    parser.add_argument("--output", default="results/v2/full_val_comparison.json")
    args = parser.parse_args()
    labels = args.labels or [Path(path).parent.parent.name for path in args.score_files]
    if len(labels) != len(args.score_files) or len(labels) < 2:
        parser.error("Provide at least two score files and one label per file.")
    if len(set(labels)) != len(labels):
        parser.error("Comparison labels must be unique.")

    tables = {name: read_score_file(path) for name, path in zip(labels, args.score_files)}
    video_ids, metadata, scores = align_tables(tables, args.intersect)
    if args.expected_videos is not None and len(video_ids) != args.expected_videos:
        raise ValueError("Expected {} paired videos, found {}.".format(
            args.expected_videos, len(video_ids)))
    alignment = alignment_diagnostics(tables, video_ids)
    label_real = np.asarray([row["label_real"] for row in metadata], dtype=np.int64)
    cluster_key, clusters = choose_clusters(
        metadata, args.cluster_key, args.fallback_cluster_key)
    sensitivity_key = (args.sensitivity_cluster_key or
                       ("source_family_id" if cluster_key == "identity" else "identity"))
    sensitivity_values = [str(row.get(sensitivity_key, "")).strip()
                          for row in metadata]
    if all(sensitivity_values) and sensitivity_key != cluster_key:
        sensitivity_report = paired_cluster_bootstrap(
            label_real, scores, np.asarray(sensitivity_values), args.draws, args.seed)
    else:
        sensitivity_report = None
    point = {name: ranking_metrics(label_real, values) for name, values in scores.items()}
    ordered = list(scores)
    point_deltas = {}
    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1:]:
            point_deltas["{}-{}".format(right, left)] = {
                metric: point[right][metric] - point[left][metric]
                for metric in ("auroc", "macro_ap_real_fake")
            }
    report = {
        "score_files": dict(zip(labels, args.score_files)),
        "videos": len(video_ids),
        "video_ids_identical": alignment["video_ids_identical"],
        "compared_on": ("shared subset" if not alignment["video_ids_identical"]
                        else "identical id sets"),
        "alignment": alignment,
        "cluster_key": cluster_key,
        "sensitivity_cluster_key": (sensitivity_key if sensitivity_report is not None
                                    else None),
        "sensitivity_paired_cluster_bootstrap": sensitivity_report,
        "metrics": point,
        "point_deltas": point_deltas,
        "paired_cluster_bootstrap": paired_cluster_bootstrap(
            label_real, scores, clusters, args.draws, args.seed),
        "manipulation_level": manipulation_rows(metadata, scores),
        "warning": ("Manipulation-level changes are descriptive; they do not by "
                    "themselves establish a temporal-signal mechanism."),
    }
    destination = Path(args.output)
    save_json(destination, json_safe(report))
    method_path = destination.with_name(destination.stem + "_methods.csv")
    method_path.parent.mkdir(parents=True, exist_ok=True)
    method_rows = report["manipulation_level"]
    if method_rows:
        fields = list(method_rows[0])
        with open(method_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(method_rows)
    print(json.dumps(json_safe({"videos": report["videos"],
                                "cluster_key": cluster_key,
                                "sensitivity_cluster_key": report["sensitivity_cluster_key"],
                                "metrics": point,
                                "point_deltas": point_deltas,
                                "deltas": report["paired_cluster_bootstrap"]["deltas"],
                                "sensitivity_deltas": (sensitivity_report["deltas"]
                                                       if sensitivity_report else None)}),
                     indent=2))
    print("wrote {} and {}".format(destination, method_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
