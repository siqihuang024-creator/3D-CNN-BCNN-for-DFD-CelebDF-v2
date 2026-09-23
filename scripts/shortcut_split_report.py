"""Split the validation set by a fixed face-box metadata diagnostic.

A pooled AUROC over CelebDF++ mixes two populations that say different things.
On the manipulations whose videos are 256x256, the face fills half the frame,
the x2.0 crop leaves the frame, and replicate-padded borders reach the network:
there the face-box-share ranker alone scores 1.000, so a model's number on that
part cannot by itself establish forgery evidence. On the manipulations whose
videos match the real ones by this measure, other shortcuts may remain.

This script reads the shortcut's own score file, measures the control per
manipulation, splits the methods on that measurement, and reports every run on
both halves with paired identity-clustered intervals. The split is defined by
the control, never by a model's score. This is a post-hoc sensitivity analysis,
not an independent held-out test.

Usage:
    python scripts/shortcut_split_report.py \
        artifacts/v2/<E0>/reports/full_val_video_scores.csv \
        artifacts/v2/<E1>/reports/full_val_video_scores.csv \
        artifacts/v2/<E2>/reports/full_val_video_scores.csv \
        --labels E0 E1 E2 \
        --shortcut-scores results/v2/shortcut_facebox_scores.csv \
        --output results/v2/E0_E1_E2_shortcut_split.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from compare_experiments import (align_tables, choose_clusters,  # noqa: E402
                                 paired_cluster_bootstrap, ranking_metrics,
                                 read_score_file)
from video_bcnn.reporting import json_safe  # noqa: E402
from video_bcnn.utils import save_json  # noqa: E402


def control_by_method(table, neutral_band, separable_threshold):
    """Group manipulations by how well the control alone separates them."""
    rows = list(table.values())
    labels = np.asarray([row["label_real"] for row in rows], dtype=np.int64)
    scores = np.asarray([row["video_score"] for row in rows], dtype=float)
    methods = np.asarray([row.get("forgery_method", "") for row in rows])
    real = labels == 1
    if not real.any():
        raise ValueError("The control score file carries no real videos.")
    groups = {"neutral": [], "separable": [], "intermediate": []}
    per_method = {}
    for method in sorted(set(methods[labels == 0])):
        mask = real | ((labels == 0) & (methods == method))
        auroc = ranking_metrics(labels[mask], scores[mask])["auroc"]
        per_method[method] = {"control_auroc": auroc,
                              "n_fake": int(((labels == 0) & (methods == method)).sum())}
        if abs(auroc - 0.5) <= neutral_band:
            name = "neutral"
        elif auroc >= separable_threshold:
            name = "separable"
        else:
            name = "intermediate"
        groups[name].append(method)
        per_method[method]["group"] = name
    return groups, per_method


def validate_groups(groups, present):
    """Require the frozen groups to partition every evaluated fake method."""
    expected = {"neutral", "separable", "intermediate"}
    if set(groups) != expected:
        raise ValueError("Method groups must be exactly {}.".format(sorted(expected)))
    assigned = [method for name in sorted(groups) for method in groups[name]]
    if len(assigned) != len(set(assigned)):
        raise ValueError("A forgery method appears in multiple groups.")
    missing = sorted(present - set(assigned))
    extra = sorted(set(assigned) - present)
    if missing or extra:
        raise ValueError("Frozen split does not match evaluated methods: "
                         "missing={!r}, extra={!r}.".format(missing, extra))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("score_files", nargs="+",
                        help="full_val_video_scores.csv files, in ladder order")
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--shortcut-scores", default=None,
                        help="A shortcut_baseline_scores.py file; it defines the split.")
    parser.add_argument("--frozen-split", default=None,
                        help="A split written earlier (this script's own report, or "
                             "{'neutral': [...], 'separable': [...]}). Prefer it once "
                             "the groups are fixed: re-deriving them from every new "
                             "run would let the diagnostic subset drift with results.")
    parser.add_argument("--neutral-band", type=float, default=0.05,
                        help="|AUROC - 0.5| <= this counts as control-neutral.")
    parser.add_argument("--separable-threshold", type=float, default=0.95,
                        help="AUROC >= this counts as solved by the control alone.")
    parser.add_argument("--cluster-key", default="identity")
    parser.add_argument("--fallback-cluster-key", default="source_family_id")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/v2/shortcut_split.json")
    args = parser.parse_args()
    labels = args.labels or [Path(path).parent.parent.name for path in args.score_files]
    if len(labels) != len(args.score_files):
        parser.error("Pass one label per score file.")
    if len(set(labels)) != len(labels):
        parser.error("Labels must be unique.")

    if not (args.shortcut_scores or args.frozen_split):
        parser.error("Pass --shortcut-scores to derive the split, or --frozen-split "
                     "to reuse the fixed one.")
    tables = {name: read_score_file(path) for name, path in zip(labels, args.score_files)}
    video_ids, metadata, scores = align_tables(tables)
    if args.frozen_split:
        with open(args.frozen_split, "r", encoding="utf-8") as handle:
            frozen = json.load(handle)
        per_method = frozen.get("methods", {})
        groups = frozen.get("groups_by_name") or {
            name: sorted(method for method, item in per_method.items()
                         if item.get("group") == name)
            for name in ("neutral", "separable", "intermediate")}
        if not any(groups.values()):
            raise ValueError("{} carries no method groups.".format(args.frozen_split))
        split_source = {"frozen_split": args.frozen_split,
                        "derived_from": frozen.get("shortcut_scores"),
                        "split_rule": frozen.get("split_rule")}
    else:
        control = read_score_file(args.shortcut_scores)
        groups, per_method = control_by_method(control, args.neutral_band,
                                               args.separable_threshold)
        split_source = {"shortcut_scores": args.shortcut_scores,
                        "split_rule": {"neutral_band": args.neutral_band,
                                       "separable_threshold": args.separable_threshold}}
    present = {row.get("forgery_method", "") for row in metadata
               if row["label_real"] == 0}
    validate_groups(groups, present)

    label_real = np.asarray([row["label_real"] for row in metadata], dtype=np.int64)
    methods = np.asarray([row.get("forgery_method", "") for row in metadata])
    cluster_key, clusters = choose_clusters(metadata, args.cluster_key,
                                            args.fallback_cluster_key)
    report = {
        "score_files": dict(zip(labels, args.score_files)),
        "videos": len(video_ids), "cluster_key": cluster_key,
        "split_source": split_source,
        "shortcut_scores": args.shortcut_scores,
        "split_rule": split_source.get("split_rule"),
        "methods": per_method,
        "groups_by_name": {name: sorted(items) for name, items in groups.items()},
        "methods_without_group": [],
        "groups": {},
        "note": ("This post-hoc diagnostic split uses face-box share alone. "
                 "A neutral face-box ranker does not exclude other metadata or "
                 "preprocessing shortcuts. Neither group is an independent test."),
    }
    for name in ("neutral", "separable", "intermediate", "all"):
        keep = groups.get(name, [])
        if name == "all":
            mask = np.ones(len(label_real), dtype=bool)
        elif not keep:
            continue
        else:
            mask = (label_real == 1) | np.isin(methods, keep)
        subset = {run: values[mask] for run, values in scores.items()}
        entry = {
            "methods": sorted(keep) if name != "all" else "every method",
            "videos": int(mask.sum()),
            "real": int((label_real[mask] == 1).sum()),
            "fake": int((label_real[mask] == 0).sum()),
            "metrics": {run: ranking_metrics(label_real[mask], values)
                        for run, values in subset.items()},
        }
        if len(subset) > 1:
            entry["paired_cluster_bootstrap"] = paired_cluster_bootstrap(
                label_real[mask], subset, clusters[mask], args.draws, args.seed)
        report["groups"][name] = entry

    save_json(Path(args.output), json_safe(report))
    for name, entry in report["groups"].items():
        print("\n== {} ({} methods, {} videos: {} real / {} fake)".format(
            name, len(entry["methods"]) if isinstance(entry["methods"], list) else 22,
            entry["videos"], entry["real"], entry["fake"]))
        for run, values in entry["metrics"].items():
            print("   {:<6} AUROC {:.4f}  Macro-AP {:.4f}".format(
                run, values["auroc"], values["macro_ap_real_fake"]))
        for pair, deltas in entry.get("paired_cluster_bootstrap", {}).get("deltas", {}).items():
            interval = deltas["auroc"]
            print("   {:<10} dAUROC 95% CI [{:+.4f}, {:+.4f}]{}".format(
                pair, interval["low"], interval["high"],
                "" if interval["low"] > 0 or interval["high"] < 0 else "  (covers 0)"))
    print("\nwrote {}".format(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
