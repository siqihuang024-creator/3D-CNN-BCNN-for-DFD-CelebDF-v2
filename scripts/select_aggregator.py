"""Select the E4-prime temporal aggregator using the two-round V2 rule."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from video_bcnn.reporting import json_safe
from video_bcnn.utils import save_json


def load_run(directory):
    directory = Path(directory)
    with open(directory / "config.json", "r", encoding="utf-8") as handle:
        config = json.load(handle)
    score_path = directory / "reports" / "val_scores.csv"
    with open(score_path, "r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    values = {(row.get("relative_path") or row["path"]): {
        "score": float(row["anomaly_score"]),
        "label": 1 - int(row["label_real"]),
        "identity": row["target_id"]} for row in rows}
    return {"directory": str(directory.resolve()),
            "aggregation": config["model"]["temporal_aggregation"],
            "seed": int(config.get("seed", 42)), "values": values}


def aligned(runs):
    paths = sorted(set.intersection(*(set(run["values"]) for run in runs)))
    labels = np.asarray([runs[0]["values"][path]["label"] for path in paths])
    identities = np.asarray([runs[0]["values"][path]["identity"] for path in paths])
    for run in runs[1:]:
        if any(run["values"][path]["label"] != labels[index] or
               run["values"][path]["identity"] != identities[index]
               for index, path in enumerate(paths)):
            raise ValueError("Aligned runs disagree on a video's label or identity.")
    scores = np.stack([[run["values"][path]["score"] for path in paths]
                       for run in runs])
    return paths, labels, identities, scores


def paired_identity_bootstrap(labels, candidate, baseline, identities,
                              draws=2000, seed=42):
    groups = defaultdict(list)
    for index, identity in enumerate(identities):
        groups[str(identity)].append(index)
    members = [np.asarray(groups[key]) for key in sorted(groups)]
    rng, differences = np.random.RandomState(seed), []
    for _ in range(int(draws)):
        selected = rng.randint(0, len(members), size=len(members))
        indices = np.concatenate([members[index] for index in selected])
        if len(np.unique(labels[indices])) < 2:
            continue
        differences.append(roc_auc_score(labels[indices], candidate[indices]) -
                           roc_auc_score(labels[indices], baseline[indices]))
    values = np.asarray(differences)
    return {"difference": float(roc_auc_score(labels, candidate) -
                                roc_auc_score(labels, baseline)),
            "low": float(np.quantile(values, 0.025)),
            "high": float(np.quantile(values, 0.975)),
            "draws": int(len(values)), "identity_clusters": len(members)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="artifacts/v2/aggregator_selection.json")
    args = parser.parse_args()
    runs = [load_run(path) for path in args.run_dirs]
    by_aggregation = defaultdict(dict)
    for run in runs:
        if run["seed"] in by_aggregation[run["aggregation"]]:
            raise ValueError("Duplicate {} seed {}.".format(run["aggregation"], run["seed"]))
        by_aggregation[run["aggregation"]][run["seed"]] = run
    if "gap" not in by_aggregation:
        raise ValueError("GAP runs are required as the paired baseline.")
    report = {"runs": [{key: run[key] for key in ("directory", "aggregation", "seed")}
                       for run in runs], "first_round_seed42": {},
              "three_seed": {}, "comparisons_to_gap": {}}
    for aggregation, seeds in sorted(by_aggregation.items()):
        if 42 in seeds:
            _, labels, _, scores = aligned([seeds[42]])
            report["first_round_seed42"][aggregation] = float(
                roc_auc_score(labels, scores[0]))
        aucs = {}
        for seed, run in sorted(seeds.items()):
            _, labels, _, scores = aligned([run])
            aucs[str(seed)] = float(roc_auc_score(labels, scores[0]))
        report["three_seed"][aggregation] = {
            "auroc_by_seed": aucs, "mean": float(np.mean(list(aucs.values()))),
            "std": float(np.std(list(aucs.values()), ddof=1)) if len(aucs) > 1 else None}
    ranking = sorted(((name, value) for name, value in
                      report["first_round_seed42"].items() if name != "gap"),
                     key=lambda item: item[1], reverse=True)
    report["first_round_top_two"] = [name for name, _ in ranking[:2]]
    report["missing_round_two"] = [
        name for name in report["first_round_top_two"]
        if not all(seed in by_aggregation[name] for seed in (42, 43, 44))]
    if report["missing_round_two"]:
        print("WARNING: round-two seeds 43/44 are missing for {}; run them before "
              "trusting the recommendation.".format(", ".join(report["missing_round_two"])))

    winners = []
    required = (42, 43, 44)
    for aggregation, seeds in sorted(by_aggregation.items()):
        if (aggregation == "gap" or aggregation not in report["first_round_top_two"]
                or not all(seed in seeds for seed in required)):
            continue
        if not all(seed in by_aggregation["gap"] for seed in required):
            raise ValueError("GAP needs seeds 42, 43 and 44 for round two.")
        candidate_runs = [seeds[seed] for seed in required]
        gap_runs = [by_aggregation["gap"][seed] for seed in required]
        paths, labels, identities, scores = aligned(candidate_runs + gap_runs)
        candidate_scores, gap_scores = scores[:3], scores[3:]
        candidate_aucs = [roc_auc_score(labels, values) for values in candidate_scores]
        gap_aucs = [roc_auc_score(labels, values) for values in gap_scores]
        wins = [left > right for left, right in zip(candidate_aucs, gap_aucs)]
        interval = paired_identity_bootstrap(
            labels, candidate_scores.mean(0), gap_scores.mean(0), identities,
            args.draws, args.seed)
        qualifies = (np.mean(candidate_aucs) > np.mean(gap_aucs)
                     and all(wins) and interval["low"] > 0)
        report["comparisons_to_gap"][aggregation] = {
            "aligned_videos": len(paths), "candidate_aurocs": candidate_aucs,
            "gap_aurocs": gap_aucs, "wins_by_seed": wins,
            "paired_identity_bootstrap": interval, "qualifies": bool(qualifies)}
        if qualifies:
            winners.append((float(np.mean(candidate_aucs)), aggregation))
    report["recommendation"] = (max(winners)[1] if winners else "gap")
    report["rule"] = ("non-GAP must have higher 3-seed mean, win all paired seeds, "
                      "and paired identity-bootstrap CI entirely above zero")
    save_json(args.output, json_safe(report))
    print(json.dumps(json_safe(report), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
