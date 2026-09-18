"""How much identity overlap survives a split that only separates target identities?

A forged video carries two identities: the target whose face is replaced, and
the donor whose face is put there. Splitting on the target alone leaves the
donor free to cross, so a test forgery can wear a face the training split has
already shown the network.

This counts that. For each dataset it rebuilds the split a target-only rule
would produce -- every forgery follows its target identity, including the ones
the donor rule discards -- and then asks, of those, how many carry a donor the
training split contains.

Two denominators, and they answer different questions. Reported over every
forgery, the rate is diluted by manipulations that have no donor at all:
CelebDF++'s TalkingFace family drives a single identity from audio, so there is
no second face to leak, and the donor rule never touches those videos. Reported
over the forgeries that do carry a donor, the rate says how often the rule
matters where it applies. The second is the honest headline; both are printed.

A third number is separate from both: how many forgeries the donor rule
discards. Not every discarded video would have leaked into training -- some
donors sit in validation -- so that figure is the protocol's cost in data, not
a leakage rate.

Usage:
    python scripts/split_audit.py \
        --manifest artifacts/manifests/combined_manifest_stage_a.csv \
        --output results/diagnostics/split_audit.json
"""

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")


def audit(rows, dataset):
    subset = [row for row in rows if row["dataset"] == dataset]
    # The split a target identity belongs to; excluded_donor rows carry the
    # target's own identity, so they rejoin it under a target-only rule.
    target_split = {}
    for row in subset:
        if row["split"] in SPLITS:
            target_split.setdefault(row["target_id"], row["split"])
    train_identities = {name for name, split in target_split.items() if split == "train"}

    counts = {split: {"target_only_fakes": 0, "with_donor": 0,
                      "donor_in_train": 0, "excluded_by_rule": 0}
              for split in SPLITS}
    for row in subset:
        if row["label"] != "0":
            continue
        split = target_split.get(row["target_id"])
        if split is None:
            continue
        entry = counts[split]
        entry["target_only_fakes"] += 1
        if row["donor_id"].strip():
            entry["with_donor"] += 1
            if row["donor_id"] in train_identities:
                entry["donor_in_train"] += 1
        if row["split"] == "excluded_donor":
            entry["excluded_by_rule"] += 1

    for entry in counts.values():
        total, donored = entry["target_only_fakes"], entry["with_donor"]
        entry["donor_in_train_share_of_all"] = (
            entry["donor_in_train"] / total if total else 0.0)
        entry["donor_in_train_share_of_donored"] = (
            entry["donor_in_train"] / donored if donored else 0.0)
        entry["excluded_share_of_all"] = entry["excluded_by_rule"] / total if total else 0.0
        entry["donor_safe_fakes"] = total - entry["excluded_by_rule"]
    return counts


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.manifest, "r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    report = {dataset: audit(rows, dataset)
              for dataset in sorted({row["dataset"] for row in rows})}

    for dataset, counts in report.items():
        print("==", dataset)
        print("   {:<6}{:>14}{:>12}{:>16}{:>14}{:>12}".format(
            "split", "target-only", "with donor", "donor in train", "of donored", "excluded"))
        for split in SPLITS:
            entry = counts[split]
            if not entry["target_only_fakes"]:
                continue
            print("   {:<6}{:>14}{:>12}{:>10} {:>5.1%}{:>13.1%}{:>7} {:>4.1%}".format(
                split, entry["target_only_fakes"], entry["with_donor"],
                entry["donor_in_train"], entry["donor_in_train_share_of_all"],
                entry["donor_in_train_share_of_donored"],
                entry["excluded_by_rule"], entry["excluded_share_of_all"]))
        print()

    destination = Path(args.output or (ROOT / "results/diagnostics" / "split_audit.json"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump({"manifest": str(Path(args.manifest).as_posix()), "datasets": report},
                  handle, indent=2)
    print("wrote {}".format(destination))
    print("The excluded count is the protocol's cost in data, not a leakage rate: a "
          "discarded forgery may have had its donor in validation rather than training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
