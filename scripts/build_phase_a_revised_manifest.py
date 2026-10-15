"""Build a separate Phase-A manifest with a capped fake training pool.

The original P0.5 manifest and completed experiments remain untouched.  The
validation and test source families are combined into one development set;
there is no independent test split in the resulting manifest.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def _rank(seed, row):
    token = "{}|{}|{}".format(seed, row["method"], row["path"])
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _method_quotas(groups, wanted):
    """Largest-remainder quotas preserve the original fake-method mixture."""
    total = sum(len(rows) for rows in groups.values())
    quotas = {name: wanted * len(rows) // total for name, rows in groups.items()}
    remainder = wanted - sum(quotas.values())
    priority = sorted(groups, key=lambda name: (-(wanted * len(groups[name]) % total), name))
    for name in priority[:remainder]:
        quotas[name] += 1
    return quotas


def build(rows, dataset, fake_per_real, seed):
    selected = [dict(row) for row in rows if row["dataset"] == dataset]
    if not selected or fake_per_real < 1:
        raise ValueError("Dataset must be present and fake_per_real must be positive.")
    if {row["split"] for row in selected} != {"train", "val", "test"}:
        raise ValueError("Expected train, val, and test in the source manifest.")
    train_real = [row for row in selected if row["split"] == "train" and row["class_name"] == "real"]
    train_fake = [row for row in selected if row["split"] == "train" and row["class_name"] == "fake"]
    wanted = len(train_real) * fake_per_real
    if wanted > len(train_fake):
        raise ValueError("Requested {} fake videos, but only {} are available.".format(wanted, len(train_fake)))

    groups = defaultdict(list)
    for row in train_fake:
        groups[row["method"]].append(row)
    quotas = _method_quotas(groups, wanted)
    chosen_fake = []
    source_counts = Counter()
    for method in sorted(groups):
        candidates = sorted(groups[method], key=lambda row: _rank(seed, row))
        for _ in range(quotas[method]):
            # Prefer a source not represented by another selected fake before
            # taking another manipulation of the same original video.
            choice = min(candidates, key=lambda row: source_counts[row["source_clip"]])
            chosen_fake.append(choice)
            source_counts[choice["source_clip"]] += 1
            candidates.remove(choice)

    held_out = [row for row in selected if row["split"] in ("val", "test")]
    output = train_real + chosen_fake + held_out
    for row in output:
        row["original_split"] = row["split"]
        row["source_protocol"] = row.get("protocol", "")
        row["protocol"] = "phase_a_pool_1to{}_valtest_merged".format(fake_per_real)
        if row["split"] == "test":
            row["split"] = "val"

    paths = [row["path"] for row in output]
    if len(paths) != len(set(paths)):
        raise AssertionError("A video path appears more than once.")
    families = defaultdict(set)
    for row in output:
        families[row["source_clip"]].add(row["split"])
    if any(len(splits) != 1 for splits in families.values()):
        raise AssertionError("A source-video family crosses train and validation.")
    counts = Counter((row["split"], row["class_name"]) for row in output)
    summary = {
        "dataset": dataset,
        "seed": seed,
        "fake_per_real_training_video": fake_per_real,
        "selection_rule": "method-proportional, source-diverse deterministic hash order",
        "validation_rule": "original val and test merged; no independent test",
        "counts": {"{}:{}".format(*key): value for key, value in sorted(counts.items())},
        "train_fake_method_counts": dict(sorted(Counter(row["method"] for row in chosen_fake).items())),
        "train_fake_source_families": len(source_counts),
        "source_family_overlap": 0,
    }
    return output, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", default="CelebDFv3")
    parser.add_argument("--fake-per-real", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    with open(args.input, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0]) if rows else []
    output, summary = build(rows, args.dataset, args.fake_per_real, args.seed)
    destination = Path(args.output)
    if destination.resolve() == Path(args.input).resolve():
        parser.error("--output must differ from --input; the original manifest is immutable.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + ["original_split", "source_protocol"])
        writer.writeheader()
        writer.writerows(output)
    summary["source_manifest"] = str(args.input)
    summary_path = destination.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
