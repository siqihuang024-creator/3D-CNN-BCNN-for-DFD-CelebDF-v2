"""Keep the original CelebDF splits and build 60 disjoint fake-training groups.

The enforced leakage rule is exact video-path disjointness. Original train,
validation and test assignments are preserved; no source-family or identity
constraint is added by this builder. Existing manifests are never modified.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def _key(row):
    return row["dataset"], row["path"].replace("\\", "/").casefold()


def _assigned(row, split):
    copied = dict(row, split=split, protocol="video_disjoint_60_group")
    # This flag was computed for the source manifest's full training pool;
    # each new group trains on only 616 reals. Avoid publishing stale claims.
    if "donor_seen_in_train" in copied:
        copied["donor_seen_in_train"] = ""
    return copied


def _rank(seed, *parts):
    return hashlib.sha256("|".join(map(str, (seed,) + parts)).encode("utf-8")).digest()


def _ordered(rows, seed, *parts):
    return sorted(rows, key=lambda row: (_rank(seed, *parts, row["path"]), row["path"]))


def _quotas(groups, wanted):
    total = sum(map(len, groups.values()))
    if wanted > total or wanted < 0:
        raise ValueError("Cannot draw {} videos from a pool of {}.".format(wanted, total))
    quotas = {name: wanted * len(group) // total for name, group in groups.items()}
    remaining = wanted - sum(quotas.values())
    priority = sorted(groups, key=lambda name: (-(wanted * len(groups[name]) % total), name))
    for name in priority[:remaining]:
        quotas[name] += 1
    return quotas


def _draw_by_method(rows, wanted, seed, label):
    groups = defaultdict(list)
    for row in rows:
        groups[row["method"]].append(row)
    quotas = _quotas(groups, wanted)
    chosen, rest = [], []
    for method in sorted(groups):
        ordered = _ordered(groups[method], seed, label, method)
        chosen.extend(ordered[:quotas[method]])
        rest.extend(ordered[quotas[method]:])
    return chosen, rest


def _partition_fake_groups(train_fake, count, seed):
    if len(train_fake) % count:
        raise ValueError("The fake training pool must divide evenly among groups.")
    result = [[] for _ in range(count)]
    methods = defaultdict(list)
    for row in train_fake:
        methods[row["method"]].append(row)
    if any(len(rows) < count for rows in methods.values()):
        raise ValueError("Every fake method needs at least one video per group.")
    for method in sorted(methods):
        ordered = _ordered(methods[method], seed, "groups", method)
        base, extra = divmod(len(ordered), count)
        for index in range(count):
            result[index].extend(ordered[index * base:(index + 1) * base])
        offset = count * base
        # Give the remainders to the least-filled groups, with a seeded tie-break.
        priority = sorted(range(count), key=lambda index: (
            len(result[index]), _rank(seed, "remainder", method, index)))
        for position, index in enumerate(priority[:extra]):
            result[index].append(ordered[offset + position])
    sizes = {len(group) for group in result}
    if sizes != {len(train_fake) // count}:
        raise AssertionError("Fake groups are not equally sized: {}".format(sizes))
    return result


def build(rows, dataset="CelebDFv3", seed=42, groups=60, train_real_count=622,
          val_real_count=134, test_real_count=133, train_fake_count=36960,
          fake_per_group=616, excluded_real_paths=()):
    selected = [dict(row) for row in rows if row["dataset"] == dataset]
    keys = [_key(row) for row in selected]
    if len(keys) != len(set(keys)):
        raise ValueError("The input contains a repeated video path.")
    if groups < 1 or fake_per_group < 1 or train_fake_count != groups * fake_per_group:
        raise ValueError("Training fake count must equal groups × fake_per_group.")
    if train_real_count < fake_per_group:
        raise ValueError("Training real pool is smaller than one balanced group.")
    excluded = {path.replace("\\", "/").casefold() for path in excluded_real_paths}
    if {row["split"] for row in selected} != {"train", "val", "test"}:
        raise ValueError("Expected the original train, val and test assignments.")
    real = [row for row in selected if row["class_name"] == "real"]
    available_real = {row["path"].replace("\\", "/").casefold() for row in real}
    if excluded - available_real:
        raise ValueError("Unknown excluded real video paths: {}".format(sorted(excluded - available_real)))
    real = [row for row in real if row["path"].replace("\\", "/").casefold() not in excluded]
    by_split = {split: [row for row in selected if row["split"] == split]
                for split in ("train", "val", "test")}
    train_real = [row for row in real if row["split"] == "train"]
    val_real = [row for row in real if row["split"] == "val"]
    test_real = [row for row in real if row["split"] == "test"]
    expected_real = (train_real_count, val_real_count, test_real_count)
    observed_real = tuple(map(len, (train_real, val_real, test_real)))
    if observed_real != expected_real:
        raise ValueError("Expected usable real train/val/test counts {}, found {}.".format(
            expected_real, observed_real))
    train_fake = [row for row in by_split["train"] if row["class_name"] == "fake"]
    if len(train_fake) != train_fake_count:
        raise ValueError("Expected {} original training fakes, found {}.".format(
            train_fake_count, len(train_fake)))
    val_fake, unused_val_fake = _draw_by_method(
        [row for row in by_split["val"] if row["class_name"] == "fake"],
        val_real_count, seed, "val")
    test_fake, unused_test_fake = _draw_by_method(
        [row for row in by_split["test"] if row["class_name"] == "fake"],
        test_real_count, seed, "test")
    unused_fake = unused_val_fake + unused_test_fake
    fake_groups = _partition_fake_groups(train_fake, groups, seed)
    validation = [_assigned(row, "val") for row in val_real + val_fake]
    test = [_assigned(row, "test") for row in test_real + test_fake]
    manifests = []
    for index, group_fake in enumerate(fake_groups, 1):
        # The real pool is reused, with a new seeded draw of 616/622 in each run.
        group_real = _ordered(train_real, seed, "training-real", index)[:fake_per_group]
        train = [_assigned(row, "train") for row in group_real + group_fake]
        manifest = train + validation + test
        if len({_key(row) for row in manifest}) != len(manifest):
            raise AssertionError("A video crosses train, val, or test in group {}.".format(index))
        if Counter(row["class_name"] for row in train) != {"real": fake_per_group,
                                                            "fake": fake_per_group}:
            raise AssertionError("Training group {} is not balanced.".format(index))
        manifests.append(manifest)
    if len({_key(row) for group in fake_groups for row in group}) != train_fake_count:
        raise AssertionError("A fake training video appears in multiple groups.")
    if any(len({row["method"] for row in group}) != len({row["method"] for row in train_fake})
           for group in fake_groups):
        raise AssertionError("A training group lacks a forgery method.")
    family_overlap = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        counts = []
        for manifest in manifests:
            left_families = {row["source_clip"] for row in manifest if row["split"] == left}
            right_families = {row["source_clip"] for row in manifest if row["split"] == right}
            counts.append(len(left_families & right_families))
        family_overlap["{}/{}".format(left, right)] = {"min": min(counts), "max": max(counts)}
    summary = {
        "protocol": "video_disjoint_60_group",
        "dataset": dataset, "seed": seed, "groups": groups,
        "rule": "original train/val/test assignments retained; exact video paths disjoint; no additional source-family or identity constraint",
        "counts": {"train_real_pool": len(train_real), "train_fake_pool": len(train_fake),
                   "per_group_real": fake_per_group, "per_group_fake": fake_per_group,
                   "val_real": len(val_real), "val_fake": len(val_fake),
                   "test_real": len(test_real), "test_fake": len(test_fake),
                   "unused_fake": len(unused_fake)},
        "excluded_real_paths": sorted(excluded_real_paths),
        "fake_methods": sorted({row["method"] for row in train_fake}),
        "method_counts_by_group": [dict(sorted(Counter(row["method"] for row in group).items()))
                                   for group in fake_groups],
        "source_family_overlap_permitted_by_rule": True,
        "source_family_overlap_counts": family_overlap,
        "validation_paths_sha256": hashlib.sha256("\n".join(sorted(row["path"] for row in validation)).encode()).hexdigest(),
        "test_paths_sha256": hashlib.sha256("\n".join(sorted(row["path"] for row in test)).encode()).hexdigest(),
    }
    return manifests, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-real-path", action="append", default=["REAL/Real_WEB/id27_0005.mp4"])
    args = parser.parse_args()
    source = Path(args.input)
    destination = Path(args.output_dir)
    if source.resolve().is_relative_to(destination.resolve()) or destination.resolve().is_relative_to(source.resolve()):
        raise ValueError("The input manifest must not be inside the output directory.")
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    manifests, summary = build(rows, seed=args.seed, excluded_real_paths=args.exclude_real_path)
    summary["source_manifest_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    destination.mkdir(parents=True, exist_ok=True)
    for index, manifest in enumerate(manifests, 1):
        path = destination / "group_{:02d}.csv".format(index)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(manifest)
    required = {_key(row): row for manifest in manifests for row in manifest}
    if len(required) != (summary["counts"]["train_real_pool"] +
                         summary["counts"]["train_fake_pool"] +
                         summary["counts"]["val_real"] + summary["counts"]["val_fake"] +
                         summary["counts"]["test_real"] + summary["counts"]["test_fake"]):
        raise AssertionError("The all-required manifest is missing a training-pool video.")
    with (destination / "all_required.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(required.values(), key=lambda row: (row["split"], row["path"])))
    summary["source_manifest"] = str(source)
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items()
                      if key != "method_counts_by_group"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
