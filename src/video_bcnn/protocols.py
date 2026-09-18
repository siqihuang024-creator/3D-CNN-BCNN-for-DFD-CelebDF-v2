"""Dataset split protocols used by the V2 experiments.

P0 is a video-level random control, P0.5 is the primary source-family split,
and P1 is the stricter legacy identity split.  P0.5 never discards a video:
the real source clip and every fake derived from it move as one family.
"""

import hashlib
from collections import Counter, defaultdict

import numpy as np

SPLITS = ("train", "val", "test")


def _rng(seed, *parts):
    text = "|".join([str(seed)] + [str(part) for part in parts])
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return np.random.RandomState(int.from_bytes(digest[:4], "little"))


def _partition(keys, train_ratio, val_ratio, seed, stratum):
    keys = sorted(keys)
    order = _rng(seed, stratum).permutation(len(keys))
    shuffled = [keys[int(index)] for index in order]
    count = len(shuffled)
    if count == 1:
        cuts = (1, 1)
    elif count == 2:
        cuts = (1, 1)
    else:
        train_end = max(1, min(count - 2, int(round(count * train_ratio))))
        val_end = max(train_end + 1, min(count - 1,
                                        int(round(count * (train_ratio + val_ratio)))))
        cuts = (train_end, val_end)
    return {key: ("train" if position < cuts[0] else
                  "val" if position < cuts[1] else "test")
            for position, key in enumerate(shuffled)}


def _source_category(family):
    real = [row for row in family if int(row["label"]) == 1]
    if len(real) != 1:
        raise ValueError("Every source family must contain exactly one real video; got {} for {}."
                         .format(len(real), family[0].get("source_clip")))
    method = real[0].get("method", "real")
    return method.split("/", 1)[-1] if real[0]["dataset"] == "CelebDFv3" else "all"


def assign_p05(rows, seed=42, train_ratio=0.70, val_ratio=0.15):
    """Assign source-video families, stratifying Celeb real sources separately."""
    families = defaultdict(list)
    for row in rows:
        source = row.get("source_clip", "")
        if not source:
            raise ValueError("P0.5 requires source_clip for every row: {}".format(row["path"]))
        families[(row["dataset"], source)].append(row)
    strata = defaultdict(list)
    for key, family in families.items():
        strata[(key[0], _source_category(family))].append(key)
    assignment = {}
    for stratum, keys in sorted(strata.items()):
        assignment.update(_partition(keys, train_ratio, val_ratio, seed, stratum))
    result = []
    for row in rows:
        copied = dict(row)
        copied["split"] = assignment[(row["dataset"], row["source_clip"])]
        copied["protocol"] = "p05"
        result.append(copied)
    return result


def assign_p0(rows, seed=42, train_ratio=0.70, val_ratio=0.15):
    """Video-level random split, stratified only by dataset and class."""
    result = [dict(row) for row in rows]
    strata = defaultdict(list)
    for index, row in enumerate(result):
        strata[(row["dataset"], row["class_name"])].append(index)
    for stratum, indices in sorted(strata.items()):
        mapping = _partition(indices, train_ratio, val_ratio, seed, stratum)
        for index in indices:
            result[index]["split"] = mapping[index]
            result[index]["protocol"] = "p0"
    return result


def assign_p1(rows, seed=42, train_ratio=0.70, val_ratio=0.15):
    """Identity-disjoint control; cross-split donor fakes are explicitly excluded."""
    identities = defaultdict(set)
    for row in rows:
        identities[row["dataset"]].add(row["target_id"])
        if row.get("donor_id"):
            identities[row["dataset"]].add(row["donor_id"])
    homes = {}
    for dataset, names in sorted(identities.items()):
        homes[dataset] = _partition(names, train_ratio, val_ratio, seed, dataset)
    result = []
    for row in rows:
        copied = dict(row)
        home = homes[row["dataset"]]
        target_split = home.get(row["target_id"])
        donor = row.get("donor_id", "")
        copied["split"] = ("excluded_donor" if donor and home.get(donor) != target_split
                           else target_split)
        copied["protocol"] = "p1"
        result.append(copied)
    return result


def assign_protocol(rows, protocol, seed=42, train_ratio=0.70, val_ratio=0.15):
    functions = {"p0": assign_p0, "p05": assign_p05, "p1": assign_p1}
    if protocol not in functions:
        raise ValueError("protocol must be p0, p05 or p1.")
    return functions[protocol](rows, seed, train_ratio, val_ratio)


def protocol_audit(rows, protocol):
    """Return machine-readable counts and leakage assertions."""
    counts = Counter((r["dataset"], r["split"], r["class_name"]) for r in rows)
    source_sets = {}
    for dataset in sorted(set(r["dataset"] for r in rows)):
        for split in SPLITS:
            source_sets[(dataset, split)] = {
                r["source_clip"] for r in rows
                if r["dataset"] == dataset and r["split"] == split}
    overlaps = {}
    for dataset in sorted(set(r["dataset"] for r in rows)):
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            overlaps["{}:{}/{}".format(dataset, left, right)] = len(
                source_sets[(dataset, left)] & source_sets[(dataset, right)])
    active = [r for r in rows if r["split"] in SPLITS]
    if protocol == "p05" and (len(active) != len(rows) or any(overlaps.values())):
        raise AssertionError("P0.5 must retain all rows with zero source-family overlap.")
    base_rates = {}
    for dataset in sorted(set(r["dataset"] for r in active)):
        for split in SPLITS:
            subset = [r for r in active if r["dataset"] == dataset and r["split"] == split]
            if subset:
                base_rates["{}:{}".format(dataset, split)] = {
                    "fake": sum(int(r["label"]) == 0 for r in subset) / float(len(subset)),
                    "real": sum(int(r["label"]) == 1 for r in subset) / float(len(subset)),
                }
    return {
        "protocol": protocol,
        "rows_total": len(rows),
        "rows_active": len(active),
        "counts": {"{}:{}:{}".format(*key): value for key, value in sorted(counts.items())},
        "source_family_overlap": overlaps,
        "base_rates": base_rates,
    }
