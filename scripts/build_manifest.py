"""Build identity-aware combined DFD and CelebDFv3 manifests."""

import argparse
import csv
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_bcnn.utils import DATASET_ROOT_ENV_VARS
from video_bcnn.protocols import assign_protocol, protocol_audit


DFD_REAL_DIR = "DFD_original sequences"
DFD_FAKE_DIR = "DFD_manipulated_sequences/DFD_manipulated_sequences"
CELEB_REAL_DIR = "REAL"
CELEB_FAKE_DIR = "FAKE"
DFD_ID_PATTERN = re.compile(r"^(\d+)")
# Google/Jigsaw document the scheme in the FaceForensics repo as
# "<target actor>_<source actor>__<sequence name>__<8 charactor long
# experiment id>", so group 1 supplies the frames and group 2 the face.
DFD_FAKE_PATTERN = re.compile(r"^(\d+)_(\d+)__(.+?)__[^_]+$")
DFD_REAL_SCENE_PATTERN = re.compile(r"^(\d+)__(.+)$")
CELEB_REAL_PATTERN = re.compile(r"^(id\d+)_(\d+)$")
CELEB_FAKE_PATTERN = re.compile(r"^(id\d+)_(id\d+)_(\d+)$")
# Talking-face outputs name only the identity whose face appears; the trailing
# field is a VoxCeleb2 audio clip, which Celeb-DF++ describes as "randomly
# select 5 audio segments from the VoxCeleb2 dataset for each video frame to
# drive the synthesis". It carries no second face, so the donor rule cannot
# apply to it -- but it is recorded so its cross-split reuse stays measurable.
CELEB_TALKING_PATTERN = re.compile(r"^(id\d+)_(\d+)_(.+)$")


def celeb_fake_parts(stem):
    """Return (target, donor, source_clip, driver) for one CelebDFv3 fake."""
    match = CELEB_FAKE_PATTERN.match(stem)
    if match:
        target, donor, clip = match.groups()
        return target, donor, "{}_{}".format(target, clip), ""
    match = CELEB_TALKING_PATTERN.match(stem)
    if match:
        target, clip, driver = match.groups()
        return target, "", "{}_{}".format(target, clip), driver
    raise ValueError("Unexpected CelebDFv3 fake filename: {}.mp4".format(stem))


def dfd_fake_parts(stem):
    """Return (target, donor, source_clip) for one DFD fake."""
    match = DFD_FAKE_PATTERN.match(stem)
    if match is None:
        raise ValueError("Unexpected DFD fake filename: {}.mp4".format(stem))
    target, donor, scene = match.groups()
    return target, donor, "{}__{}".format(target, scene)


def split_identities(identity_ids, seed, train_ratio, val_ratio):
    identity_ids = sorted(identity_ids)
    random.Random(seed).shuffle(identity_ids)
    train_end = max(1, int(round(len(identity_ids) * train_ratio)))
    val_end = max(train_end + 1, int(round(len(identity_ids) * (train_ratio + val_ratio))))
    val_end = min(val_end, len(identity_ids) - 1)
    assignments = {}
    for identity_id in identity_ids[:train_end]:
        assignments[identity_id] = "train"
    for identity_id in identity_ids[train_end:val_end]:
        assignments[identity_id] = "val"
    for identity_id in identity_ids[val_end:]:
        assignments[identity_id] = "test"
    return assignments


def make_row(dataset, path, root, label, split, target_id, method,
             donor_id="", donor_seen_in_train="", source_clip="", driver_id=""):
    return {
        "dataset": dataset,
        "path": path.relative_to(root).as_posix(),
        "label": label,
        "class_name": "real" if label else "fake",
        "split": split,
        "source_id": "{}:{}".format(dataset, target_id),
        "target_id": target_id,
        "donor_id": donor_id,
        "donor_seen_in_train": donor_seen_in_train,
        # The real clip a fake was generated from. Every fake traces to one, and
        # a real clip yields a median of 84 fakes on CelebDFv3 and 9 on DFD, so
        # videos are not independent samples: confidence intervals have to
        # resample this column, not rows (Obuchowski, Biometrics 1997).
        "source_clip": source_clip,
        "driver_id": driver_id,
        "method": method,
    }


def dfd_rows(root, seed, train_ratio, val_ratio):
    real_paths = sorted((root / DFD_REAL_DIR).glob("*.mp4"))
    fake_paths = sorted((root / DFD_FAKE_DIR).glob("*.mp4"))
    if not real_paths or not fake_paths:
        raise FileNotFoundError("DFD MP4 files were not found below {}".format(root))
    real_ids = {DFD_ID_PATTERN.match(path.name).group(1) for path in real_paths}
    assignments = split_identities(real_ids, seed, train_ratio, val_ratio)
    rows = []
    for path in real_paths:
        target_id = DFD_ID_PATTERN.match(path.name).group(1)
        rows.append(make_row(
            "DFD", path, root, 1, assignments[target_id], target_id, "real",
            source_clip=path.stem,
        ))
    train_ids = {identity for identity, split in assignments.items() if split == "train"}
    for path in fake_paths:
        target_id, donor_id, source_clip = dfd_fake_parts(path.stem)
        split = assignments.get(target_id, "unused")
        if split == "train":
            split = "unused"
        rows.append(make_row(
            "DFD", path, root, 0, split, target_id, "DeepFakeDetection",
            donor_id, str(donor_id in train_ids).lower(), source_clip,
        ))
    return rows, assignments


def donor_safe_dfd(rows, assignments):
    """Exclude DFD val/test fakes whose donor appeared in DFD real training."""
    train_ids = {identity for identity, split in assignments.items() if split == "train"}
    result = []
    for item in rows:
        copied = dict(item)
        if (
            copied["dataset"] == "DFD"
            and int(copied["label"]) == 0
            and copied["split"] in ("val", "test")
            and copied["donor_id"] in train_ids
        ):
            copied["split"] = "excluded_donor_train"
        result.append(copied)
    return result


def stage_a_rows(rows, assignments):
    """Apply the strict donor rule and restore fakes to the training split.

    FaceForensics++ publishes its splits as identity *pairs* (360/70/70), and
    all three pairwise overlaps are zero once both roles are counted. A forged
    video carries two subjects -- the one supplying the frames and the one
    supplying the face -- so it belongs to a split only when both live there.
    That is stricter than the existing manifests, which only kept training
    donors out of evaluation and left 15 CelebDFv3 and 5 DFD identities
    crossing val to test.

    Stage A is supervised, so training fakes are needed: the parked "unused"
    rows come back as "train" whenever both of their subjects are training
    identities.
    """
    result = []
    for item in rows:
        copied = dict(item)
        home = assignments[copied["dataset"]]
        target_split = home.get(copied["target_id"])
        if target_split is None:
            copied["split"] = "excluded_unknown_identity"
        elif int(copied["label"]) == 1:
            copied["split"] = target_split
        elif copied["donor_id"] and home.get(copied["donor_id"]) != target_split:
            copied["split"] = "excluded_donor"
        else:
            copied["split"] = target_split
        result.append(copied)
    return result


def celeb_rows(root, seed, train_ratio, val_ratio,
               celeb_train_ratio=None, celeb_val_ratio=None):
    """Rows for CelebDFv3, with the two real sources stratified separately.

    Only the 59 celebrities carry forgeries; the 300 Real_YouTube subjects are
    one video each and have none. Shuffling all 359 together lets the fake-less
    subjects consume evaluation places -- that is how the celebrities landed at
    40/7/12 and validation ended up with seven usable identities. Stratifying
    keeps every celebrity slot for a subject that can actually be evaluated.
    """
    real_root, fake_root = root / CELEB_REAL_DIR, root / CELEB_FAKE_DIR
    real_paths = sorted(real_root.rglob("*.mp4"))
    fake_paths = sorted(fake_root.rglob("*.mp4"))
    if not real_paths or not fake_paths:
        raise FileNotFoundError("CelebDFv3 MP4 files were not found below {}".format(root))

    def real_parts(path):
        match = CELEB_REAL_PATTERN.match(path.stem)
        if match:
            return match.group(1), path.stem
        if path.parent.name == "Real_YouTube" and path.stem.isdigit():
            return "youtube_{}".format(path.stem), path.stem
        raise ValueError("Unexpected CelebDFv3 real filename: {}".format(path.name))

    real_ids = {real_parts(path)[0] for path in real_paths}
    celebrities = {value for value in real_ids if not value.startswith("youtube_")}
    assignments = split_identities(
        celebrities, seed,
        train_ratio if celeb_train_ratio is None else celeb_train_ratio,
        val_ratio if celeb_val_ratio is None else celeb_val_ratio,
    )
    assignments.update(split_identities(
        real_ids - celebrities, seed, train_ratio, val_ratio))
    rows = []
    for path in real_paths:
        target_id, source_clip = real_parts(path)
        rows.append(make_row(
            "CelebDFv3", path, root, 1, assignments[target_id], target_id,
            "real/{}".format(path.parent.name), source_clip=source_clip,
        ))
    for path in fake_paths:
        target_id, donor_id, source_clip, driver_id = celeb_fake_parts(path.stem)
        if target_id not in assignments:
            raise ValueError("CelebDFv3 fake target has no real identity: {}".format(path.name))
        split = assignments[target_id]
        if split == "train":
            split = "unused"
        method = "/".join(path.relative_to(fake_root).parts[:-1]) or "CelebDFv3"
        rows.append(make_row(
            "CelebDFv3", path, root, 0, split, target_id, method, donor_id,
            source_clip=source_clip, driver_id=driver_id,
        ))
    return rows, assignments


def audit(rows, assignments):
    """Numbers the split has to be checked against, not just counted by."""
    report = {}
    for dataset, home in assignments.items():
        subset = [item for item in rows if item["dataset"] == dataset]
        members = {}
        for split in ("train", "val", "test"):
            members[split] = {
                identity for identity, value in home.items() if value == split}
        overlaps = {}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            crossing = set()
            for item in subset:
                if item["split"] not in (a, b) or not item["donor_id"]:
                    continue
                if home.get(item["donor_id"]) != item["split"]:
                    crossing.add(item["donor_id"])
            overlaps["{}/{}".format(a, b)] = len(crossing)
        per_split = {}
        for split in ("train", "val", "test"):
            here = [item for item in subset if item["split"] == split]
            reals = [item for item in here if int(item["label"]) == 1]
            fakes = [item for item in here if int(item["label"]) == 0]
            with_real = {item["target_id"] for item in reals}
            with_fake = {item["target_id"] for item in fakes}
            per_split[split] = {
                "identities": len(members[split]),
                "identities_with_both_classes": len(with_real & with_fake),
                "real": len(reals),
                "fake": len(fakes),
                "source_clips": len({item["source_clip"] for item in fakes}),
                # A fake whose own source real video sits elsewhere would be a
                # near-duplicate leak; the identity rule should make this 100%.
                "fakes_whose_source_real_is_in_split": sum(
                    1 for item in fakes
                    if item["source_clip"] in {r["source_clip"] for r in reals}),
            }
        report[dataset] = {
            "donor_identities_crossing": overlaps,
            "splits": per_split,
            "excluded": Counter(
                item["split"] for item in subset
                if item["split"].startswith("excluded")),
        }
    return report


def summarize(rows, assignments):
    video_counts = Counter((item["dataset"], item["split"], item["class_name"]) for item in rows)
    method_counts = Counter((item["dataset"], item["method"]) for item in rows if item["label"] == 0)
    return {
        "identity_protocol": (
            "Dataset-local identity splits. DFD val/test fakes are donor-safe in the strict manifest; "
            "fake videos are never assigned to training."
        ),
        "identities": {
            dataset: {
                split: sum(value == split for value in values.values())
                for split in ("train", "val", "test")
            }
            for dataset, values in assignments.items()
        },
        "videos": {
            "{}:{}:{}".format(dataset, split, label): count
            for (dataset, split, label), count in sorted(video_counts.items())
        },
        "fake_methods": {
            "{}:{}".format(dataset, method): count
            for (dataset, method), count in sorted(method_counts.items())
        },
    }


def write_manifest(path, rows):
    fields = [
        "dataset", "path", "label", "class_name", "split", "source_id",
        "target_id", "donor_id", "donor_seen_in_train", "source_clip",
        "driver_id", "method", "protocol",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    # Default to the environment so a git-synced checkout needs no local edits.
    parser.add_argument(
        "--dfd-root",
        default=os.environ.get(DATASET_ROOT_ENV_VARS["DFD"]),
        help="Defaults to $DFD_ROOT.",
    )
    parser.add_argument(
        "--celeb-root",
        default=os.environ.get(DATASET_ROOT_ENV_VARS["CelebDFv3"]),
        help="Defaults to $CELEBDFV3_ROOT.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--protocol", choices=["p0", "p05", "p1"], default="p05",
        help="p05 (default) keeps each real source and all derived fakes in one split; "
             "p0 is video-random and p1 is identity-disjoint.",
    )
    parser.add_argument(
        "--celeb-train-ratio", type=float, default=0.60,
        help="Ratio for the 59 CelebDFv3 celebrities, stratified apart from "
             "the 300 fake-less Real_YouTube subjects. 0.60/0.20 gives "
             "35/12/12 rather than the 40/7/12 a joint shuffle produced, "
             "which nearly doubles the usable validation identities.",
    )
    parser.add_argument("--celeb-val-ratio", type=float, default=0.20)
    parser.add_argument(
        "--only-stage-a", action="store_true",
        help="Write combined_manifest_stage_a.csv alone. Stratifying the "
             "celebrities changes the CelebDFv3 split, so re-emitting the "
             "legacy manifests under their old names into artifacts/manifests "
             "would silently replace the split the finished runs were scored "
             "on. Pass this whenever the output directory is the repository's.",
    )
    args = parser.parse_args()
    missing = [
        flag for flag, value in (("--dfd-root", args.dfd_root), ("--celeb-root", args.celeb_root))
        if not value
    ]
    if missing:
        parser.error(
            "{} not given and the matching environment variable is unset. Either pass the "
            "flags or export DFD_ROOT / CELEBDFV3_ROOT.".format(" and ".join(missing))
        )

    dfd_root, celeb_root = Path(args.dfd_root).resolve(), Path(args.celeb_root).resolve()
    dfd, dfd_assignments = dfd_rows(dfd_root, args.seed, args.train_ratio, args.val_ratio)
    celeb, celeb_assignments = celeb_rows(
        celeb_root, args.seed, args.train_ratio, args.val_ratio,
        args.celeb_train_ratio, args.celeb_val_ratio)
    raw_rows = dfd + celeb
    rows = assign_protocol(raw_rows, args.protocol, args.seed,
                           args.train_ratio, args.val_ratio)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "combined_manifest_{}.csv".format(args.protocol)
    summary_path = output_dir / "combined_summary_{}.json".format(args.protocol)
    write_manifest(manifest_path, rows)
    report = protocol_audit(rows, args.protocol)
    report.update({
        "roots": {"DFD": str(dfd_root), "CelebDFv3": str(celeb_root)},
        "seed": args.seed,
        "ratios": {"train": args.train_ratio, "val": args.val_ratio,
                   "test": 1.0 - args.train_ratio - args.val_ratio},
    })
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2))
    print("wrote {} and {}".format(manifest_path, summary_path))


if __name__ == "__main__":
    main()
