"""Rank the evaluation splits with scalars that carry no forgery evidence.

This is the gate that has to clear before Stage A training starts. Each control
sorts the videos by a single number that says nothing about whether a face was
manipulated -- the frame's resolution, its bitrate, how far the crop has to be
resampled -- and scores that ordering as if it were a detector. A control near
0.5 is inert. A control well away from 0.5 is a free detector the network will
find and learn instead of the artefacts, and preprocessing has to be fixed
before any training run is worth starting.

The method has a name outside vision: Poliak et al. (*SEM 2018) and Gururangan
et al. (NAACL 2018) exposed SNLI's annotation artefacts by classifying from the
hypothesis alone, 71% against a 33% chance rate. Smeu et al. (CVPR 2025) found
the same shape of problem in deepfake data -- forged clips in two widely used
audio-video sets begin with a brief silence, and "based on this feature alone,
we can separate the real and fake samples almost perfectly".

We have already paid for skipping this check. The whole-frame runs of
2026-09-02 reached 0.819 on CelebDFv3 while a ranker reading only the share of
canvas filled by replicated border reached 0.828: sixty epochs lost to a
control nobody had computed.

Intervals resample source clips and identities rather than rows, because the
rows are not independent -- see ``clustered_auroc_interval``.

Usage:
    python scripts/shortcut_controls.py \
        --manifest artifacts/manifests/combined_manifest_stage_a.csv \
        --video-sizes scripts/video_size_reports/video_sizes.csv \
        [--box-cache artifacts/face_boxes] [--target 256] [--margin 2.0]
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_bcnn.metrics import clustered_auroc_interval


# A control is inert when its interval covers 0.5. This is the margin by which
# the point estimate alone is allowed to drift before it is worth reporting
# even if the interval still covers chance.
NOTABLE = 0.03

# video_sizes.csv names a dataset by its folder on disk.
DATASET_ALIASES = {"DFD-Kaggle": "DFD"}

# Only what the network can actually see may block a run. It is handed decoded
# pixels, so a file's duration and container bitrate reach it solely through
# whatever visible trace they leave; compression artefacts do survive a crop,
# a clip count does not. Which tier the frame geometry belongs to depends on
# the input mode: a face crop resampled to a fixed canvas destroys the source
# resolution, whole-frame input preserves it.
CROP_CONTROLS = ("resampling_factor", "face_box_width_px", "face_box_share_of_frame")
COMPRESSION_CONTROLS = ("bitrate_mb_per_second", "bits_per_pixel")
GEOMETRY_CONTROLS = ("frame_width", "frame_height", "aspect_ratio", "frame_pixels")
FILE_CONTROLS = ("frame_count",)


def gating_controls(frame_mode):
    blocking = set(CROP_CONTROLS) | set(COMPRESSION_CONTROLS)
    if frame_mode == "whole":
        blocking |= set(GEOMETRY_CONTROLS)
    return blocking


def read_video_sizes(path):
    """Map (dataset, posix path relative to the dataset root) to its metadata.

    For CelebDFv3 the recorded path begins with the dataset's own folder, which
    the manifest's ``path`` column does not carry, so that component is dropped.
    DFD's rows already start below the root and keep every component.
    """
    table = {}
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "ok":
                continue
            dataset = DATASET_ALIASES.get(row["dataset"], row["dataset"])
            relative = row["relative_path"].replace("\\", "/")
            parts = relative.split("/")
            if len(parts) > 1 and parts[0] == row["dataset"]:
                relative = "/".join(parts[1:])
            table[(dataset, relative)] = row
    return table


def read_box_cache(directory):
    """Median detected box width per video, keyed the way the manifest names it.

    cache_face_boxes.py files a video at <cache>/<dataset>/<manifest path>.npz,
    so the key here is the (dataset, path) pair build_features looks up. Keying
    on Path.stem instead matched nothing, ever: the cache name keeps the video's
    own extension, so the stem read "id0_0000.mp4" against the manifest's
    "id0_0000" -- and 37364 cached files carry only 6913 distinct stems, because
    identity folders reuse filenames, so a stem was never an identifier at all.
    The three crop-scale controls were silently absent from every run of this
    script until this was fixed. `directory` is the cache root holding the
    per-dataset folders, not one dataset's folder.
    """
    widths = {}
    directory = Path(directory)
    if not directory.exists():
        return widths
    for item in directory.rglob("*.npz"):
        parts = item.relative_to(directory).parts
        if len(parts) < 2:
            continue
        dataset = DATASET_ALIASES.get(parts[0], parts[0])
        video = Path(*parts[1:]).as_posix()[: -len(".npz")]
        try:
            with np.load(item) as handle:
                boxes = handle["boxes"]
        except (OSError, KeyError):
            continue
        if len(boxes) == 0:
            continue
        # cache_face_boxes.py stores [x, y, w, h], the cascade's convention and
        # the one _crop_box reads, so column 2 is already the width. Subtracting
        # column 0 treated them as corners and returned w - x.
        widths[(dataset, video)] = float(np.median(boxes[:, 2]))
    return widths


def build_features(rows, sizes, box_widths, target, margin):
    """One dict of control scalars per video, plus the columns to cluster on."""
    records = []
    for row in rows:
        meta = sizes.get((row["dataset"], row["path"]))
        if meta is None:
            continue
        try:
            width, height = float(meta["width"]), float(meta["height"])
            frames = float(meta["frame_count"])
            duration = float(meta["duration_seconds"])
            megabytes = float(meta["file_size_mb"])
        except (KeyError, TypeError, ValueError):
            continue
        if width <= 0 or height <= 0 or duration <= 0:
            continue
        controls = {
            "frame_width": width,
            "frame_height": height,
            "aspect_ratio": width / height,
            "frame_pixels": width * height,
            # Encoder settings travel with the generator that produced a video,
            # so bitrate is a plausible free detector even after cropping --
            # cropping changes the pixels fed to the network, never the file
            # the pixels were decoded from.
            "bitrate_mb_per_second": megabytes / duration,
            "bits_per_pixel": (megabytes * 8e6) / max(frames * width * height, 1.0),
            "frame_count": frames,
        }
        box = box_widths.get((row["dataset"], row["path"]))
        if box:
            controls["face_box_width_px"] = box
            controls["face_box_share_of_frame"] = box / width
            # The quantity that survives a resize to a fixed canvas. The box's
            # share of the frame is normalised away by the resize; how far the
            # crop had to be stretched is not.
            controls["resampling_factor"] = target / (margin * box)
        records.append({
            "label": int(row["label"]),
            "source_clip": "{}:{}".format(row["dataset"], row["source_clip"]),
            "identity": "{}:{}".format(row["dataset"], row["target_id"]),
            "controls": controls,
        })
    return records


def evaluate(records, draws, seed, blocking):
    """AUROC and clustered intervals for every control the records carry."""
    names = sorted({name for item in records for name in item["controls"]})
    labels = np.asarray([item["label"] for item in records])
    # label 1 is real, so a control that increases with "fake" needs the sign
    # flip that scoring the fake class supplies.
    fake = 1 - labels
    results = {}
    for name in names:
        usable = [item for item in records if name in item["controls"]]
        if len(usable) < len(records):
            fake_here = np.asarray([1 - item["label"] for item in usable])
        else:
            fake_here = fake
        scores = np.asarray([item["controls"][name] for item in usable], dtype=float)
        if len(np.unique(fake_here)) < 2 or not np.isfinite(scores).all():
            continue
        entry = {}
        for unit in ("source_clip", "identity"):
            clusters = [item[unit] for item in usable]
            entry[unit] = clustered_auroc_interval(
                fake_here, scores, clusters, draws=draws, seed=seed)
        point = entry["source_clip"]["auroc"]
        # A control is equally informative when it ranks backwards, so the
        # distance from chance is what matters, not the direction.
        entry["distance_from_chance"] = abs(point - 0.5)
        entry["videos"] = len(usable)
        entry["gating"] = name in blocking
        entry["verdict"] = verdict_for(entry)
        results[name] = entry
    return results


def verdict_for(entry):
    intervals = [entry[unit] for unit in ("source_clip", "identity")]
    if any(math.isnan(item["low"]) for item in intervals):
        return "UNDECIDED"
    inert = all(item["low"] <= 0.5 <= item["high"] for item in intervals)
    if inert:
        return "PASS" if entry["distance_from_chance"] < NOTABLE else "PASS (drifting)"
    return "FAIL" if entry["gating"] else "noted"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--video-sizes", required=True)
    parser.add_argument(
        "--box-cache", default=None,
        help="Root of the detection cache, holding one folder per dataset -- "
             "artifacts/face_cache, not artifacts/face_cache/CelebDFv3. Without "
             "it the crop controls are skipped and only frame-level ones run.")
    parser.add_argument("--target", type=int, default=256,
                        help="Output canvas edge, for the resampling factor.")
    parser.add_argument("--margin", type=float, default=2.0,
                        help="Total box expansion, e.g. 2.0 for x2.0.")
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument(
        "--frame-mode", choices=["face", "whole"], default="face",
        help="What the network will be fed. 'face' resamples a crop onto a "
             "fixed canvas, which destroys the source resolution, so the "
             "geometry controls drop to diagnostics. 'whole' keeps the frame, "
             "so they block. Compression and crop-scale controls block in "
             "both, because a crop does not remove encoder artefacts.")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.manifest, "r", newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    sizes = read_video_sizes(args.video_sizes)
    boxes = read_box_cache(args.box_cache) if args.box_cache else {}
    print("video size rows: {}   cached detections: {}".format(len(sizes), len(boxes)))
    if not boxes:
        print("no detection cache given, so the crop-scale controls are skipped -- "
              "re-run with --box-cache once boxes are cached.")

    blocking = gating_controls(args.frame_mode)
    print("frame mode {!r}: blocking controls are {}".format(
        args.frame_mode, ", ".join(sorted(blocking))))

    report, failures, noted, provisional = {}, [], [], []
    for dataset in sorted({row["dataset"] for row in manifest}):
        for split in args.splits:
            rows = [row for row in manifest
                    if row["dataset"] == dataset and row["split"] == split]
            records = build_features(rows, sizes, boxes, args.target, args.margin)
            if not records:
                print("\n=== {}:{}  no video metadata matched {} manifest rows"
                      .format(dataset, split, len(rows)))
                continue
            results = evaluate(records, args.draws, args.seed, blocking)
            key = "{}:{}".format(dataset, split)
            report[key] = results
            # A blocking control with no column did not pass -- it was never
            # computed, because no video on this split had a cached box. A bare
            # PASS/FAIL list hid that.
            with_box = sum(1 for item in records if "resampling_factor" in item["controls"])
            absent = sorted(blocking - set(results))
            if absent:
                provisional.append("{}: {} (boxes cached for {} of {} videos)".format(
                    key, ", ".join(absent), with_box, len(records)))
            print("\n=== {}  ({} of {} videos matched, {} with cached boxes)".format(
                key, len(records), len(rows), with_box))
            print("{:<28} {:>7}  {:<16} {:<16} {}".format(
                "control", "auroc", "95% CI by clip", "by identity", "verdict"))
            for name in sorted(results, key=lambda n: -results[n]["distance_from_chance"]):
                entry = results[name]
                clip, ident = entry["source_clip"], entry["identity"]
                print("{:<28} {:>7.4f}  [{:.3f},{:.3f}]    [{:.3f},{:.3f}]    {}".format(
                    name, clip["auroc"], clip["low"], clip["high"],
                    ident["low"], ident["high"], entry["verdict"]))
                if entry["verdict"] == "FAIL":
                    failures.append("{} / {} ({:.3f})".format(key, name, clip["auroc"]))
                elif entry["verdict"] == "noted":
                    noted.append("{} / {} ({:.3f})".format(key, name, clip["auroc"]))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print("\nwrote {}".format(args.output))

    print("\n" + "=" * 72)
    if noted:
        print("Separable in the source files, but not reaching the network in "
              "frame mode {!r} -- recorded, not blocking:".format(args.frame_mode))
        for item in noted:
            print("  - {}".format(item))
        print()
    if provisional:
        print("INCOMPLETE. A blocking control could not be computed at all on "
              "these splits, so any verdict below covers only what ran:")
        for item in provisional:
            print("  - {}".format(item))
        print("  The crop-scale controls need cached boxes for the videos being "
              "gated. Finish scripts/cache_face_boxes.py and re-run; "
              "resampling_factor is the one that says whether source resolution "
              "survives the resize onto a fixed canvas.")
        print()
    if failures:
        print("GATE FAILED. {} blocking control(s) rank better than chance:"
              .format(len(failures)))
        for item in failures:
            print("  - {}".format(item))
        print("Fix preprocessing before training; a network given these will "
              "learn them instead of the manipulation.")
        return 1
    if provisional:
        print("No control that ran beats chance, but the gate is INCOMPLETE -- "
              "see above. This is not a pass.")
        return 2
    print("GATE PASSED. Every blocking control's interval covers 0.5 on every split.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
