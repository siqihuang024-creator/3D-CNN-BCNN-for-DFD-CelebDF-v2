#!/usr/bin/env python
"""Detect faces once, offline, and store boxes and landmarks for the loader.

The loader currently runs a Haar cascade on every frame of every clip of every
epoch, which is why the GPU sits near idle during training. Detecting once and
reading the result back is the fix, and it is also what makes the published
pipeline reachable: that pipeline aligns the face on its landmarks before it
takes the box, so landmarks have to be cached alongside the boxes or alignment
would mean a second pass over every video.

Two knobs make the job finishable. Detecting every frame of both corpora is
about 150 GPU-hours at MTCNN's measured throughput, so:

  --detect-stride   Detect every Nth frame and let the loader interpolate the
                    rest. Faces move a few pixels between neighbouring frames,
                    well under the movement the EMA smoother already absorbs.

  --max-detect-side Downscale the frame before detection and scale the
                    coordinates back. DFD is 1080p with a 200px face, so half
                    resolution still leaves a 100px face, far above MTCNN's
                    20px floor, and costs a quarter of the time.

Both defaults are conservative. Run with --limit first: the script prints a
measured projection for the full job before you commit a pod to it.

Output is one .npz per video mirroring the manifest path, so the job resumes by
skipping what exists, plus an index CSV carrying the per-video detection rate
and box stability that the preprocessing note promised as a QC report.
"""

import argparse
import collections
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class MtcnnBackend(object):
    """MTCNN via facenet-pytorch: 5 landmarks, batches frames of one shape."""

    name = "mtcnn"

    def __init__(self, args):
        import torch
        from facenet_pytorch import MTCNN
        self.torch = torch
        self.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MTCNN(keep_all=True, device=self.device,
                           min_face_size=args.min_face_size)

    def detect(self, frames):
        """Return (box, landmarks, score) per frame, None where nothing is found."""
        rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
        with self.torch.no_grad():
            boxes, scores, points = self.model.detect(rgb, landmarks=True)
        out = []
        for frame_boxes, frame_scores, frame_points in zip(boxes, scores, points):
            if frame_boxes is None or not len(frame_boxes):
                out.append(None)
                continue
            # The largest face, matching "extract the largest face" in the
            # published pipeline rather than the most confident one.
            areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in frame_boxes]
            best = int(np.argmax(areas))
            x0, y0, x1, y1 = frame_boxes[best]
            out.append((
                np.asarray([x0, y0, x1 - x0, y1 - y0], dtype=np.float32),
                np.asarray(frame_points[best], dtype=np.float32),
                float(frame_scores[best]),
            ))
        return out


class DlibBackend(object):
    """The protocol backend: dlib detection plus the 81-point shape predictor.

    DeepfakeBench does "face detection, face cropping, and alignment ... using
    DLIB" (paper Sec. 4.1) and its README ships the 81-landmark predictor, so
    this is the path that keeps our preprocessing traceable to the benchmark
    whose baselines we report against. Landmarks are the reason detection is
    cached at all: alignment consumes them, and re-deriving them later would
    mean a second pass over every video.
    """

    name = "dlib"

    def __init__(self, args):
        import dlib
        predictor = Path(args.dlib_predictor)
        if not predictor.exists():
            raise SystemExit(
                "shape_predictor_81_face_landmarks.dat not found at {}.\n"
                "Fetch it from https://github.com/codeniko/shape_predictor_81_face_landmarks"
                .format(predictor))
        self.dlib = dlib
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(str(predictor))
        self.upsample = args.dlib_upsample
        self.device = "cpu"

    def detect(self, frames):
        out = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rects = self.detector(rgb, self.upsample)
            if not len(rects):
                out.append(None)
                continue
            # "Extract the largest face", as the published pipeline specifies.
            rect = max(rects, key=lambda r: r.width() * r.height())
            shape = self.predictor(rgb, rect)
            points = np.asarray([[p.x, p.y] for p in shape.parts()], dtype=np.float32)
            box = np.asarray([rect.left(), rect.top(), rect.width(), rect.height()],
                             dtype=np.float32)
            # dlib's HOG detector returns no confidence, so the slot carries a
            # constant rather than a fabricated score.
            out.append((box, points, 1.0))
        return out


BACKENDS = {"mtcnn": MtcnnBackend, "dlib": DlibBackend}


def normalise(path):
    return path.replace("\\", "/").lstrip("/")


def plan_indices(frame_count, stride):
    """Frames to detect: a regular grid that always includes the last frame.

    The last frame is forced in so the loader interpolates between two detected
    anchors everywhere rather than extrapolating past the final one.
    """
    indices = list(range(0, frame_count, max(1, stride)))
    if indices[-1] != frame_count - 1:
        indices.append(frame_count - 1)
    return indices


def read_frames(capture, indices, max_side):
    """Decode the planned frames in one forward pass, downscaled for detection.

    Seeking per frame is slower than grabbing through the video once when the
    stride is small, and grabbing is what keeps decode off the critical path.
    """
    frames, scales, kept = [], [], []
    wanted = collections.deque(indices)
    position = 0
    while wanted:
        target = wanted[0]
        while position < target:
            if not capture.grab():
                return frames, scales, kept
            position += 1
        ok, frame = capture.read()
        position += 1
        wanted.popleft()
        if not ok:
            continue
        scale = 1.0
        height, width = frame.shape[:2]
        longest = max(height, width)
        if max_side and longest > max_side:
            scale = max_side / float(longest)
            frame = cv2.resize(frame, (int(round(width * scale)), int(round(height * scale))),
                               interpolation=cv2.INTER_AREA)
        frames.append(frame)
        scales.append(scale)
        kept.append(target)
    return frames, scales, kept


def cache_video(backend, video_path, indices, args):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None
    frames, scales, kept = read_frames(capture, indices, args.max_detect_side)
    capture.release()
    if not frames:
        return None

    rows = []
    for start in range(0, len(frames), args.batch_size):
        chunk = frames[start:start + args.batch_size]
        # A batch must share one shape; a video whose resolution changes
        # mid-stream would otherwise raise deep inside the detector.
        if len({f.shape for f in chunk}) != 1:
            rows.extend(item for frame in chunk for item in backend.detect([frame]))
        else:
            rows.extend(backend.detect(chunk))

    frame_indices, boxes, landmarks, scores = [], [], [], []
    for index, scale, found in zip(kept, scales, rows):
        if found is None:
            continue
        box, points, score = found
        frame_indices.append(index)
        boxes.append(box / scale)
        landmarks.append(points / scale)
        scores.append(score)
    # The landmark count is the backend's, not a constant: MTCNN returns 5,
    # dlib's protocol predictor returns 81, and the consumer reads the shape.
    points = np.asarray(landmarks, dtype=np.float32)
    points = points.reshape(len(landmarks), -1, 2) if len(landmarks) else points.reshape(0, 0, 2)
    return {
        "frame_indices": np.asarray(frame_indices, dtype=np.int32),
        "boxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        "landmarks": points,
        "scores": np.asarray(scores, dtype=np.float32),
        "planned": len(kept),
    }


def quality(payload):
    """Per-video QC: how much was found, and how steady it was."""
    planned = payload["planned"]
    found = len(payload["frame_indices"])
    record = {"planned_frames": planned, "detected_frames": found,
              "detection_rate": found / float(max(planned, 1))}
    if found:
        widths = payload["boxes"][:, 2]
        centres = np.stack([payload["boxes"][:, 0] + widths / 2.0,
                            payload["boxes"][:, 1] + payload["boxes"][:, 3] / 2.0], axis=1)
        record["median_box_width_px"] = float(np.median(widths))
        record["median_score"] = float(np.median(payload["scores"]))
        # Frame-to-frame movement, not movement over the whole video: the first
        # is the detector disagreeing with itself, the second is the subject
        # walking across the room.
        if found > 1:
            steps = np.linalg.norm(np.diff(centres, axis=0), axis=1)
            record["median_centre_step_px"] = float(np.median(steps))
            record["max_centre_step_px"] = float(np.max(steps))
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest",
                        default=str(ROOT / "artifacts/manifests/combined_manifest_supervised_clean.csv"))
    parser.add_argument("--dataset-root", action="append", default=None, metavar="NAME=PATH")
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts/face_cache"))
    parser.add_argument("--detector", default="dlib", choices=sorted(BACKENDS),
                        help="dlib is the protocol backend and the default; mtcnn "
                             "is kept for the robustness comparison only.")
    parser.add_argument("--dlib-predictor",
                        default=str(ROOT / "scripts/models/shape_predictor_81_face_landmarks.dat"),
                        help="The 81-point predictor DeepfakeBench's README ships.")
    parser.add_argument("--dlib-upsample", type=int, default=1,
                        help="dlib upsampling passes before detection. DFD faces "
                             "are ~11%% of frame width, so 0 will miss them.")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Restrict to these corpora; default is every corpus present.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        help="Excluded-donor rows are skipped by default since no "
                             "run reads them.")
    parser.add_argument("--detect-stride", type=int, default=4,
                        help="Detect every Nth frame; the loader interpolates between.")
    parser.add_argument("--max-detect-side", type=int, default=640,
                        help="Downscale so the long side is at most this before "
                             "detection. 0 disables. Coordinates are stored at "
                             "full resolution either way.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--min-face-size", type=int, default=40,
                        help="MTCNN floor in detection-scale pixels. DFD faces are "
                             "~213px at 1080p, so ~100px after the default "
                             "downscale; 40 leaves headroom without inviting noise.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--shard", type=int, default=0,
                        help="Which slice of the manifest this process takes.")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Split the manifest across this many processes. "
                             "dlib is single-threaded, so N shards on N cores "
                             "give close to N times the throughput; each shard "
                             "skips files that already exist, so any shard can "
                             "be re-run or resumed independently.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after this many videos and project the full job.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Redo videos already cached instead of skipping them.")
    args = parser.parse_args(argv)

    # Same precedence as override_dataset_roots, which every other script here
    # uses: --dataset-root beats DFD_ROOT / CELEBDFV3_ROOT, which beat the YAML.
    # Reading the YAML alone made this the one script that ignored
    # remote_env.sh and died on the Windows paths written in the config.
    roots = {}
    for name, variable in (("DFD", "DFD_ROOT"), ("CelebDFv3", "CELEBDFV3_ROOT")):
        value = os.environ.get(variable)
        if value:
            roots[name] = Path(value)
    for entry in (args.dataset_root or []):
        name, _, path = entry.partition("=")
        roots[name] = Path(path)
    if not roots:
        import yaml
        with open(ROOT / "configs/combined_3d_bcnn.yaml", "r", encoding="utf-8") as handle:
            roots = {k: Path(v) for k, v in
                     yaml.safe_load(handle)["data"]["dataset_roots"].items()}
    for name, path in roots.items():
        if not path.exists():
            raise SystemExit("dataset root {} does not exist: {}".format(name, path))

    with open(args.manifest, "r", newline="", encoding="utf-8-sig") as handle:
        records = [row for row in csv.DictReader(handle)
                   if row["split"] in args.splits
                   and (not args.datasets or row["dataset"] in args.datasets)]
    # One entry per file: CelebDFv3 lists a video once, but keeping this explicit
    # means a manifest that ever repeats a path costs one detection, not two.
    unique, seen = [], set()
    for row in records:
        key = (row["dataset"], normalise(row["path"]))
        if key not in seen:
            seen.add(key)
            unique.append(row)
    # dlib's HOG detector is single-threaded, so one process leaves thirteen
    # cores idle and DFD alone projects at 12.7 hours. Sharding by position lets
    # N processes cover disjoint slices of the same manifest; each still skips
    # what already exists, so a shard can be re-run or resumed on its own.
    if args.num_shards > 1:
        if not 0 <= args.shard < args.num_shards:
            raise SystemExit("--shard must be in [0, --num-shards).")
        unique = [row for index, row in enumerate(unique)
                  if index % args.num_shards == args.shard]
        print("shard {}/{}: {} of the videos".format(
            args.shard, args.num_shards, len(unique)))
    print("{} videos to consider from {}".format(len(unique), Path(args.manifest).name))

    output = Path(args.output_dir)
    backend = BACKENDS[args.detector](args)
    print("detector {} on {}, stride {}, detect side <= {}, batch {}\n".format(
        args.detector, getattr(backend, "device", "?"), args.detect_stride,
        args.max_detect_side or "native", args.batch_size))

    index_rows, failures = [], []
    started, frames_done, videos_done = time.time(), 0, 0
    for row in unique:
        relative = normalise(row["path"])
        destination = output / row["dataset"] / (relative + ".npz")
        if destination.exists() and not args.overwrite:
            continue
        video_path = roots[row["dataset"]] / row["path"]
        capture = cv2.VideoCapture(str(video_path))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if frame_count <= 0:
            failures.append({"dataset": row["dataset"], "path": relative,
                             "reason": "unreadable"})
            continue

        payload = cache_video(backend, video_path,
                              plan_indices(frame_count, args.detect_stride), args)
        if payload is None:
            failures.append({"dataset": row["dataset"], "path": relative,
                             "reason": "no decodable frames"})
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination, frame_indices=payload["frame_indices"], boxes=payload["boxes"],
            landmarks=payload["landmarks"], scores=payload["scores"],
            frame_count=np.int32(frame_count), detect_stride=np.int32(args.detect_stride))
        stats = quality(payload)
        stats.update({"dataset": row["dataset"], "path": relative,
                      "class_name": row["class_name"], "split": row["split"],
                      "method": row["method"], "frame_count": frame_count})
        index_rows.append(stats)

        frames_done += payload["planned"]
        videos_done += 1
        if videos_done % 50 == 0:
            rate = (time.time() - started) / videos_done
            print("  {} videos, {:.2f}s each, {:.0f} frames/s".format(
                videos_done, rate, frames_done / (time.time() - started)))
        if args.limit and videos_done >= args.limit:
            break

    elapsed = time.time() - started
    if not videos_done:
        print("\nNothing to do: every video already has a cache entry.")
        return 0

    print("\n{} videos in {:.0f}s ({:.2f}s each, {:.0f} detections/s)".format(
        videos_done, elapsed, elapsed / videos_done, frames_done / max(elapsed, 1e-9)))
    if args.limit:
        remaining = len(unique) - videos_done
        print("PROJECTION: {} videos remain -> {:.1f} hours at this rate.".format(
            remaining, remaining * (elapsed / videos_done) / 3600.0))

    by_group = collections.defaultdict(list)
    for stats in index_rows:
        by_group[(stats["dataset"], stats["class_name"])].append(stats)
    print("\n{:<10} {:<6} {:>7} {:>10} {:>10} {:>11} {:>10}".format(
        "corpus", "class", "videos", "det.rate", "rate<90%", "box px", "step px"))
    for key in sorted(by_group):
        group = by_group[key]
        weak = sum(1 for s in group if s["detection_rate"] < 0.9)
        scored = [s for s in group if "median_box_width_px" in s]
        print("{:<10} {:<6} {:>7} {:>10} {:>10} {:>11} {:>10}".format(
            key[0], key[1], len(group),
            "{:.1%}".format(float(np.mean([s["detection_rate"] for s in group]))),
            "{:.1%}".format(weak / float(len(group))),
            "{:.0f}".format(np.median([s["median_box_width_px"] for s in scored])) if scored else "-",
            "{:.1f}".format(np.median([s.get("median_centre_step_px", np.nan)
                                       for s in scored])) if scored else "-"))
    if failures:
        print("\n{} videos could not be read; listed in the report.".format(len(failures)))

    output.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "split", "class_name", "method", "path", "frame_count",
              "planned_frames", "detected_frames", "detection_rate",
              "median_box_width_px", "median_score", "median_centre_step_px",
              "max_centre_step_px"]
    report = output / "cache_index.csv"
    write_header = not report.exists() or args.overwrite
    with open(report, "a" if not write_header else "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(index_rows)
    with open(output / "cache_report.json", "w", encoding="utf-8") as handle:
        json.dump({"settings": vars(args), "videos_cached": videos_done,
                   "seconds": elapsed, "failures": failures},
                  handle, indent=2, sort_keys=True, default=str)
    print("\nwrote {} and {}".format(report, output / "cache_report.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
