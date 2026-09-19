"""Contiguous video-clip decoding with temporally consistent face preprocessing."""

import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from torchvision.transforms import CenterCrop, Normalize, Resize, ToTensor


class UnreadableVideoError(RuntimeError):
    """A video the decoder cannot open at all, after retries."""


class MissingFaceCacheError(RuntimeError):
    """Required protocol face detections are absent; this must stop a run."""


def preflight_face_cache(records, config, limit=10):
    """Fail before training when a face-mode manifest lacks cached detections."""
    data = config.get("data", config)
    if data.get("frame_mode", "face") != "face" or not data.get("face_box_cache"):
        return {"required": False, "checked": 0, "missing": 0}
    cache_root = Path(data["face_box_cache"])
    missing = []
    for row in records:
        path = cache_root / row["dataset"] / (Path(row["path"]).as_posix() + ".npz")
        if not path.is_file():
            missing.append(str(path))
    if missing:
        preview = "\n  ".join(missing[:int(limit)])
        raise MissingFaceCacheError(
            "Face-cache preflight failed: {} of {} videos have no cache file.\n"
            "First {} missing:\n  {}\nRun scripts/cache_face_boxes.py for the "
            "same manifest before training or evaluation.".format(
                len(missing), len(records), min(len(missing), int(limit)), preview
            )
        )
    return {"required": True, "checked": len(records), "missing": 0}


def skip_unreadable_collate(batch):
    """Drop unreadable items instead of failing the epoch.

    One corrupt file in 30k training videos is statistically irrelevant, but it
    has now ended three multi-hour runs. Returning None lets the caller count
    and report the skip rather than lose the run.
    """
    skipped = [item["path"] for item in batch
               if isinstance(item, dict) and item.get("_skip_video")]
    items = [item for item in batch
             if item is not None and not item.get("_skip_video", False)]
    if not items:
        return ({"_skip_only": True, "_skipped_paths": skipped}
                if skipped else None)
    result = default_collate(items)
    result["_skipped_paths"] = skipped
    return result


def load_manifest(path):
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)


class VideoClipDataset(Dataset):
    """Decode contiguous clips in memory without writing extracted frames to disk."""

    def __init__(self, records, dataset_roots, config, training, clips_per_video=None):
        self.records = list(records)
        self.dataset_roots = {name: Path(root) for name, root in dataset_roots.items()}
        self.training = bool(training)
        self.clip_length = int(config["clip_length"])
        self.train_clip_strides = tuple(int(value) for value in config.get("train_clip_strides", [1]))
        self.eval_clip_stride = int(config.get("eval_clip_stride", 1))
        default_clips = 1 if self.training else int(config.get("eval_clips_per_video", 1))
        self.clips_per_video = int(default_clips if clips_per_video is None else clips_per_video)
        self.face_crop = bool(config.get("face_crop", True))
        self.face_margin = float(config.get("face_margin", 0.25))
        # "face" keeps the tracked-box behaviour every run up to v15 used.
        # "letterbox" keeps the whole frame instead: it rescales it to fit the
        # target canvas and pads the short side, so no face detection runs at
        # all. That removes the 93% of per-item time Haar costs, and on DFD it
        # removes a 50.3% detection failure rate, at the price of a face that is
        # smaller in the input and of a padding fraction that is visible to the
        # model -- CelebDFv3's reals are 99.8% wide while 59.6% of its fakes are
        # square, so padding is class-correlated and results need stratifying.
        # "decimate" also keeps the whole frame, but never resamples it: it
        # takes every Nth pixel along both axes and feeds the result straight
        # to the network. That matters because cv2's INTER_AREA -- what
        # "letterbox" uses on any downscale -- is a low-pass filter, and it
        # averaged 4.6 input pixels into each output pixel on the DFD run that
        # failed. Manipulation traces are high-frequency, so the resize was
        # discarding the evidence before conv1 ever saw it. Discarding pixels
        # instead aliases those frequencies rather than removing them.
        # It does NOT change how much of the input the face occupies: that
        # ratio is fixed by the frame, and stays at 1.2% of the 22x22 grid
        # either way. This mode addresses the frequency problem, not dilution.
        self.frame_mode = str(config.get("frame_mode", "face"))
        if self.frame_mode not in ("face", "letterbox", "decimate"):
            raise ValueError(
                "Unknown frame_mode {!r}; expected 'face', 'letterbox' or "
                "'decimate'.".format(self.frame_mode)
            )
        self.decimate_step = int(config.get("decimate_step", 2))
        if self.decimate_step < 1:
            raise ValueError("decimate_step must be a positive integer.")
        size = config.get("letterbox_size", (768, 432))
        self.letterbox_size = (int(size[0]), int(size[1]))
        if min(self.letterbox_size) < 32:
            raise ValueError("letterbox_size must be at least 32 pixels a side.")
        self.crop_padding = str(config.get("crop_padding", "clamp"))
        if self.crop_padding not in ("clamp", "replicate"):
            raise ValueError(
                "Unknown crop_padding {!r}; expected 'clamp' or "
                "'replicate'.".format(self.crop_padding)
            )
        self.face_detector_scale_factor = float(config.get("face_detector_scale_factor", 1.1))
        self.face_detector_min_neighbors = int(config.get("face_detector_min_neighbors", 5))
        self.box_smoothing_alpha = float(config.get("box_smoothing_alpha", 0.6))
        # Boxes and 81-point landmarks detected once by scripts/cache_face_boxes.py.
        # Without this the only detector here is the Haar cascade below, which is
        # not the protocol: DeepfakeBench specifies dlib, and alignment needs
        # landmarks a cascade never produces. A missing entry raises rather than
        # falling back, so a run cannot silently be Haar-without-alignment while
        # its config claims otherwise.
        cache_root = config.get("face_box_cache")
        self.face_box_cache = Path(cache_root) if cache_root else None
        self.align_landmarks = bool(config.get("align_landmarks", False))
        if self.align_landmarks and self.face_box_cache is None:
            raise ValueError(
                "align_landmarks needs face_box_cache: alignment consumes the "
                "landmarks the cache stores, and the Haar cascade emits none."
            )
        # One decompressed .npz per video, held per worker. Videos are revisited
        # every epoch and the arrays are small next to a decoded clip.
        self.detection_store = {}
        self.horizontal_flip_probability = float(config.get("horizontal_flip_probability", 0.5))
        # Overfit diagnostics need the same frames every epoch, so the random
        # training clip position can be pinned to the middle of the video.
        self.deterministic_clips = bool(config.get("deterministic_train_clips", False))
        self.resize = Resize(int(config["input_resize"]))
        self.center_crop = CenterCrop(int(config["center_crop"]))
        self.to_tensor = ToTensor()
        self.normalize = Normalize(config["normalize_mean"], config["normalize_std"])
        self.detector = None
        # Repaired frame counts, so a video with bad metadata is measured once
        # per worker rather than on every epoch.
        self.frame_count_cache = {}
        # Videos this worker could not open, reported instead of crashing.
        self.unreadable = set()
        if self.clip_length < 1:
            raise ValueError("clip_length must be positive.")
        if not self.train_clip_strides or min(self.train_clip_strides) < 1:
            raise ValueError("train_clip_strides must contain positive integers.")
        if self.eval_clip_stride < 1 or self.clips_per_video < 1:
            raise ValueError("eval clip stride and clips_per_video must be positive.")
        if not 0.0 <= self.box_smoothing_alpha <= 1.0:
            raise ValueError("box_smoothing_alpha must be in [0, 1].")

    def __len__(self):
        return len(self.records)

    def _detector(self):
        if self.detector is None:
            cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.detector = cv2.CascadeClassifier(cascade)
            if self.detector.empty():
                raise RuntimeError("OpenCV Haar face cascade could not be initialized.")
        return self.detector

    def _cache_path(self, video_path):
        """Where cache_face_boxes.py filed this video: <cache>/<dataset>/<rel>.npz."""
        video_path = Path(video_path)
        for name, root in self.dataset_roots.items():
            try:
                relative = video_path.relative_to(root)
            except ValueError:
                continue
            return self.face_box_cache / name / (relative.as_posix() + ".npz")
        raise UnreadableVideoError(
            "{} sits under none of the configured dataset roots, so its "
            "detections cannot be located.".format(video_path)
        )

    def _cached_detections(self, video_path):
        """Frame indices, boxes and landmarks as detected offline.

        Boxes are [x, y, w, h] in the source frame's pixels, matching both the
        cascade's convention and `_crop_box`. Landmarks are (frames, 81, 2).
        """
        key = str(video_path)
        stored = self.detection_store.get(key)
        if stored is not None:
            return stored
        path = self._cache_path(video_path)
        if not path.exists():
            raise MissingFaceCacheError(
                "No cached detections at {}. Run scripts/cache_face_boxes.py "
                "over this manifest, or clear face_box_cache from the config."
                .format(path)
            )
        with np.load(path) as handle:
            indices = np.asarray(handle["frame_indices"], dtype=np.int64)
            boxes = np.asarray(handle["boxes"], dtype=np.float64).reshape(-1, 4)
            landmarks = np.asarray(handle["landmarks"], dtype=np.float64)
            stride = int(handle["detect_stride"]) if "detect_stride" in handle else 1
        order = np.argsort(indices)
        stored = (indices[order], boxes[order],
                  landmarks[order] if len(landmarks) == len(indices) else landmarks,
                  max(stride, 1))
        self.detection_store[key] = stored
        return stored

    @staticmethod
    def _align_on_eyes(frame, box, landmark):
        """Rotate the frame so the eye line is level, about the box centre.

        The protocol aligns before taking the box, and rotating about the box's
        own centre leaves that centre fixed, so the box survives the warp
        unchanged and its width keeps meaning what the margin assumes. The
        81-point predictor is dlib's 68 plus 13 forehead points, so the eye
        indices are the usual 36-41 and 42-47.
        """
        if landmark is None or len(landmark) < 48:
            return frame
        left = np.asarray(landmark[36:42], dtype=np.float64).mean(axis=0)
        right = np.asarray(landmark[42:48], dtype=np.float64).mean(axis=0)
        delta = right - left
        if not np.isfinite(delta).all() or np.allclose(delta, 0.0):
            return frame
        angle = float(np.degrees(np.arctan2(delta[1], delta[0])))
        x, y, width, height = [float(value) for value in box]
        centre = (x + width / 2.0, y + height / 2.0)
        matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)
        return cv2.warpAffine(
            frame, matrix, (frame.shape[1], frame.shape[0]),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )

    @staticmethod
    def _interpolate_at(indices, values, wanted):
        """Linear interpolation between the two detections bracketing `wanted`.

        Detection runs every `detect_stride` frames, so most requested frames
        have no entry of their own. A face moves a few pixels between adjacent
        frames -- well inside what the EMA smoother already absorbs -- so
        interpolating is closer to the truth than snapping to the nearest
        detected frame. Queries outside the detected range clamp to the end.
        """
        position = int(np.searchsorted(indices, wanted))
        if position <= 0:
            return values[0].copy()
        if position >= len(indices):
            return values[-1].copy()
        left, right = position - 1, position
        if indices[right] == wanted:
            return values[right].copy()
        span = float(indices[right] - indices[left])
        fraction = 0.0 if span <= 0 else float(wanted - indices[left]) / span
        return (1.0 - fraction) * values[left] + fraction * values[right]

    @staticmethod
    def _max_start(frame_count, clip_length, stride):
        return max(0, int(frame_count) - 1 - (int(clip_length) - 1) * int(stride))

    def clip_indices(self, frame_count):
        """Return one random training clip or deterministic evaluation clips."""
        if frame_count <= 0:
            raise RuntimeError("Video reports no decodable frames.")
        if self.training and not self.deterministic_clips:
            stride = int(np.random.choice(self.train_clip_strides))
            max_start = self._max_start(frame_count, self.clip_length, stride)
            start = int(np.random.randint(0, max_start + 1)) if max_start else 0
            starts = [start]
        elif self.training:
            stride = int(self.train_clip_strides[0])
            starts = [self._max_start(frame_count, self.clip_length, stride) // 2]
        else:
            stride = self.eval_clip_stride
            max_start = self._max_start(frame_count, self.clip_length, stride)
            starts = np.linspace(0, max_start, self.clips_per_video).round().astype(int).tolist()
        return [
            [min(frame_count - 1, start + offset * stride) for offset in range(self.clip_length)]
            for start in starts
        ]

    @staticmethod
    def _center_square_box(frame):
        height, width = frame.shape[:2]
        side = min(height, width)
        return np.asarray([(width - side) / 2.0, (height - side) / 2.0, side, side], dtype=np.float64)

    def _detect_box(self, frame):
        if not self.face_crop:
            return self._center_square_box(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = self._detector().detectMultiScale(
            gray,
            scaleFactor=self.face_detector_scale_factor,
            minNeighbors=self.face_detector_min_neighbors,
        )
        if not len(boxes):
            return None
        return np.asarray(max(boxes, key=lambda box: int(box[2]) * int(box[3])), dtype=np.float64)

    @staticmethod
    def _interpolate_missing_boxes(raw_boxes, frames):
        valid = [index for index, box in enumerate(raw_boxes) if box is not None]
        if not valid:
            return [VideoClipDataset._center_square_box(frame) for frame in frames]
        filled = []
        for index, box in enumerate(raw_boxes):
            if box is not None:
                filled.append(box.copy())
                continue
            left = max((item for item in valid if item < index), default=None)
            right = min((item for item in valid if item > index), default=None)
            if left is None:
                filled.append(raw_boxes[right].copy())
            elif right is None:
                filled.append(raw_boxes[left].copy())
            else:
                fraction = float(index - left) / float(right - left)
                filled.append((1.0 - fraction) * raw_boxes[left] + fraction * raw_boxes[right])
        return filled

    def _cached_boxes(self, frames, frame_indices, video_path):
        """Boxes and landmarks for these frames, read from the offline cache."""
        indices, boxes, landmarks, stride = self._cached_detections(video_path)
        if len(indices) == 0:
            # The detector found nothing anywhere in this video. Say so instead
            # of inventing a face: the centre square keeps the shapes valid and
            # the audit carries the failure downstream.
            return ([self._center_square_box(frame) for frame in frames],
                    [None] * len(frames), len(frames))
        has_landmarks = len(landmarks) == len(indices) and landmarks.ndim == 3
        picked, points, misses = [], [], 0
        for wanted in frame_indices:
            picked.append(self._interpolate_at(indices, boxes, int(wanted)))
            points.append(self._interpolate_at(indices, landmarks, int(wanted))
                          if has_landmarks else None)
            # Detection ran every `stride` frames, so the nearest entry should
            # never be farther than that. Farther means the scheduled detection
            # there failed, which is a miss however smooth the interpolation.
            nearest = int(np.min(np.abs(indices - int(wanted))))
            misses += int(nearest > stride)
        return picked, points, misses

    def _prepare_boxes(self, frames, frame_indices=None, video_path=None):
        if self.frame_mode in ("letterbox", "decimate"):
            # The whole frame is kept, so there is no box to find and no
            # detection to fail. The audit keys stay so downstream reporting
            # does not have to special-case the mode.
            boxes = [
                np.asarray([0.0, 0.0, float(frame.shape[1]), float(frame.shape[0])])
                for frame in frames
            ]
            audit = {
                "any_miss": 0.0, "miss_fraction": 0.0,
                "center_x_jitter": 0.0, "center_y_jitter": 0.0,
                "width_jitter": 0.0, "height_jitter": 0.0,
            }
            return boxes, [None] * len(frames), audit
        if self.face_box_cache is not None:
            filled, points, misses = self._cached_boxes(
                frames, frame_indices, video_path)
        else:
            raw_boxes = [self._detect_box(frame) for frame in frames]
            filled = self._interpolate_missing_boxes(raw_boxes, frames)
            points = [None] * len(frames)
            misses = sum(box is None for box in raw_boxes)
        boxes, previous = [], None
        alpha = self.box_smoothing_alpha
        for detected in filled:
            smoothed = detected.copy() if previous is None else alpha * detected + (1.0 - alpha) * previous
            boxes.append(smoothed)
            previous = smoothed
        normalized = []
        for frame, box in zip(frames, boxes):
            frame_height, frame_width = frame.shape[:2]
            x, y, width, height = box
            normalized.append([
                (x + width / 2.0) / float(frame_width),
                (y + height / 2.0) / float(frame_height),
                width / float(frame_width),
                height / float(frame_height),
            ])
        normalized = np.asarray(normalized, dtype=np.float64)
        audit = {
            "any_miss": float(misses > 0),
            "miss_fraction": float(misses) / float(max(len(frames), 1)),
            "center_x_jitter": float(normalized[:, 0].std()),
            "center_y_jitter": float(normalized[:, 1].std()),
            "width_jitter": float(normalized[:, 2].std()),
            "height_jitter": float(normalized[:, 3].std()),
        }
        return boxes, points, audit

    def _crop_box(self, frame, box):
        """Cut the margin-expanded box out of the frame.

        `crop_padding="clamp"` truncates the box at the frame edge, which is
        what every run up to v15 did. That silently breaks the scale
        normalisation a face-anchored crop is for: 16% of CelebDFv3 faces and
        22% of DFD faces sit more than 30% off centre, and for those the box
        comes back rectangular and no longer centred on the face, so the later
        Resize/CenterCrop trims one side. The wider the margin the more often it
        happens, so a wide-context crop needs `"replicate"`, which pads with the
        edge pixels and always returns the full requested square.
        """
        if self.frame_mode in ("letterbox", "decimate"):
            return frame
        frame_height, frame_width = frame.shape[:2]
        x, y, width, height = [float(value) for value in box]
        x0 = int(round(x - width * self.face_margin))
        y0 = int(round(y - height * self.face_margin))
        x1 = int(round(x + width * (1.0 + self.face_margin)))
        y1 = int(round(y + height * (1.0 + self.face_margin)))
        if self.crop_padding == "clamp":
            crop = frame[max(0, y0):min(frame_height, y1),
                         max(0, x0):min(frame_width, x1)]
            if crop.size:
                return crop
            fallback = self._center_square_box(frame)
            fx, fy, fw, fh = [int(round(value)) for value in fallback]
            return frame[fy:fy + fh, fx:fx + fw]
        crop = frame[max(0, y0):min(frame_height, y1),
                     max(0, x0):min(frame_width, x1)]
        if not crop.size:
            fallback = self._center_square_box(frame)
            fx, fy, fw, fh = [int(round(value)) for value in fallback]
            return frame[fy:fy + fh, fx:fx + fw]
        top, bottom = max(0, -y0), max(0, y1 - frame_height)
        left, right = max(0, -x0), max(0, x1 - frame_width)
        if top or bottom or left or right:
            crop = cv2.copyMakeBorder(
                crop, top, bottom, left, right, cv2.BORDER_REPLICATE
            )
        return crop

    def _letterbox(self, bgr_frame):
        """Fit the whole frame into the canvas and pad the short side.

        Edge replication rather than black bars: a constant border is a signal
        no natural image carries, and the padded fraction already correlates
        with the class here.
        """
        target_width, target_height = self.letterbox_size
        height, width = bgr_frame.shape[:2]
        scale = min(target_width / float(width), target_height / float(height))
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(bgr_frame, (new_width, new_height), interpolation=interpolation)
        left = (target_width - new_width) // 2
        top = (target_height - new_height) // 2
        return cv2.copyMakeBorder(
            resized, top, target_height - new_height - top,
            left, target_width - new_width - left, cv2.BORDER_REPLICATE,
        )

    def _decimate(self, bgr_frame):
        """Keep every Nth pixel on both axes; no filtering, no interpolation.

        Plain slicing, not cv2.INTER_NEAREST: nearest-neighbour maps through
        float coordinates and rounds, so its sample positions drift on any
        non-integer ratio. Slicing is the literal operation -- the surviving
        pixels keep their original values exactly.

        Both axes, so 1920x1080 becomes 960x540 and the aspect ratio holds.
        Decimating one axis alone would squeeze every face 2:1.
        """
        step = self.decimate_step
        if step == 1:
            return bgr_frame
        return bgr_frame[::step, ::step]

    def _transform(self, bgr_frame, flip):
        if self.frame_mode == "letterbox":
            bgr_frame = self._letterbox(bgr_frame)
        elif self.frame_mode == "decimate":
            bgr_frame = self._decimate(bgr_frame)
        image = Image.fromarray(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
        if flip:
            image = ImageOps.mirror(image)
        if self.frame_mode == "face":
            # Only the face path resamples. Sending a decimated frame through
            # Resize would interpolate away exactly what decimating preserved.
            image = self.resize(image)
            image = self.center_crop(image)
        return self.normalize(self.to_tensor(image))

    @staticmethod
    def _decode_clips(capture, video_path, clip_indices):
        """Seek once per clip, then decode sequentially to preserve frame order.

        Returns None when the seek lands past the last decodable frame, which
        happens whenever container metadata overstates the frame count or the
        installed FFmpeg seeks differently from the one the manifest was built
        with. The caller then falls back to a seek-free pass.
        """
        clips = []
        for indices in clip_indices:
            start, end = int(indices[0]), int(indices[-1])
            wanted = set(int(index) for index in indices)
            capture.set(cv2.CAP_PROP_POS_FRAMES, start)
            decoded, previous = {}, None
            for frame_index in range(start, end + 1):
                ok, frame = capture.read()
                # A stalled read can report success and still hand back None:
                # FFmpeg's interrupt callback fires on timeout after OpenCV has
                # already set the success flag.
                if not ok or frame is None:
                    if previous is None:
                        return None
                    frame = previous.copy()
                if frame_index in wanted:
                    decoded[frame_index] = frame.copy()
                previous = frame
            missing = [index for index in wanted if index not in decoded]
            if missing:
                if previous is None:
                    return None
                for index in missing:
                    decoded[index] = previous.copy()
            clips.append([decoded[int(index)].copy() for index in indices])
        return clips

    @staticmethod
    def _measure_decodable_frames(capture):
        """Count frames the decoder actually yields, ignoring container metadata."""
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        count = 0
        while capture.grab():
            count += 1
        return count

    @staticmethod
    def _decode_clips_sequential(capture, video_path, clip_indices):
        """Decode every clip in one forward pass, never seeking mid-video.

        `grab` skips colour conversion for frames no clip wants, so the extra
        cost over seeking is small, and it works on files whose seek index is
        unusable.
        """
        wanted = sorted({int(index) for indices in clip_indices for index in indices})
        if not wanted:
            raise RuntimeError("No frames requested for {}".format(video_path))
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frames, position, target, last = {}, 0, set(wanted), wanted[-1]
        while position <= last:
            if not capture.grab():
                break
            if position in target:
                ok, frame = capture.retrieve()
                if not ok or frame is None:
                    break
                frames[position] = frame.copy()
            position += 1
        if not frames:
            raise UnreadableVideoError(
                "Video yielded no decodable frames; it is likely truncated or "
                "corrupt: {}".format(video_path)
            )
        available = sorted(frames)
        clips = []
        for indices in clip_indices:
            clip = []
            for index in indices:
                index = int(index)
                if index not in frames:
                    # Clamp onto the nearest frame at or before the request.
                    earlier = [key for key in available if key <= index]
                    index = earlier[-1] if earlier else available[0]
                clip.append(frames[index].copy())
            clips.append(clip)
        return clips

    @staticmethod
    def _open_capture(video_path, attempts=3):
        """Open a video, retrying briefly before giving up.

        A single open can fail transiently under IO contention -- a DFD file
        that failed a full scan opened fine seconds later -- so a retry
        distinguishes a busy filesystem from a genuinely broken file.
        """
        for attempt in range(int(attempts)):
            capture = cv2.VideoCapture(str(video_path))
            if capture.isOpened():
                return capture
            capture.release()
            if attempt + 1 < int(attempts):
                time.sleep(0.25 * (attempt + 1))
        raise UnreadableVideoError("Unable to open video: {}".format(video_path))

    def _read_clips(self, video_path):
        capture = self._open_capture(video_path)
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        key = str(video_path)
        frame_count = self.frame_count_cache.get(key)
        if frame_count is None:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        clip_indices = self.clip_indices(frame_count)
        try:
            decoded_clips = self._decode_clips(capture, video_path, clip_indices)
            if decoded_clips is None:
                # Metadata or the seek index lied. Measure the real length once,
                # remember it for later epochs, and decode without seeking.
                measured = self._measure_decodable_frames(capture)
                if measured < 1:
                    raise UnreadableVideoError(
                        "Video has no decodable frames; it is likely truncated or "
                        "corrupt: {}".format(video_path)
                    )
                self.frame_count_cache[key] = measured
                clip_indices = self.clip_indices(measured)
                decoded_clips = self._decode_clips_sequential(
                    capture, video_path, clip_indices
                )
        finally:
            capture.release()
        clips, audits = [], []
        for frames, indices in zip(decoded_clips, clip_indices):
            boxes, points, audit = self._prepare_boxes(frames, indices, video_path)
            flip = self.training and np.random.random() < self.horizontal_flip_probability
            tensors = []
            for frame, box, landmark in zip(frames, boxes, points):
                if self.align_landmarks:
                    # Align first, then take the box, per the protocol's step
                    # order. Rotating about the box centre leaves the box valid.
                    frame = self._align_on_eyes(frame, box, landmark)
                tensors.append(self._transform(self._crop_box(frame, box), flip))
            clips.append(torch.stack(tensors, dim=1))
            audits.append(audit)
        return clips, audits, fps if np.isfinite(fps) and fps > 0.0 else 0.0

    def __getitem__(self, index):
        record = self.records[index]
        video_path = self.dataset_roots[record["dataset"]] / record["path"]
        try:
            clips, audits, fps = self._read_clips(video_path)
        except UnreadableVideoError as error:
            # Signal the collate function to drop this item. Every other error
            # still propagates, so real bugs are not swallowed.
            self.unreadable.add(str(video_path))
            return {"_skip_video": True, "path": str(video_path),
                    "error": str(error)}
        result = {
            "label": torch.tensor(int(record["label"]), dtype=torch.long),
            "path": str(video_path),
            "relative_path": record["path"],
            "dataset": record["dataset"],
            "method": record["method"],
            "target_id": record.get("target_id", ""),
            "donor_id": record.get("donor_id", ""),
            "source_clip": record.get("source_clip", ""),
            "fps": torch.tensor(fps, dtype=torch.float32),
            "clip_face_any_miss": torch.tensor([item["any_miss"] for item in audits], dtype=torch.float32),
            "clip_face_miss_fraction": torch.tensor([item["miss_fraction"] for item in audits], dtype=torch.float32),
            "clip_box_center_x_jitter": torch.tensor([item["center_x_jitter"] for item in audits], dtype=torch.float32),
            "clip_box_center_y_jitter": torch.tensor([item["center_y_jitter"] for item in audits], dtype=torch.float32),
            "clip_box_width_jitter": torch.tensor([item["width_jitter"] for item in audits], dtype=torch.float32),
            "clip_box_height_jitter": torch.tensor([item["height_jitter"] for item in audits], dtype=torch.float32),
        }
        if self.training:
            result["clip"] = clips[0]
        else:
            result["clips"] = torch.stack(clips, dim=0)
        return result


class CachedClipDataset(Dataset):
    """Decode each video once and keep the tensors in RAM.

    Only for the overfit diagnostic. Decoding dominates the epoch (Haar
    detection alone is ~93% of per-item time), so caching turns "can this
    architecture fit 40 videos" into a question answered in minutes instead of
    hours. Requires num_workers=0, otherwise each respawned worker starts with
    an empty cache.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.cache = {}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if index not in self.cache:
            self.cache[index] = self.dataset[index]
        return self.cache[index]
