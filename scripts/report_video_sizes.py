"""Scan video files and export their dimensions and file sizes to CSV.

The two dataset roots below match the local Windows layout used by this
project.  Override them with repeated ``--root NAME=PATH`` arguments when the
datasets are moved to another machine.
"""

import argparse
import collections
import csv
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2


DEFAULT_ROOTS = (
    ("CelebDFv3", Path(r"E:\PhD\Deepfake Video TCN+BCNN\Datasets\CelebDFv3")),
    ("DFD-Kaggle", Path(r"E:\PhD\Deepfake Video TCN+BCNN\Datasets\DFD-Kaggle")),
)
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}
DETAIL_FIELDS = (
    "dataset",
    "relative_path",
    "width",
    "height",
    "resolution",
    "frame_count",
    "fps",
    "duration_seconds",
    "file_size_bytes",
    "file_size_mb",
    "status",
    "error",
)


def parse_root(value):
    """Parse NAME=PATH from the command line."""
    if "=" not in value:
        raise argparse.ArgumentTypeError("--root 必须使用 NAME=PATH 格式")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("--root 的名称和路径都不能为空")
    return name.strip(), Path(raw_path.strip())


def find_videos(roots):
    """Return deterministic (dataset, root, path) records."""
    records = []
    for dataset, root in roots:
        if not root.is_dir():
            raise FileNotFoundError("数据集目录不存在: {}".format(root))
        paths = (
            path for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        )
        records.extend((dataset, root, path) for path in sorted(paths))
    return records


def finite_positive(value):
    return math.isfinite(value) and value > 0


def inspect_video(record):
    """Read one video's container metadata, decoding one frame only as fallback."""
    dataset, root, path = record
    row = {
        "dataset": dataset,
        "relative_path": str(path.relative_to(root)),
        "width": "",
        "height": "",
        "resolution": "",
        "frame_count": "",
        "fps": "",
        "duration_seconds": "",
        "file_size_bytes": "",
        "file_size_mb": "",
        "status": "ok",
        "error": "",
    }
    try:
        size_bytes = path.stat().st_size
        row["file_size_bytes"] = size_bytes
        row["file_size_mb"] = round(size_bytes / (1024.0 * 1024.0), 3)

        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise RuntimeError("OpenCV 无法打开视频")

            width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
            height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            fps = float(capture.get(cv2.CAP_PROP_FPS))

            # Some containers omit dimensions in metadata. Decode only the
            # first frame in that uncommon case, rather than decoding every file.
            if width <= 0 or height <= 0:
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise RuntimeError("无法读取视频分辨率或第一帧")
                height, width = frame.shape[:2]

            row["width"] = width
            row["height"] = height
            row["resolution"] = "{}x{}".format(width, height)
            if frames >= 0:
                row["frame_count"] = frames
            if finite_positive(fps):
                row["fps"] = round(fps, 6)
                if frames >= 0:
                    row["duration_seconds"] = round(frames / fps, 3)
        finally:
            capture.release()
    except Exception as exc:
        row["status"] = "error"
        row["error"] = str(exc)
    return row


def write_summary(path, resolution_counts, dataset_totals, error_counts):
    fields = ("dataset", "width", "height", "resolution", "video_count")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (dataset, width, height), count in sorted(
                resolution_counts.items(), key=lambda item: (item[0][0], -item[1], item[0][1], item[0][2])):
            writer.writerow({
                "dataset": dataset,
                "width": width,
                "height": height,
                "resolution": "{}x{}".format(width, height),
                "video_count": count,
            })

    print("\n扫描结果:")
    for dataset in sorted(dataset_totals):
        print("  {}: {} 个视频，{} 个错误".format(
            dataset, dataset_totals[dataset], error_counts[dataset]))


def main():
    parser = argparse.ArgumentParser(
        description="统计视频分辨率、帧数、FPS、时长和磁盘文件大小。")
    parser.add_argument(
        "--root", action="append", type=parse_root, default=None,
        metavar="NAME=PATH",
        help="数据集名称与目录；可重复使用。未指定时扫描本机的两个默认目录。")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("video_size_reports"),
        help="CSV 输出目录（默认: ./video_size_reports）。")
    parser.add_argument(
        "--workers", type=int, default=min(8, max(1, os.cpu_count() or 1)),
        help="并发读取视频头的线程数（默认最多 8）。")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="每个数据集最多扫描多少个视频；适合先做小规模测试。")
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers 必须大于或等于 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit 必须大于或等于 1")

    roots = args.root or list(DEFAULT_ROOTS)
    records = find_videos(roots)
    if args.limit is not None:
        limited = []
        per_dataset = collections.Counter()
        for record in records:
            if per_dataset[record[0]] < args.limit:
                limited.append(record)
                per_dataset[record[0]] += 1
        records = limited

    if not records:
        print("没有找到支持的视频文件。", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "video_sizes.csv"
    summary_path = args.output_dir / "resolution_summary.csv"
    resolution_counts = collections.Counter()
    dataset_totals = collections.Counter()
    error_counts = collections.Counter()

    print("找到 {:,} 个视频，使用 {} 个线程读取元数据。".format(
        len(records), args.workers))
    started = time.time()
    with detail_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETAIL_FIELDS)
        writer.writeheader()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for index, row in enumerate(executor.map(inspect_video, records), 1):
                writer.writerow(row)
                dataset_totals[row["dataset"]] += 1
                if row["status"] == "ok":
                    resolution_counts[(row["dataset"], row["width"], row["height"])] += 1
                else:
                    error_counts[row["dataset"]] += 1
                if index % 500 == 0 or index == len(records):
                    print("  已处理 {:,}/{:,} ({:.1f}%)".format(
                        index, len(records), index * 100.0 / len(records)), flush=True)

    write_summary(summary_path, resolution_counts, dataset_totals, error_counts)
    print("耗时: {:.1f} 秒".format(time.time() - started))
    print("逐视频明细: {}".format(detail_path.resolve()))
    print("分辨率汇总: {}".format(summary_path.resolve()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
