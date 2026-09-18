"""Find this machine's best DataLoader worker count by measuring it.

More workers is not automatically faster here. Clip loading is dominated by
Haar face detection on full-resolution frames (~93% of per-item time on DFD's
1920x1080 videos), which is memory-bandwidth bound, and OpenCV already
parallelises it internally. Extra worker processes then contend for the same
bandwidth while multiplying resident frame buffers.

Run this on the target machine before committing to a value:

    python scripts/benchmark_loader.py --config configs/dfd_supervised_3d.yaml \
      --manifest artifacts/manifests/combined_manifest_supervised.csv

Each configuration is warmed until every worker has spun up and filled its
prefetch queue, so the reported figure is steady-state throughput rather than
process startup.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
from torch.utils.data import DataLoader

from video_bcnn.data import VideoClipDataset, seed_worker
from video_bcnn.utils import load_config, override_dataset_roots, verify_dataset_roots


CV2_THREADS = None


def worker_init(worker_id):
    seed_worker(worker_id)
    if CV2_THREADS is not None:
        cv2.setNumThreads(int(CV2_THREADS))


def build_dataset(config, manifest, dataset, videos):
    data_config = dict(config["data"])
    data_config.update({
        "input_resize": config["model"]["input_resize"],
        "center_crop": config["model"]["center_crop"],
    })
    with open(manifest, "r", newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["split"] == "train" and (not dataset or row["dataset"] == dataset)
        ]
    if not rows:
        raise ValueError("No training rows found for dataset={!r}.".format(dataset))
    return VideoClipDataset(
        rows[:videos], config["data"]["dataset_roots"], data_config, training=True
    )


def measure(dataset, workers, items):
    loader = DataLoader(
        dataset, batch_size=1, num_workers=workers,
        worker_init_fn=worker_init, shuffle=False,
    )
    iterator = iter(loader)
    for _ in range(max(workers * 2, 8)):
        next(iterator)
    start = time.time()
    for _ in range(items):
        next(iterator)
    elapsed = time.time() - start
    del iterator, loader
    return elapsed / items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset", default=None,
                        help="Restrict to one dataset; defaults to the config's first active one.")
    parser.add_argument("--workers", type=int, nargs="+", default=[2, 4, 8],
                        help="Worker counts to compare.")
    parser.add_argument("--cv2-threads", type=int, default=None,
                        help="Force OpenCV threads per worker. Leave unset to keep "
                             "OpenCV's default, which measured fastest on a 20-core box.")
    parser.add_argument("--items", type=int, default=32, help="Timed items per configuration.")
    parser.add_argument("--clips-per-epoch", type=int, default=1226,
                        help="Used only to project minutes per epoch.")
    parser.add_argument("--dataset-root", action="append", default=None, metavar="NAME=PATH")
    args = parser.parse_args()

    global CV2_THREADS
    CV2_THREADS = args.cv2_threads

    config = override_dataset_roots(load_config(args.config), args.dataset_root)
    verify_dataset_roots(config)
    active = config["data"].get("active_datasets") or []
    dataset_name = args.dataset or (active[0] if active else None)
    videos = max(args.items + max(args.workers) * 2 + 8, 64)
    dataset = build_dataset(config, args.manifest, dataset_name, videos)

    print("CPU cores: {} | OpenCV default threads: {} | dataset: {} | cv2 threads/worker: {}".format(
        os.cpu_count(), cv2.getNumThreads(), dataset_name, args.cv2_threads or "default"
    ))
    print("\n{:<9} {:>10} {:>18}".format("workers", "s/item", "est. min/epoch"))
    results = {}
    for workers in args.workers:
        try:
            per_item = measure(dataset, workers, args.items)
        except (RuntimeError, MemoryError, cv2.error) as error:
            print("{:<9} {:>10} {:>18}".format(workers, "FAILED", str(error)[:40]))
            continue
        results[workers] = per_item
        print("{:<9} {:>9.3f}s {:>17.1f}".format(
            workers, per_item, per_item * args.clips_per_epoch / 60.0
        ))
    if results:
        best = min(results, key=results.get)
        print("\nFastest: num_workers={} ({:.3f} s/item). Set it with "
              "`export NUM_WORKERS={}` or --num-workers {}.".format(
                  best, results[best], best, best))


if __name__ == "__main__":
    main()
