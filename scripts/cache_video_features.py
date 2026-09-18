"""Cache one deterministic Phase-C feature vector per real training video."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from video_bcnn.data import load_manifest, skip_unreadable_collate
from video_bcnn.evaluation import extract_video_features
from video_bcnn.experiment import active_records, make_dataset, select_records
from video_bcnn.model import build_feature_extractor, freeze_extractor
from video_bcnn.utils import (load_checkpoint, load_config, override_dataset_roots,
                              resolve_device, seed_everything)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-root", action="append", default=None)
    parser.add_argument("--clips-per-video", type=int, default=8)
    args = parser.parse_args()
    config = override_dataset_roots(load_config(args.config), args.dataset_root)
    device = resolve_device(config.get("device", "cuda"))
    seed_everything(int(config.get("seed", 42)))
    checkpoint = load_checkpoint(args.checkpoint, device)
    extractor = build_feature_extractor(config["model"]).to(device)
    weights = checkpoint.get("extractor", checkpoint.get("feature_extractor"))
    if weights is None:
        raise KeyError("Checkpoint contains no extractor weights.")
    extractor.load_state_dict(weights, strict=True)
    freeze_extractor(extractor)
    records = select_records(active_records(load_manifest(args.manifest), config),
                             "train", label=1)
    dataset = make_dataset(records, config, training=False,
                           clips_per_video=args.clips_per_video)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=skip_unreadable_collate)
    record_by_absolute_path = {
        str((Path(config["data"]["dataset_roots"][row["dataset"]]) /
             row["path"]).resolve()): row for row in records}
    features, paths, datasets = [], [], []
    for batch in tqdm(loader, desc="Caching real-video features"):
        if batch is None:
            continue
        vector = extract_video_features(
            extractor, batch["clips"][0], device,
            int(config["data"].get("eval_clip_chunk_size", 4)), use_amp=False)
        features.append(vector.cpu().numpy())
        # Store manifest-relative keys, not host-specific absolute paths.
        row = record_by_absolute_path[batch["path"][0]]
        paths.append(row["path"])
        datasets.append(row["dataset"])
    if len(features) != len(records):
        raise RuntimeError("{} of {} real videos were unreadable; cache is incomplete."
                           .format(len(records) - len(features), len(records)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, features=np.concatenate(features),
                        paths=np.asarray(paths), datasets=np.asarray(datasets))
    metadata = {"checkpoint": str(Path(args.checkpoint).resolve()),
                "manifest": str(Path(args.manifest).resolve()),
                "videos": len(records), "feature_dim": int(features[0].shape[1]),
                "clips_per_video": args.clips_per_video}
    with open(str(output) + ".json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(metadata)
    return 0


if __name__ == "__main__":
    sys.exit(main())
