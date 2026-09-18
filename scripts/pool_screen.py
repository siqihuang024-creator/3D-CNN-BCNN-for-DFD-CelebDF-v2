"""DFD diagnostic screen S: compare final spatial pooling without training."""

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from video_bcnn.data import load_manifest, skip_unreadable_collate
from video_bcnn.evaluation import extract_video_features
from video_bcnn.experiment import active_records, make_dataset, select_records
from video_bcnn.model import build_feature_extractor
from video_bcnn.reporting import json_safe
from video_bcnn.utils import (load_checkpoint, load_config, override_dataset_roots,
                              resolve_device, save_json, seed_everything)


CANDIDATES = (("avg4x4", "avg", [4, 4]), ("max4x4", "max", [4, 4]),
              ("avg4x7", "avg", [4, 7]), ("avg8x14", "avg", [8, 14]),
              ("avg22x22", "avg", [22, 22]))


def describe(matrix, labels):
    norms = matrix.norm(dim=1)
    spread = matrix.std(dim=0, unbiased=False).mean()
    per_dimension = norms.mean() / math.sqrt(matrix.shape[1])
    real, fake = matrix[labels == 1], matrix[labels == 0]
    gap = (real.mean(0) - fake.mean(0)).norm()
    within = torch.cat([(real - real.mean(0)).norm(dim=1),
                        (fake - fake.mean(0)).norm(dim=1)]).mean()
    return {"videos": len(matrix), "feature_dim": matrix.shape[1],
            "embedding_variance_mean": float(matrix.var(0, unbiased=False).mean()),
            "relative_variation": float(spread / per_dimension.clamp_min(1e-12)),
            "centroid_separation_ratio": float(gap / within.clamp_min(1e-12))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True,
                        help="E2 Phase-A checkpoint; only convolutional weights are reused.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--max-videos", type=int, default=80)
    parser.add_argument("--dataset-root", action="append", default=None)
    parser.add_argument("--output", default="artifacts/v2/pool_screen_dfd.json")
    args = parser.parse_args()
    config = override_dataset_roots(load_config(args.config), args.dataset_root)
    if config["data"].get("active_datasets") != ["DFD"]:
        raise ValueError("Pool screen S is defined for DFD only.")
    device = resolve_device(config.get("device", "cuda"))
    seed_everything(int(config.get("seed", 42)))
    rows = select_records(active_records(load_manifest(args.manifest), config), args.split)
    reals = [row for row in rows if int(row["label"]) == 1][:args.max_videos // 2]
    fakes = [row for row in rows if int(row["label"]) == 0][:args.max_videos // 2]
    rows = reals + fakes
    dataset = make_dataset(rows, config, training=False, clips_per_video=2)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=skip_unreadable_collate)
    saved = load_checkpoint(args.checkpoint, device)
    report = {"checkpoint": str(Path(args.checkpoint).resolve()), "split": args.split,
              "candidates": {}}
    for name, pool, size in CANDIDATES:
        model_config = copy.deepcopy(config["model"])
        model_config.update({"temporal_head": "mean", "spatial_pool_type": pool,
                             "spatial_output_size": size,
                             "feature_dim": 32 * size[0] * size[1]})
        extractor = build_feature_extractor(model_config).to(device)
        incompatible = extractor.load_state_dict(saved["extractor"], strict=False)
        # Pooling has no parameters; missing/unexpected keys indicate a wrong checkpoint.
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError("Checkpoint is not an E2 mean-head extractor: {}".format(incompatible))
        extractor.eval()
        features, labels = [], []
        for batch in loader:
            if batch is None:
                continue
            vector = extract_video_features(extractor, batch["clips"][0], device, 2)
            features.append(vector.cpu())
            labels.append(int(batch["label"][0]))
        report["candidates"][name] = describe(torch.cat(features), np.asarray(labels))
    save_json(args.output, json_safe(report))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
