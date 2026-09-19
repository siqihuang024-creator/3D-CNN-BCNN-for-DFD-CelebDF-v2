"""Cache 16 reproducible random clip features per real Phase-C video."""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from video_bcnn.data import (load_manifest, preflight_face_cache)
from video_bcnn.evaluation import extract_clip_features
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
    parser.add_argument("--clips-per-video", type=int, default=16)
    args = parser.parse_args()
    runtime = override_dataset_roots(load_config(args.config), args.dataset_root)
    device = resolve_device(runtime.get("device", "cuda"))
    checkpoint = load_checkpoint(args.checkpoint, device)
    # The checkpoint is authoritative for the selected E4'/E7 architecture;
    # the command-line config only supplies this machine's dataset roots.
    config = copy.deepcopy(checkpoint["config"])
    config["data"]["dataset_roots"] = runtime["data"]["dataset_roots"]
    seed_everything(int(config.get("seed", 42)))
    extractor = build_feature_extractor(config["model"]).to(device)
    weights = checkpoint.get("extractor", checkpoint.get("feature_extractor"))
    if weights is None:
        raise KeyError("Checkpoint contains no extractor weights.")
    extractor.load_state_dict(weights, strict=True)
    freeze_extractor(extractor)
    records = select_records(active_records(load_manifest(args.manifest), config),
                             "train", label=1)
    preflight_face_cache(records, config)
    # training=True supplies random position, stride in {1,2}, and one
    # clip-consistent horizontal flip. The global seed makes the 16 draws fixed.
    dataset = make_dataset(records, config, training=True)
    features, paths, datasets, clip_indices = [], [], [], []
    for record_index, row in enumerate(tqdm(records, desc="Caching real videos")):
        for clip_index in range(int(args.clips_per_video)):
            item = dataset[record_index]
            if item.get("_skip_video", False):
                raise RuntimeError("Unreadable cache input: {}".format(item["path"]))
            vector = extract_clip_features(
                extractor, item["clip"].unsqueeze(0), device, 1, use_amp=False)
            features.append(vector.cpu().numpy())
            paths.append(row["path"])
            datasets.append(row["dataset"])
            clip_indices.append(clip_index)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, features=np.concatenate(features),
                        paths=np.asarray(paths), datasets=np.asarray(datasets),
                        clip_indices=np.asarray(clip_indices, dtype=np.int64))
    metadata = {"checkpoint": str(Path(args.checkpoint).resolve()),
                "manifest": str(Path(args.manifest).resolve()),
                "videos": len(records), "training_units": len(features),
                "feature_dim": int(features[0].shape[1]),
                "clips_per_video": int(args.clips_per_video),
                "sampling": "random start; stride {1,2}; clip-consistent flip p=0.5",
                "seed": int(config.get("seed", 42))}
    with open(str(output) + ".json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(metadata)
    return 0


if __name__ == "__main__":
    sys.exit(main())
