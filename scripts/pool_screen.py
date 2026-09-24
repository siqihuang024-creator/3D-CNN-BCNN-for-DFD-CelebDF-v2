"""DFD screen S: five final spatial pools with trained/random linear probes."""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from feature_diagnostics import collect_features, probe, spread_by_identity
from video_bcnn.data import load_manifest, preflight_face_cache
from video_bcnn.experiment import active_records, make_dataset, select_records
from video_bcnn.model import build_feature_extractor
from video_bcnn.reporting import json_safe
from video_bcnn.utils import (load_checkpoint, load_config, override_dataset_roots,
                              resolve_device, save_json, seed_everything)


CANDIDATES = (("avg4x4", "avg", [4, 4]), ("max4x4", "max", [4, 4]),
              ("avg4x7", "avg", [4, 7]), ("avg8x14", "avg", [8, 14]),
              ("avg22x22", "avg", [22, 22]))


def recommend_final_pool(trained_delta, random_deltas):
    """'max' only when the trained delta beats every random delta; NaN -> undetermined."""
    values = [trained_delta] + list(random_deltas)
    if not values[1:] or not all(np.isfinite(float(value)) for value in values):
        return "undetermined"
    return "max" if trained_delta > max(random_deltas) else "avg"


def best_probe(features, labels, identities, seed):
    """Best out-of-fold AUROC over the ridge grid, and the whole curve beside it.

    The headline is a maximum over five ridge strengths scored on the same
    folds, so it is optimistic, and unevenly so: a 15,488-dimensional candidate
    is far more sensitive to the penalty than a 512-dimensional one when only a
    few hundred videos are available. Comparing pooling sizes on the maximum
    alone would read that difference as information content, so the per-strength
    curve is reported too and belongs in any such comparison.
    """
    results, identity_count = probe(features, labels, identities, folds=5, seed=seed)
    if not results:
        return float("nan"), identity_count, {}
    curve = {str(strength): float(item["out_of_fold"])
             for strength, item in sorted(results.items())}
    return max(item["out_of_fold"] for item in results.values()), identity_count, curve


def candidate_config(config, final_pool, size):
    result = copy.deepcopy(config)
    result["model"].update({"architecture": "3d_cnn", "temporal_head": "mean",
                            "stage_pool_type": "avg", "final_pool_type": final_pool,
                            "spatial_output_size": size,
                            "feature_dim": 32 * size[0] * size[1]})
    return result


def extract_candidate(config, records, state, device):
    extractor = build_feature_extractor(config["model"]).to(device)
    incompatible = extractor.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Checkpoint is not a compatible E1/E2 mean extractor: {}"
                           .format(incompatible))
    dataset = make_dataset(records, config, training=False, clips_per_video=4)
    return collect_features(extractor, dataset, device, clip_chunk_size=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--dataset-root", action="append", default=None)
    parser.add_argument("--random-draws", type=int, default=8)
    parser.add_argument("--output", default="artifacts/v2/pool_screen_dfd.json")
    args = parser.parse_args()
    runtime = override_dataset_roots(load_config(args.config), args.dataset_root)
    # Screen S was written for DFD, where whole frames made the final pool the
    # open question. The same measurement answers a different question on
    # CelebDF++: holding one trained extractor fixed, how much of what a linear
    # probe could read at 22x22 survives 4x4? Re-pooling costs no training, so
    # the screen is the cheap way to separate "the bottleneck destroys the
    # signal" from "the smaller model simply learned a little less".
    active = runtime["data"].get("active_datasets") or []
    if len(active) != 1:
        raise ValueError("Screen S needs exactly one active dataset, found {}."
                         .format(active))
    device = resolve_device(runtime.get("device", "cuda"))
    seed = int(runtime.get("seed", 42))
    seed_everything(seed)
    saved = load_checkpoint(args.checkpoint, device)
    config = copy.deepcopy(saved["config"])
    config["data"]["dataset_roots"] = runtime["data"]["dataset_roots"]
    config["data"]["active_datasets"] = list(active)
    rows = select_records(active_records(load_manifest(args.manifest), config), args.split)
    reals = spread_by_identity([row for row in rows if int(row["label"]) == 1],
                               sum(int(row["label"]) == 1 for row in rows), seed)
    fakes = spread_by_identity([row for row in rows if int(row["label"]) == 0],
                               len(reals), seed)
    rows = reals + fakes
    preflight_face_cache(rows, config)
    trained_state = saved["extractor"]
    report = {"checkpoint": str(Path(args.checkpoint).resolve()), "split": args.split,
              "videos": len(rows), "real": len(reals), "fake": len(fakes),
              "clips_per_video": 4,
              "scope": ("Exploratory diagnostic: a balanced identity-spread subset "
                        "at 4 clips per video, scored by frozen-feature linear "
                        "probes. These numbers are not comparable with the "
                        "full-validation AUROC of a retrained model, and the "
                        "recommendation below covers avg vs max at one size, not "
                        "which pooling size to train with."),
              "trained": {}, "random": {}, "criterion": {}}
    trained_scores = {}
    for name, final_pool, size in CANDIDATES:
        current = candidate_config(config, final_pool, size)
        features, labels, identities = extract_candidate(
            current, rows, trained_state, device)
        value, identity_count, curve = best_probe(features, labels, identities, seed)
        trained_scores[name] = value
        report["trained"][name] = {"held_out_probe_auroc": value,
                                    "probe_auroc_by_ridge": curve,
                                    "feature_dim": int(features.shape[1]),
                                    "identities": identity_count}
    if min(item["identities"] for item in report["trained"].values()) < 10:
        report["warning"] = "Fewer than 10 identities; the screen is unstable."

    deltas = []
    for draw in range(int(args.random_draws)):
        torch.manual_seed(seed + 1000 + draw)
        # One random convolutional state is reused across all pooling choices.
        reference_config = candidate_config(config, "avg", [22, 22])
        reference = build_feature_extractor(reference_config["model"])
        random_state = reference.state_dict()
        draw_scores = {}
        for name, final_pool, size in CANDIDATES:
            current = candidate_config(config, final_pool, size)
            features, labels, identities = extract_candidate(
                current, rows, random_state, device)
            draw_scores[name], _, _ = best_probe(features, labels, identities, seed + draw)
        delta = draw_scores["max4x4"] - draw_scores["avg4x4"]
        deltas.append(delta)
        report["random"][str(draw + 1)] = {"probes": draw_scores,
                                            "delta_max_minus_avg": delta}
    trained_delta = trained_scores["max4x4"] - trained_scores["avg4x4"]
    report["criterion"] = {
        "delta_trained": trained_delta,
        "delta_random": deltas,
        "delta_random_min": float(np.min(deltas)),
        "delta_random_mean": float(np.mean(deltas)),
        "delta_random_max": float(np.max(deltas)),
        "recommend_final_pool_type": recommend_final_pool(trained_delta, deltas),
        "rule": "choose max only when trained delta exceeds every random-init delta",
    }
    save_json(args.output, json_safe(report))
    print(json.dumps(json_safe(report), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
