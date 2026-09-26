"""Ask whether a temporal head reads the order of its input sequence.

E4 recovered what the 512-D bottleneck cost, but "the TCN helped" and "the TCN
used temporal order" are different claims: a residual block adds capacity and
moves the normalisation whether or not it reads order. The control is to keep
the checkpoint fixed and permute the per-frame sequence that reaches the head.

Permuting raw frames would also scramble the trunk's own three-frame
convolutions, so the drop could not be attributed to the head. This script
permutes [f_1 ... f_T] between the trunk and the head instead, and because
everything before that point is identical for every permutation, the expensive
half -- decoding and the 3D-CNN -- runs once and is cached. Twenty permutations
then cost seconds rather than twenty more passes over the split.

Conditions beyond a plain shuffle earn their place: reversing time keeps every
local transition intact but inverts their direction, and block shuffling keeps
windows of w feature positions contiguous while permuting the windows. The
TCN sees seven feature positions; thirteen is the combined raw-frame receptive
field of trunk plus TCN. Small perturbation effects do not prove independence
or equivalence: the cached features already contain local temporal information.

Usage:
    python scripts/temporal_order_control.py \
        --config configs/v2/phase_a_celeb.yaml \
        --manifest artifacts/manifests/combined_manifest_p05.csv \
        --checkpoint artifacts/v2/celebdfv3_e4_gap_seed42/checkpoints/best.pt \
        --split val --expected-videos 8205 --draws 2000 \
        --output results/v2/E4_temporal_order_control.json
"""

import argparse
import copy
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from compare_experiments import (BOOTSTRAP_METRICS, choose_clusters,  # noqa: E402
                                 ranking_metrics, read_score_file)
from video_bcnn.data import (load_manifest, preflight_face_cache,  # noqa: E402
                             seed_worker, skip_unreadable_collate)
from video_bcnn.experiment import active_records, make_dataset, select_records  # noqa: E402
from video_bcnn.model import DeterministicHead, build_feature_extractor  # noqa: E402
from video_bcnn.reporting import json_safe  # noqa: E402
from video_bcnn.utils import (load_checkpoint, load_config,  # noqa: E402
                              override_dataset_roots, override_num_workers,
                              resolve_device, save_json, seed_everything)


@torch.no_grad()
def cache_sequences(extractor, loader, device, chunk=4):
    """Run the trunk once and keep the per-clip sequences the head will read."""
    extractor.eval()
    sequences, meta = [], []
    for batch in tqdm(loader, desc="Caching pre-TCN sequences"):
        if batch is None or batch.get("_skip_only", False):
            raise RuntimeError("Unreadable video; refusing a partial order control.")
        if batch.get("_skipped_paths"):
            raise RuntimeError("Skipped video; refusing a partial order control.")
        clips = batch["clips"][0]
        parts = []
        for start in range(0, clips.shape[0], chunk):
            part = clips[start:start + chunk].to(device, non_blocking=True)
            # Match full-val AMP, without introducing extra FP16 quantisation.
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                parts.append(extractor.trunk_sequence(part).cpu())
        sequences.append(torch.cat(parts, 0))          # [clips, steps, channels]
        meta.append({"label_real": int(batch["label"][0]),
                     "identity": batch["target_id"][0],
                     "source_family_id": batch["source_clip"][0],
                     "forgery_method": batch["method"][0],
                     "relative_path": batch["relative_path"][0],
                     "dataset": batch["dataset"][0],
                     "video_id": "{}::{}".format(batch["dataset"][0],
                         batch["relative_path"][0].replace("\\", "/"))})
    return sequences, meta


@torch.no_grad()
def score(sequences, extractor, head, device, permute=None, generator=None,
          bypass_head=False, metadata=None, condition_seed=None, chunk=4):
    """Video scores (higher = more fake) under one ordering of the sequence."""
    values = []
    extractor.eval()
    head.eval()
    for video_index, item in enumerate(sequences):
        batch = item.to(device)
        if permute is not None:
            orders = []
            for index in range(batch.shape[0]):
                rng = (sample_generator(condition_seed, metadata[video_index]["video_id"], index)
                       if condition_seed is not None else generator)
                orders.append(batch[index][permute(batch.shape[1], rng).to(device)])
            batch = torch.stack(orders)
        parts = []
        # Same chunking, autocast boundaries and FP32 video mean as full-val.
        for start in range(0, len(batch), chunk):
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                part = batch[start:start + chunk]
                features = (extractor.aggregator(part) if bypass_head
                            else extractor.temporal_head_forward(part))
            parts.append(features.float())
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = head(torch.cat(parts, 0)).float()
        values.append(float(-logits.mean()))           # anomaly = -logit
    return np.asarray(values)


def sample_generator(seed, video_id, clip_index):
    """Stable across loader ordering, chunk size, skipped videos and processes."""
    key = "{}|{}|{}".format(seed, video_id, clip_index).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2 ** 63)
    return torch.Generator().manual_seed(value)


def verify_reference(metadata, ordered, path, atol=1e-4):
    table = read_score_file(path)
    ids = [row["video_id"] for row in metadata]
    if len(ids) != len(set(ids)) or set(ids) != set(table):
        raise ValueError("Ordered control and reference do not have identical video IDs.")
    if any(row["label_real"] != table[row["video_id"]]["label_real"] for row in metadata):
        raise ValueError("Ordered reference labels disagree.")
    reference = np.asarray([table[key]["video_score"] for key in ids])
    error = float(np.max(np.abs(reference - ordered)))
    if not np.isfinite(error) or error > atol:
        raise ValueError("Ordered baseline differs from full-val: max score error {:.6g} "
                         "> tolerance {}. Do not interpret perturbations.".format(error, atol))
    return {"path": str(Path(path).resolve()), "max_abs_score_error": error,
            "absolute_tolerance": atol, "videos": len(ids), "passed": True}


def control_intervals(labels, conditions, clusters, draws, seed):
    """Paired identity draws; mean-of-shuffle AUC, NOT AUC of averaged scores."""
    members = [np.flatnonzero(clusters == key) for key in sorted(set(clusters))]
    rng = np.random.default_rng(seed)
    names = [name for name in conditions if name != "ordered"]
    samples = {name: {metric: [] for metric in BOOTSTRAP_METRICS} for name in names}
    mean_samples = {metric: [] for metric in BOOTSTRAP_METRICS}
    shuffles = [name for name in names if name.startswith("shuffled_")]
    valid = 0
    for _ in range(draws):
        indices = np.concatenate([members[index] for index in
                                  rng.integers(len(members), size=len(members))])
        if np.unique(labels[indices]).size < 2:
            continue
        metrics = {name: ranking_metrics(labels[indices], values[indices])
                   for name, values in conditions.items()}
        for name in names:
            for metric in BOOTSTRAP_METRICS:
                samples[name][metric].append(metrics[name][metric] - metrics["ordered"][metric])
        for metric in BOOTSTRAP_METRICS:
            mean_samples[metric].append(metrics["ordered"][metric] -
                np.mean([metrics[name][metric] for name in shuffles]))
        valid += 1
    if not valid:
        raise ValueError("No valid two-class bootstrap draws.")
    def interval(values):
        return {"mean": float(np.mean(values)), "low": float(np.quantile(values, .025)),
                "high": float(np.quantile(values, .975))}
    return ({name: {metric: interval(values) for metric, values in metrics.items()}
             for name, metrics in samples.items()},
            {metric: interval(values) for metric, values in mean_samples.items()}, valid)


def random_order(steps, generator):
    return torch.randperm(steps, generator=generator)


def reversed_order(steps, generator):
    del generator
    return torch.arange(steps - 1, -1, -1)


def block_order(width):
    def order(steps, generator):
        if not 0 < width < steps:
            raise ValueError("Block width must be positive and smaller than sequence length.")
        blocks = [torch.arange(start, min(start + width, steps))
                  for start in range(0, steps, width)]
        permutation = torch.randperm(len(blocks), generator=generator)
        # In particular block16 at T=32 must swap, not leave half the videos
        # unchanged. Record this non-identity convention in the report.
        while torch.equal(permutation, torch.arange(len(blocks))):
            permutation = torch.randperm(len(blocks), generator=generator)
        return torch.cat([blocks[int(index)] for index in permutation])
    return order


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--dataset-root", action="append", default=None, metavar="NAME=PATH")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-videos", type=int, default=None,
                        help="Debug subset; omit for a result that counts.")
    parser.add_argument("--expected-videos", type=int, default=None)
    parser.add_argument("--shuffle-seeds", type=int, default=10,
                        help="How many random permutations to average over.")
    parser.add_argument("--block-widths", type=int, nargs="*", default=[4, 8, 16],
                        help="Block shuffle widths; 0 disables. 16 of 32 steps keeps "
                             "half the clip contiguous and only swaps the halves, so it "
                             "isolates long-range arrangement from local continuity.")
    parser.add_argument("--skip-bypass", action="store_true",
                        help="Omit the TCN-bypass diagnostic (aggregate the sequence "
                             "directly, same head). It measures how much the temporal "
                             "head contributes at all, which a shuffle cannot: a "
                             "residual block plus temporal GAP averages its identity "
                             "path unchanged whatever the order.")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--reference-scores", default=None,
                        help="Ordered full-val CSV. Defaults to checkpoint run/reports/full_val_video_scores.csv.")
    parser.add_argument("--reference-atol", type=float, default=1e-4)
    parser.add_argument("--output", default="results/v2/temporal_order_control.json")
    args = parser.parse_args()
    if args.shuffle_seeds < 1 or args.draws < 1 or args.reference_atol < 0:
        parser.error("shuffle-seeds/draws must be positive and reference-atol nonnegative.")
    if any(width < 0 for width in args.block_widths):
        parser.error("block-widths cannot be negative.")
    reference = (Path(args.reference_scores) if args.reference_scores else
                 Path(args.checkpoint).parent.parent / "reports" /
                 ("full_val_video_scores.csv" if args.split == "val" else "test_video_scores.csv"))
    if args.max_videos is None and not reference.is_file():
        raise FileNotFoundError("Run the ordered full evaluation first: {}".format(reference))

    runtime = override_dataset_roots(load_config(args.config), args.dataset_root)
    override_num_workers(runtime, args.num_workers)
    device = resolve_device(runtime.get("device", "cuda"))
    checkpoint = load_checkpoint(args.checkpoint, device)
    if checkpoint.get("stage") != "phase_a":
        raise ValueError("This control reads a Phase-A checkpoint.")
    config = copy.deepcopy(checkpoint["config"])
    config["data"]["dataset_roots"] = runtime["data"]["dataset_roots"]
    config["data"]["num_workers"] = runtime["data"].get("num_workers", 0)
    if config["model"].get("temporal_head") != "tcn":
        raise ValueError(
            "The mean head averages over time and is permutation invariant, so "
            "this control is vacuous for it; it is defined for a TCN head.")
    seed = int(config.get("seed", 42))
    seed_everything(seed)

    records = select_records(active_records(load_manifest(args.manifest), config), args.split)
    if args.max_videos is not None:
        half = max(1, int(args.max_videos) // 2)
        records = ([row for row in records if int(row["label"]) == 1][:half] +
                   [row for row in records if int(row["label"]) == 0][:half])
    if args.expected_videos is not None and len(records) != args.expected_videos:
        raise ValueError("Expected {} videos, found {}.".format(
            args.expected_videos, len(records)))
    preflight_face_cache(records, config)
    dataset = make_dataset(records, config, training=False,
                           clips_per_video=config["data"].get("eval_clips_per_video", 8))
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=int(config["data"].get("num_workers", 0)),
                        pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
                        collate_fn=skip_unreadable_collate)

    extractor = build_feature_extractor(config["model"]).to(device)
    extractor.load_state_dict(checkpoint["extractor"])
    head = DeterministicHead(extractor.feature_dim,
                             config["model"].get("deterministic_hidden_dim", 0),
                             config["model"].get("dropout", 0.2)).to(device)
    head.load_state_dict(checkpoint["head"])
    head.eval()

    print("caching trunk sequences for {} videos ...".format(len(records)))
    sequences, meta = cache_sequences(
        extractor, loader, device, int(config["data"].get("eval_clip_chunk_size", 4)))
    if args.expected_videos is not None and len(sequences) != args.expected_videos:
        raise RuntimeError("Scored {} of {} videos; refusing a partial control.".format(
            len(sequences), args.expected_videos))
    labels = np.asarray([row["label_real"] for row in meta], dtype=np.int64)
    if not sequences or np.unique(labels).size != 2:
        raise ValueError("Order control needs nonempty real and fake videos.")
    cluster_key, clusters = choose_clusters(meta)
    steps = int(sequences[0].shape[1])
    print("cached {} videos, {} steps per clip".format(len(sequences), steps))

    if any(width >= steps for width in args.block_widths):
        raise ValueError("Block width must be smaller than sequence length.")
    chunk = int(config["data"].get("eval_clip_chunk_size", 4))
    conditions = {"ordered": score(sequences, extractor, head, device, chunk=chunk)}
    reference_check = (verify_reference(meta, conditions["ordered"], reference, args.reference_atol)
                       if args.max_videos is None else {"passed": None, "scope": "debug-subset"})
    condition_seeds = {}
    for draw in range(int(args.shuffle_seeds)):
        name = "shuffled_{}".format(draw)
        condition_seeds[name] = seed + draw
        conditions[name] = score(sequences, extractor, head, device, random_order,
            metadata=meta, condition_seed=seed + draw, chunk=chunk)
    conditions["reversed"] = score(sequences, extractor, head, device, reversed_order, chunk=chunk)
    if not args.skip_bypass:
        conditions["tcn_bypass"] = score(sequences, extractor, head, device,
                                         bypass_head=True, chunk=chunk)
    for width in [w for w in args.block_widths if w]:
        conditions["block{}".format(width)] = score(
            sequences, extractor, head, device, block_order(int(width)),
            metadata=meta, condition_seed=seed, chunk=chunk)

    report = {"checkpoint": str(Path(args.checkpoint).resolve()),
              "split": args.split, "videos": len(sequences), "steps": steps,
              "real": int((labels == 1).sum()), "fake": int((labels == 0).sum()),
              "shuffle_seeds": int(args.shuffle_seeds),
              "condition_seeds": condition_seeds, "seed": seed,
              "cluster_key": cluster_key, "ordered_reference_check": reference_check,
              "evaluation_scope": "debug-subset" if args.max_videos else "full-" + args.split,
              "inference_precision": "cuda-amp" if device.type == "cuda" else "float32",
              "cache_dtypes": sorted({str(item.dtype) for item in sequences}),
              "clip_chunk_size": chunk, "delta_definition": "condition minus ordered",
              "block_permutation": "non-identity; block16 at T32 always swaps halves",
              "bypass_caveat": "Inference ablation with the SAME trained classifier; distribution "
                               "shift, not a retrained capacity control or causal decomposition.",
              "conditions": {}, "deltas_vs_ordered": {},
              "note": ("The permutation is applied between the trunk and the temporal "
                       "head, so every frame feature is unchanged and only their "
                       "arrangement differs. Little observed drop does not prove order "
                       "independence: features retain trunk temporal context, residual GAP "
                       "can dilute effects, and the CI may be wide.")}
    for name, values in conditions.items():
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite scores in condition {}.".format(name))
        report["conditions"][name] = ranking_metrics(labels, values)
    deltas, mean_interval, valid = control_intervals(labels, conditions, clusters, args.draws, seed)
    report["deltas_vs_ordered"] = deltas
    for name, metrics in deltas.items():
        for metric, interval in metrics.items():
            interval["point"] = report["conditions"][name][metric] - report["conditions"]["ordered"][metric]
    shuffle_names = [name for name in conditions if name.startswith("shuffled_")]
    for metric, interval in mean_interval.items():
        interval["point"] = report["conditions"]["ordered"][metric] - np.mean(
            [report["conditions"][name][metric] for name in shuffle_names])
    report["bootstrap"] = {"draws_requested": args.draws, "draws_valid": valid,
                           "clusters": len(set(clusters)),
                           "scope": "paired video-cluster uncertainty conditional on fixed shuffle seeds"}

    shuffles = [v["auroc"] for k, v in report["conditions"].items()
                if k.startswith("shuffled_")]
    report["shuffle_summary"] = {
        "ordered_auroc": report["conditions"]["ordered"]["auroc"],
        "shuffled_auroc_mean": float(np.mean(shuffles)) if shuffles else None,
        "shuffled_auroc_std": float(np.std(shuffles, ddof=1)) if len(shuffles) > 1 else 0.0,
        "mean_drop_ci": mean_interval,
        "mean_drop_definition": "ordered metric minus mean of per-seed metrics (not mean scores)",
        "mean_drop": (report["conditions"]["ordered"]["auroc"] - float(np.mean(shuffles))
                      if shuffles else None)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    score_path = output.with_name(output.stem + "_video_scores.csv")
    with score_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(meta[0]) + list(conditions))
        writer.writeheader()
        for index, row in enumerate(meta):
            writer.writerow(dict(row, **{name: values[index] for name, values in conditions.items()}))
    report["video_scores_csv"] = str(score_path)
    save_json(output, json_safe(report))

    print("\n%-14s %8s %10s   %s" % ("condition", "AUROC", "vs ordered", "95% CI"))
    print("%-14s %8.4f" % ("ordered", report["conditions"]["ordered"]["auroc"]))
    for name in sorted(report["deltas_vs_ordered"]):
        interval = report["deltas_vs_ordered"][name]["auroc"]
        print("%-14s %8.4f %+10.4f   [%+0.4f, %+0.4f]%s" % (
            name, report["conditions"][name]["auroc"],
            report["conditions"][name]["auroc"] - report["conditions"]["ordered"]["auroc"],
            interval["low"], interval["high"],
            "" if interval["high"] < 0 or interval["low"] > 0 else "  (covers 0)"))
    summary = report["shuffle_summary"]
    if summary["shuffled_auroc_mean"] is not None:
        print("\nrandom shuffles: %.4f +/- %.4f over %d seeds; mean drop %.4f" % (
            summary["shuffled_auroc_mean"], summary["shuffled_auroc_std"],
            args.shuffle_seeds, summary["mean_drop"]))
        interval = summary["mean_drop_ci"]["auroc"]
        print("ordered minus mean shuffle AUC: 95%% CI [%+.4f, %+.4f]" %
              (interval["low"], interval["high"]))
    print("wrote {}".format(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
