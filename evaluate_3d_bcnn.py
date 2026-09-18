"""Evaluate a V2 Phase-A or Phase-C checkpoint at video level."""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pyro
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from video_bcnn.data import load_manifest, seed_worker, skip_unreadable_collate
from video_bcnn.evaluation import score_bayesian, score_deterministic
from video_bcnn.experiment import active_records, make_dataset, select_records
from video_bcnn.metrics import (calibrate_threshold, clustered_auroc_interval,
                                detection_metrics)
from video_bcnn.model import DeterministicHead, build_feature_extractor, freeze_extractor
from video_bcnn.reporting import save_evaluation_report
from video_bcnn.utils import (load_checkpoint, load_config, override_dataset_roots,
                              override_num_workers, resolve_device, seed_everything)
from train_3d_bcnn import build_model


def evaluate(values, threshold, draws, seed):
    labels, scores = values["labels"], values["scores"]
    metrics = detection_metrics(labels, scores, threshold)
    metrics["clustered_bootstrap"] = clustered_auroc_interval(
        1 - labels, scores, values["source_clips"], draws=draws, seed=seed)
    metrics["per_dataset"] = {}
    for dataset in sorted(set(values["datasets"].tolist())):
        mask = values["datasets"] == dataset
        metrics["per_dataset"][dataset] = detection_metrics(
            labels[mask], scores[mask], threshold)
    metrics["per_method"] = {}
    for dataset in sorted(set(values["datasets"].tolist())):
        real = (values["datasets"] == dataset) & (labels == 1)
        methods = sorted(set(values["methods"][(values["datasets"] == dataset) &
                                               (labels == 0)].tolist()))
        for method in methods:
            fake = (values["datasets"] == dataset) & (values["methods"] == method) & (labels == 0)
            mask = real | fake
            metrics["per_method"]["{}/{}".format(dataset, method)] = detection_metrics(
                labels[mask], scores[mask], threshold)
    metrics["macro_dataset_auroc"] = float(np.nanmean(
        [item["auroc"] for item in metrics["per_dataset"].values()]))
    metrics["num_videos"] = int(len(labels))
    metrics["embedding_variance_mean"] = values.get("embedding_variance_mean")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True,
                        help="Runtime config; checkpoint keeps all preprocessing settings.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--dataset-root", action="append", default=None, metavar="NAME=PATH")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--mc-samples", type=int, default=None)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--max-videos", type=int, default=None,
                        help="Balanced debug subset; omit for every final result.")
    parser.add_argument("--recalibrate-threshold", action="store_true")
    parser.add_argument("--export-embeddings", action="store_true")
    args = parser.parse_args()

    runtime = override_dataset_roots(load_config(args.config), args.dataset_root)
    override_num_workers(runtime, args.num_workers)
    device = resolve_device(runtime.get("device", "cuda"))
    checkpoint = load_checkpoint(args.checkpoint, device)
    config = copy.deepcopy(checkpoint["config"])
    config["data"]["dataset_roots"] = runtime["data"]["dataset_roots"]
    config["data"]["num_workers"] = runtime["data"].get("num_workers", 0)
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    records = select_records(active_records(load_manifest(args.manifest), config), args.split)
    if args.max_videos is not None:
        half = max(1, int(args.max_videos) // 2)
        records = ([row for row in records if int(row["label"]) == 1][:half] +
                   [row for row in records if int(row["label"]) == 0][:half])
    dataset = make_dataset(records, config, training=False,
                           clips_per_video=config["data"].get("eval_clips_per_video", 8))
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=int(config["data"].get("num_workers", 0)),
                        pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
                        collate_fn=skip_unreadable_collate)

    stage = checkpoint.get("stage", "phase_c")
    if stage == "phase_a":
        extractor = build_feature_extractor(config["model"]).to(device)
        extractor.load_state_dict(checkpoint["extractor"])
        head = DeterministicHead(extractor.feature_dim,
                                 config["model"].get("deterministic_hidden_dim", 0),
                                 config["model"].get("dropout", 0.2)).to(device)
        head.load_state_dict(checkpoint["head"])
        raw = score_deterministic(
            extractor, head, loader, device,
            int(config["data"].get("eval_clip_chunk_size", 4)), use_amp=True)
        values = dict(raw)
        values.update({"scores": -raw["logits"], "means": raw["logits"],
                       "stds": np.zeros_like(raw["logits"]),
                       "embedding_norms": np.full(len(raw["labels"]), np.nan)})
    elif stage == "phase_c":
        pyro.clear_param_store()
        extractor, model = build_model(config, device)
        extractor.load_state_dict(checkpoint["feature_extractor"])
        freeze_extractor(extractor)
        pyro.get_param_store().set_state(checkpoint["pyro_params"])
        samples = int(args.mc_samples if args.mc_samples is not None else
                      config["train"].get("mc_samples", 30))
        values = score_bayesian(
            model, loader, device, samples,
            int(config["data"].get("eval_clip_chunk_size", 4)),
            collect_embeddings=args.export_embeddings)
    else:
        raise ValueError("Unknown checkpoint stage {!r}.".format(stage))

    threshold = checkpoint.get("threshold")
    if args.recalibrate_threshold or threshold is None:
        threshold = calibrate_threshold(values["scores"][values["labels"] == 1],
                                        config["train"].get("calibration_fpr", 0.05))
    metrics = evaluate(values, float(threshold), args.bootstrap_draws, seed)
    metrics.update({"split": args.split, "stage": stage,
                    "checkpoint": str(Path(args.checkpoint).resolve()),
                    "threshold_source": ("evaluation_reals" if args.recalibrate_threshold
                                         else "checkpoint"),
                    "eval_batch_size": 1,
                    "eval_clip_chunk_size": int(config["data"].get("eval_clip_chunk_size", 4))})
    checkpoint_dir = Path(args.checkpoint).resolve().parent
    report_dir = checkpoint_dir.parent / "reports"
    report = save_evaluation_report(values, metrics, report_dir, args.split)
    if args.export_embeddings and "embeddings" in values:
        np.savez_compressed(report_dir / "{}_embeddings.npz".format(args.split),
                            embeddings=values["embeddings"], labels=values["labels"],
                            paths=np.asarray(values["paths"]))
    print("{} AUROC={:.4f}, 95% clustered CI [{:.4f}, {:.4f}]".format(
        args.split, metrics["auroc"], metrics["clustered_bootstrap"]["low"],
        metrics["clustered_bootstrap"]["high"]))
    print("Report: {}".format(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
