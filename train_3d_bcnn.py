"""Phase C: real-only Bayesian anomaly modelling on frozen Phase-A features."""

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pyro
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from video_bcnn.data import load_manifest, seed_worker, skip_unreadable_collate
from video_bcnn.evaluation import score_bayesian
from video_bcnn.experiment import active_records, make_dataset, select_records
from video_bcnn.features import FeatureCacheDataset
from video_bcnn.metrics import calibrate_threshold, detection_metrics
from video_bcnn.model import VideoBayesianCNN, build_feature_extractor, freeze_extractor
from video_bcnn.reporting import json_safe, save_history
from video_bcnn.utils import (apply_experiment, ensure_dir, load_checkpoint, load_config,
                              override_dataset_roots, override_num_workers,
                              resolve_device, runtime_metadata, save_json,
                              seed_everything, verify_dataset_roots)


def build_model(config, device):
    extractor = build_feature_extractor(config["model"]).to(device)
    bayes = config.get("bayesian", config.get("model", {}))
    model = VideoBayesianCNN(
        extractor, hidden_dims=bayes.get("hidden_dims", [256, 64]),
        dropout=bayes.get("dropout", 0.2), prior_std=bayes.get("prior_std", 0.1),
        observation_std=bayes.get("observation_std", 1.0),
        rho_init=bayes.get("rho_init", -5.0),
        kl_weight=bayes.get("kl_weight", 5e-4))
    return extractor, model


def _options(config, device):
    workers = int(config["data"].get("num_workers", 0))
    result = {"num_workers": workers, "pin_memory": device.type == "cuda",
              "worker_init_fn": seed_worker, "collate_fn": skip_unreadable_collate}
    if workers > 0:
        result["persistent_workers"] = bool(config["data"].get("persistent_workers", True))
    return result


def _metrics(values, fpr):
    labels, scores = values["labels"], values["scores"]
    threshold = calibrate_threshold(scores[labels == 1], fpr)
    result = detection_metrics(labels, scores, threshold)
    per_dataset = {}
    for name in sorted(set(values["datasets"].tolist())):
        mask = values["datasets"] == name
        per_dataset[name] = detection_metrics(labels[mask], scores[mask], threshold)
    result["per_dataset"] = per_dataset
    result["macro_dataset_auroc"] = float(np.nanmean(
        [item["auroc"] for item in per_dataset.values()]))
    result["embedding_variance_mean"] = values["embedding_variance_mean"]
    return result, threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--init-extractor", default=None,
                        help="Phase-A best.pt. May be omitted only for optional E7b frozen pretrained MC3.")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--experiment", default=None,
                        help="Optional extractor override from configs/v2/experiment_matrix.yaml.")
    parser.add_argument("--matrix", default=str(ROOT / "configs" / "v2" / "experiment_matrix.yaml"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dataset-root", action="append", default=None, metavar="NAME=PATH")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--smoke-test", type=int, default=None, metavar="BATCHES")
    parser.add_argument("--feature-cache", default=None,
                        help="Optional .npz from scripts/cache_video_features.py.")
    args = parser.parse_args()

    config = apply_experiment(load_config(args.config), args.matrix, args.experiment)
    if args.seed is not None:
        config["seed"] = int(args.seed)
    config = override_dataset_roots(config, args.dataset_root)
    override_num_workers(config, args.num_workers)
    run_dir = Path(args.run_dir or config["train"]["run_dir"])
    if args.run_dir is None and args.experiment:
        dataset_tag = "-".join(name.lower() for name in config["data"].get("active_datasets", []))
        run_dir = Path(config["train"]["run_dir"]).parent / (
            "{}_e6_{}_seed{}".format(dataset_tag, args.experiment.lower(),
                                     config.get("seed", 42)))
    config["train"]["run_dir"] = str(run_dir)
    device = resolve_device(config.get("device", "cuda"))
    seed_everything(int(config.get("seed", 42)))
    verify_dataset_roots(config)
    pyro.clear_param_store()

    records = active_records(load_manifest(args.manifest), config)
    train_records = select_records(records, "train", label=1)
    val_records = select_records(records, "val")
    if not train_records or not val_records:
        raise ValueError("Phase C needs real training videos and a two-class validation split.")
    if args.smoke_test:
        per_class = max(2, int(args.smoke_test))
        val_records = ([row for row in val_records if int(row["label"]) == 1][:per_class] +
                       [row for row in val_records if int(row["label"]) == 0][:per_class])
    train_set = (FeatureCacheDataset(args.feature_cache, train_records)
                 if args.feature_cache else make_dataset(train_records, config, training=True))
    val_set = make_dataset(val_records, config, training=False,
                           clips_per_video=config["data"].get("selection_clips_per_video", 8))
    options = _options(config, device)
    train_loader = DataLoader(train_set,
                              batch_size=int(config["train"].get("physical_batch_size", 8)),
                              shuffle=True, **options)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, **options)

    extractor, model = build_model(config, device)
    if args.init_extractor:
        phase_a = load_checkpoint(args.init_extractor, device)
        extractor.load_state_dict(phase_a["extractor"], strict=True)
        config["model"]["init_extractor"] = str(Path(args.init_extractor).resolve())
    elif config["model"].get("architecture") != "mc3_18":
        raise ValueError("--init-extractor is required except for frozen pretrained MC3 (E7b).")
    else:
        config["model"]["init_extractor"] = "torchvision KINETICS400_V1 (E7b)"
    freeze_extractor(extractor)
    optimizer = ClippedAdam({"lr": float(config["train"].get("learning_rate", 1e-3)),
                             "clip_norm": float(config["train"].get("gradient_clip_norm", 5.0)),
                             "lrd": float(config["train"].get("lr_decay", 0.98))})
    svi = SVI(model.model, model.guide, optimizer, loss=Trace_ELBO())

    checkpoint_dir, log_dir = ensure_dir(run_dir / "checkpoints"), ensure_dir(run_dir / "logs")
    metadata = runtime_metadata(config, ROOT)
    metadata["precision"] = "fp32"
    save_json(run_dir / "config.json", json_safe(config))
    save_json(run_dir / "runtime.json", metadata)
    print("Phase C on {}: {} real training videos; extractor frozen in eval mode; "
          "Bayesian head {}->256->64->1, fp32.".format(
          device, len(train_records), extractor.feature_dim))

    history, best, best_epoch = [], -float("inf"), 0
    epochs = 1 if args.smoke_test else int(config["train"].get("epochs", 50))
    mc_samples = int(config["train"].get("mc_samples", 30))
    for epoch in range(1, epochs + 1):
        started = time.time()
        extractor.eval()  # never allow frozen BatchNorm running statistics to drift
        if any(parameter.requires_grad for parameter in extractor.parameters()):
            raise RuntimeError("Frozen extractor unexpectedly contains trainable parameters.")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        total, batches, skipped = 0.0, 0, 0
        progress = tqdm(train_loader, desc="Phase C {}/{}".format(epoch, epochs))
        for step, batch in enumerate(progress):
            if args.smoke_test is not None and step >= args.smoke_test:
                break
            if batch is None:
                skipped += 1
                continue
            if "feature" in batch:
                features = batch["feature"].to(device, non_blocking=True)
            else:
                clips = batch["clip"].to(device, non_blocking=True)
                with torch.no_grad():
                    features = extractor(clips).detach()
            targets = torch.ones(features.shape[0], device=device)
            loss = float(svi.step(features, targets, len(train_records)))
            if not np.isfinite(loss):
                raise FloatingPointError("Non-finite ELBO at epoch {}.".format(epoch))
            total += loss
            batches += 1
            progress.set_postfix(elbo="{:.4f}".format(total / batches))

        values = score_bayesian(
            model, val_loader, device, mc_samples,
            clip_chunk_size=int(config["data"].get("eval_clip_chunk_size", 4)))
        metrics, threshold = _metrics(values, config["train"].get("calibration_fpr", 0.05))
        diagnostics = model.diagnostics()
        diagnostics["activation_stages"] = copy.deepcopy(extractor.last_activation_stats)
        diagnostics["temporal"] = copy.deepcopy(getattr(extractor, "last_temporal_stats", {}))
        row = {"epoch": epoch, "train_loss": total / max(1, batches),
               "skipped_training_clips": skipped,
               "learning_rate": float(config["train"].get("learning_rate", 1e-3)),
               "selection_metric": "auroc", "selection_value": metrics["auroc"],
               "epoch_duration_seconds": time.time() - started,
               "peak_vram_bytes": (torch.cuda.max_memory_allocated(device)
                                    if device.type == "cuda" else 0),
               "validation": metrics, "posterior_diagnostics": diagnostics}
        history.append(json_safe(row))
        payload = {"stage": "phase_c", "epoch": epoch,
                   "feature_extractor": extractor.state_dict(),
                   "pyro_params": pyro.get_param_store().get_state(),
                   "config": copy.deepcopy(config), "validation": metrics,
                   "threshold": threshold, "runtime": metadata, "score_sign": -1.0}
        torch.save(payload, checkpoint_dir / "last.pt")
        if best_epoch == 0 or metrics["auroc"] > best:
            best, best_epoch = metrics["auroc"], epoch
            torch.save(payload, checkpoint_dir / "best.pt")
        save_history(history, log_dir)
        print("val AUROC={:.4f}; fake AP={:.4f} ({:.2f}x); real AP={:.4f} "
              "({:.2f}x); sigma mean={:.3g}.".format(
              metrics["auroc"], metrics["fake_average_precision"],
              metrics["fake_ap_lift"], metrics["real_average_precision"],
              metrics["real_ap_lift"], diagnostics["sigma_mean"]))
    print("Best Phase-C epoch {} AUROC {:.4f}: {}".format(
        best_epoch, best, (checkpoint_dir / "best.pt").resolve()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
