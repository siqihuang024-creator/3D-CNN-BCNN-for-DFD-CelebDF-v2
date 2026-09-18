"""Phase A: supervised representation learning for the V2 experiment matrix."""

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from video_bcnn.data import load_manifest, seed_worker, skip_unreadable_collate
from video_bcnn.evaluation import score_deterministic
from video_bcnn.experiment import (BalancedBatchSampler, GroupBalancedEpochSampler,
                                   active_records, make_dataset, select_records)
from video_bcnn.metrics import calibrate_threshold, detection_metrics
from video_bcnn.model import DeterministicHead, build_feature_extractor
from video_bcnn.reporting import json_safe, save_history
from video_bcnn.utils import (apply_experiment, ensure_dir, load_config, override_dataset_roots,
                              override_num_workers, resolve_device, runtime_metadata,
                              save_json, seed_everything, verify_dataset_roots)


def _loader_options(config, device):
    workers = int(config["data"].get("num_workers", 0))
    result = {"num_workers": workers, "pin_memory": device.type == "cuda",
              "worker_init_fn": seed_worker, "collate_fn": skip_unreadable_collate}
    if workers > 0:
        result["persistent_workers"] = bool(config["data"].get("persistent_workers", True))
    return result


def _per_dataset(labels, anomaly, datasets, threshold):
    output = {}
    for name in sorted(set(datasets)):
        mask = np.asarray(datasets) == name
        output[name] = detection_metrics(labels[mask], anomaly[mask], threshold)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--experiment", default=None,
                        help="Entry from configs/v2/experiment_matrix.yaml, e.g. E3 or E4_attention.")
    parser.add_argument("--matrix", default=str(ROOT / "configs" / "v2" / "experiment_matrix.yaml"))
    parser.add_argument("--dataset-root", action="append", default=None, metavar="NAME=PATH")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke-test", type=int, default=None, metavar="BATCHES")
    args = parser.parse_args()

    config = apply_experiment(load_config(args.config), args.matrix, args.experiment)
    if args.seed is not None:
        config["seed"] = int(args.seed)
    config = override_dataset_roots(config, args.dataset_root)
    override_num_workers(config, args.num_workers)
    default_run = config["train"]["run_dir"]
    if args.experiment:
        dataset_tag = "-".join(name.lower() for name in config["data"].get("active_datasets", []))
        default_run = str(Path(default_run).parent /
                          ("{}_{}_seed{}".format(dataset_tag, args.experiment.lower(),
                                                config.get("seed", 42))))
    run_dir = Path(args.run_dir or default_run)
    config["train"]["run_dir"] = str(run_dir)
    device = resolve_device(config.get("device", "cuda"))
    seed_everything(int(config.get("seed", 42)))
    verify_dataset_roots(config)

    records = active_records(load_manifest(args.manifest), config)
    train_records = select_records(records, "train")
    val_records = select_records(records, "val")
    if not train_records or not val_records:
        raise ValueError("Manifest must contain non-empty train and val splits.")
    if {int(row["label"]) for row in train_records} != {0, 1}:
        raise ValueError("Phase A needs both real and fake training videos.")
    if args.smoke_test:
        # Manifests are sorted real-first; truncating the loader would otherwise
        # score one class only and never exercise AUROC/AP/checkpoint selection.
        per_class = max(2, int(args.smoke_test))
        val_records = ([row for row in val_records if int(row["label"]) == 1][:per_class] +
                       [row for row in val_records if int(row["label"]) == 0][:per_class])

    train_set = make_dataset(train_records, config, training=True)
    val_set = make_dataset(val_records, config, training=False,
                           clips_per_video=config["data"].get("selection_clips_per_video", 8))
    physical = int(config["train"].get("physical_batch_size", 8))
    balance_keys = tuple(config["data"].get("train_balance_keys",
                                           ["dataset", "class_name"]))
    group_count = len({tuple(row[key] for key in balance_keys) for row in train_records})
    options = _loader_options(config, device)
    if physical >= group_count and physical % group_count == 0:
        sampler = BalancedBatchSampler(
            train_records, physical, int(config.get("seed", 42)),
            group_keys=balance_keys,
            batches_per_epoch=config["train"].get("batches_per_epoch"))
        train_loader = DataLoader(train_set, batch_sampler=sampler, **options)
        batch_protocol = "every physical batch balanced"
    else:
        # E0/E1 deliberately use batch=1, so balance can only be enforced over
        # the epoch. E2 is the intervention that first balances each batch.
        sampler = GroupBalancedEpochSampler(
            train_records, int(config.get("seed", 42)),
            config["train"].get("samples_per_group", "max"),
            group_keys=balance_keys)
        train_loader = DataLoader(train_set, batch_size=physical, sampler=sampler,
                                  shuffle=False, **options)
        batch_protocol = "epoch-balanced (batch=1 baseline)"
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, **options)

    extractor = build_feature_extractor(config["model"]).to(device)
    head = DeterministicHead(extractor.feature_dim,
                             config["model"].get("deterministic_hidden_dim", 0),
                             config["model"].get("dropout", 0.2)).to(device)
    parameters = list(extractor.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(parameters,
                                  lr=float(config["train"].get("learning_rate", 1e-4)),
                                  weight_decay=float(config["train"].get("weight_decay", 1e-4)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config["train"].get("epochs", 30)))
    criterion = nn.BCEWithLogitsLoss()
    accumulation = int(config["train"].get("gradient_accumulation_steps", 1))
    use_amp = config["train"].get("precision", "amp") == "amp" and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    clip_norm = float(config["train"].get("gradient_clip_norm", 5.0))

    checkpoint_dir, log_dir = ensure_dir(run_dir / "checkpoints"), ensure_dir(run_dir / "logs")
    metadata = runtime_metadata(config, ROOT)
    save_json(run_dir / "config.json", json_safe(config))
    save_json(run_dir / "runtime.json", metadata)
    print("Phase A on {}: {} training / {} validation videos; physical={} accum={} "
          "effective={} precision={}.".format(device, len(train_records), len(val_records),
          physical, accumulation, physical * accumulation, metadata["precision"]))
    print("Sampling protocol: {}.".format(batch_protocol))

    history, best, best_epoch = [], -float("inf"), 0
    epochs = 1 if args.smoke_test else int(config["train"].get("epochs", 30))
    for epoch in range(1, epochs + 1):
        started = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        extractor.train()
        head.train()
        optimizer.zero_grad(set_to_none=True)
        running, seen, skipped, optimizer_updates = 0.0, 0, 0, 0
        progress = tqdm(train_loader, desc="Phase A {}/{}".format(epoch, epochs))
        last_step = -1
        for step, batch in enumerate(progress):
            if args.smoke_test is not None and step >= args.smoke_test:
                break
            last_step = step
            if batch is None:
                skipped += 1
                continue
            clips = batch["clip"].to(device, non_blocking=True)
            targets = batch["label"].to(device, non_blocking=True).float()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = head(extractor(clips))
                loss = criterion(logits, targets) / accumulation
            scaler.scale(loss).backward()
            if (step + 1) % accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, clip_norm)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_updates += int(scaler.get_scale() >= scale_before)
                optimizer.zero_grad(set_to_none=True)
            running += float(loss.detach()) * accumulation * targets.numel()
            seen += targets.numel()
            progress.set_postfix(bce="{:.4f}".format(running / max(1, seen)))
        if last_step >= 0 and (last_step + 1) % accumulation:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, clip_norm)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_updates += int(scaler.get_scale() >= scale_before)
            optimizer.zero_grad(set_to_none=True)
        if optimizer_updates:
            scheduler.step()

        scored = score_deterministic(
            extractor, head, val_loader, device,
            clip_chunk_size=int(config["data"].get("eval_clip_chunk_size", 4)),
            use_amp=use_amp,
            limit=None)
        labels, anomaly = scored["labels"], -scored["logits"]
        threshold = calibrate_threshold(anomaly[labels == 1],
                                        config["train"].get("calibration_fpr", 0.05))
        metrics = detection_metrics(labels, anomaly, threshold)
        # Dataset names follow the same order as a batch-size-one validation loader.
        per_dataset = _per_dataset(labels, anomaly, scored["datasets"], threshold)
        metrics["per_dataset"] = per_dataset
        metrics["macro_dataset_auroc"] = float(np.nanmean(
            [item["auroc"] for item in per_dataset.values()]))
        metrics["embedding_variance_mean"] = scored["embedding_variance_mean"]
        duration = time.time() - started
        diagnostics = {"activation_stages": copy.deepcopy(extractor.last_activation_stats),
                       "temporal": copy.deepcopy(getattr(extractor, "last_temporal_stats", {}))}
        row = {"epoch": epoch, "train_loss": running / max(1, seen),
               "skipped_training_clips": skipped,
               "learning_rate": optimizer.param_groups[0]["lr"],
               "selection_metric": "auroc", "selection_value": metrics["auroc"],
               "epoch_duration_seconds": duration,
               "peak_vram_bytes": (torch.cuda.max_memory_allocated(device)
                                    if device.type == "cuda" else 0),
               "validation": metrics, "diagnostics": diagnostics}
        history.append(json_safe(row))
        payload = {"stage": "phase_a", "epoch": epoch,
                   "extractor": extractor.state_dict(), "head": head.state_dict(),
                   "config": copy.deepcopy(config), "validation": metrics,
                   "threshold": threshold, "runtime": metadata}
        torch.save(payload, checkpoint_dir / "last.pt")
        if best_epoch == 0 or metrics["auroc"] > best:
            best, best_epoch = metrics["auroc"], epoch
            torch.save(payload, checkpoint_dir / "best.pt")
        save_history(history, log_dir)
        print("val AUROC={:.4f}, fake AP={:.4f} ({:.2f}x), real AP={:.4f} "
              "({:.2f}x), embedding-var={:.3g}".format(
              metrics["auroc"], metrics["fake_average_precision"],
              metrics["fake_ap_lift"], metrics["real_average_precision"],
              metrics["real_ap_lift"], metrics["embedding_variance_mean"]))
    print("Best Phase-A epoch {} AUROC {:.4f}: {}".format(
        best_epoch, best, (checkpoint_dir / "best.pt").resolve()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
