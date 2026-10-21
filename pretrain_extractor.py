"""Phase A: supervised representation learning for the V2 experiment matrix."""

import argparse
import copy
import hashlib
import re
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

from video_bcnn.data import (load_manifest, preflight_face_cache, seed_worker,
                             skip_unreadable_collate)
from video_bcnn.evaluation import score_deterministic
from video_bcnn.experiment import (BalancedBatchSampler, PoolPassBatchSampler,
                                   GroupBalancedEpochSampler,
                                   active_records, capped_validation_records,
                                   make_dataset, select_records)
from video_bcnn.metrics import (calibrate_threshold, detection_metrics,
                                probability_half_metrics)
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
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Total epochs of a NEW independent run, not checkpoint resume.")
    parser.add_argument("--run-suffix", default=None,
                        help="Separate run name such as E4_long. Starts from scratch, not resume.")
    parser.add_argument("--experiment", default=None,
                        help="Entry from configs/v2/experiment_matrix.yaml, e.g. E3 or E4_attention.")
    parser.add_argument("--matrix", default=str(ROOT / "configs" / "v2" / "experiment_matrix.yaml"))
    parser.add_argument("--profile-input-pipeline", action="store_true",
                        help="Record per-epoch loader-wait versus compute time. Adds a CUDA "
                             "sync per step, so use it for the feasibility run, not for a "
                             "result that will be reported.")
    parser.add_argument("--dataset-root", action="append", default=None, metavar="NAME=PATH")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--temporal-aggregation", choices=[
        "gap", "max", "attention", "cls", "flatten", "flatten64"], default=None)
    parser.add_argument("--smoke-test", type=int, default=None, metavar="BATCHES")
    args = parser.parse_args()
    if args.max_epochs is not None and args.max_epochs < 1:
        parser.error("max-epochs must be positive.")
    if args.run_suffix and not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_suffix):
        parser.error("run-suffix must contain only letters, digits, underscores or hyphens.")

    config = apply_experiment(load_config(args.config), args.matrix, args.experiment)
    if (config["train"].get("epoch_protocol") in ("pool_pass", "balanced_pool_pass") and
            int(config["data"].get("selection_clips_per_video", 4)) !=
            int(config["data"].get("eval_clips_per_video", 8))):
        raise ValueError("The revised protocol requires the same clips per "
                         "video for selection-threshold calibration and full "
                         "validation accuracy.")
    if args.max_epochs is not None:
        config["train"]["epochs"] = args.max_epochs
    if args.seed is not None:
        config["seed"] = int(args.seed)
    if args.temporal_aggregation is not None:
        config["model"]["temporal_aggregation"] = args.temporal_aggregation
    config = override_dataset_roots(config, args.dataset_root)
    override_num_workers(config, args.num_workers)
    default_run = config["train"]["run_dir"]
    if args.experiment:
        dataset_tag = "-".join(name.lower() for name in config["data"].get("active_datasets", []))
        aggregation = config["model"].get("temporal_aggregation", "gap")
        suffix = args.experiment.lower()
        if not suffix.endswith("_" + aggregation):
            suffix += "_" + aggregation
        default_run = str(Path(default_run).parent /
                          ("{}_{}_seed{}".format(dataset_tag, suffix,
                                                config.get("seed", 42))))
    run_dir = Path(args.run_dir or default_run)
    if args.run_suffix:
        dataset_tag = "-".join(name.lower() for name in config["data"].get("active_datasets", []))
        run_dir = Path(args.run_dir) if args.run_dir else run_dir.parent / (
            "{}_{}_seed{}".format(dataset_tag, args.run_suffix.lower(), config.get("seed", 42)))
    if (args.max_epochs is not None or args.run_suffix or
            config["train"].get("epoch_protocol") == "balanced_pool_pass"):
        if (run_dir / "checkpoints" / "best.pt").exists() or (run_dir / "logs" / "history.csv").exists():
            raise FileExistsError("Independent extended run would overwrite existing results: {}".format(run_dir))
        config["train"]["optimization_scope"] = (
            "independent-group-run (not resume)"
            if config["train"].get("epoch_protocol") == "balanced_pool_pass"
            else "independent-extended-budget-run (not resume)")
    config["train"]["run_dir"] = str(run_dir)
    device = resolve_device(config.get("device", "cuda"))
    seed_everything(int(config.get("seed", 42)))
    verify_dataset_roots(config)

    records = active_records(load_manifest(args.manifest), config)
    train_records = select_records(records, "train")
    full_val_records = select_records(records, "val")
    if not train_records or not full_val_records:
        raise ValueError("Manifest must contain non-empty train and val splits.")
    if {int(row["label"]) for row in train_records} != {0, 1}:
        raise ValueError("Phase A needs both real and fake training videos.")
    if config["train"].get("epoch_protocol") == "balanced_pool_pass":
        by_split = {split: select_records(records, split) for split in ("train", "val", "test")}
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            left_paths = {(row["dataset"], row["path"].replace("\\", "/").casefold())
                          for row in by_split[left]}
            right_paths = {(row["dataset"], row["path"].replace("\\", "/").casefold())
                           for row in by_split[right]}
            if left_paths & right_paths:
                raise ValueError("A video path crosses {}/{} in the manifest.".format(left, right))
        expected = {"val": {"real": 134, "fake": 134},
                    "test": {"real": 133, "fake": 133}}
        for split, counts in expected.items():
            observed = {name: sum(row["class_name"] == name for row in by_split[split])
                        for name in counts}
            if observed != counts:
                raise ValueError("Expected fixed {} class counts {}, found {}.".format(
                    split, counts, observed))
    # The balanced 60-group protocol evaluates its entire fixed validation
    # split after every epoch. Older protocols may still cap selection videos.
    if config["train"].get("epoch_protocol") == "balanced_pool_pass":
        val_records = list(full_val_records)
    else:
        val_records = capped_validation_records(
            full_val_records, int(config["data"].get("selection_seed", 42)),
            config["data"].get("selection_max_fakes_per_dataset", 1000))
    if args.smoke_test:
        # Manifests are sorted real-first; truncating the loader would otherwise
        # score one class only and never exercise AUROC/AP/checkpoint selection.
        per_class = max(2, int(args.smoke_test))
        val_records = ([row for row in val_records if int(row["label"]) == 1][:per_class] +
                       [row for row in val_records if int(row["label"]) == 0][:per_class])
    # A pool-pass run will later score the entire merged development set. Catch
    # missing caches and short videos before committing to a long training run.
    preflight_records = (train_records + full_val_records
                         if config["train"].get("epoch_protocol") in ("pool_pass", "balanced_pool_pass")
                         else train_records + val_records)
    preflight_face_cache(preflight_records, config)

    train_set = make_dataset(train_records, config, training=True)
    val_set = make_dataset(val_records, config, training=False,
                           clips_per_video=config["data"].get("selection_clips_per_video", 4))
    physical = int(config["train"].get("physical_batch_size", 8))
    balance_keys = tuple(config["data"].get("train_balance_keys",
                                           ["dataset", "class_name"]))
    group_count = len({tuple(row[key] for key in balance_keys) for row in train_records})
    per_group = int(config["train"].get("clips_per_class_per_epoch", 1000))
    options = _loader_options(config, device)
    if config["train"].get("epoch_protocol") == "balanced_pool_pass":
        if group_count != 2 or physical % 2:
            raise ValueError("balanced_pool_pass needs one dataset, two classes and even batch size.")
        by_class = {name: sum(row["class_name"] == name for row in train_records)
                    for name in ("real", "fake")}
        if by_class["real"] != by_class["fake"] or by_class["real"] % (physical // 2):
            raise ValueError("balanced_pool_pass requires equal class pools divisible by half the batch size.")
        sampler = BalancedBatchSampler(
            train_records, physical, int(config.get("seed", 42)),
            group_keys=balance_keys,
            batches_per_epoch=by_class["real"] // (physical // 2))
        train_loader = DataLoader(train_set, batch_sampler=sampler, **options)
        batch_protocol = "one pass over each class pool per epoch; every batch has 4 real + 4 fake in random order"
    elif config["train"].get("epoch_protocol") == "pool_pass":
        # The epoch is one pass over the training pool, not a per-class clip
        # budget, so clips_per_class_per_epoch does not apply.
        if group_count != 2:
            raise ValueError("pool_pass expects one dataset with real and fake videos.")
        sampler = PoolPassBatchSampler(train_records, physical, int(config.get("seed", 42)))
        train_loader = DataLoader(train_set, batch_sampler=sampler, **options)
        batch_protocol = "one pass over the pool per epoch, random class mix, >=1 real and >=1 fake per batch"
    elif physical >= group_count and physical % group_count == 0:
        clips_per_epoch = group_count * per_group
        if clips_per_epoch % physical:
            raise ValueError("clips_per_epoch must be divisible by physical batch size.")
        sampler = BalancedBatchSampler(
            train_records, physical, int(config.get("seed", 42)),
            group_keys=balance_keys,
            batches_per_epoch=clips_per_epoch // physical)
        train_loader = DataLoader(train_set, batch_sampler=sampler, **options)
        batch_protocol = "every physical batch balanced"
    else:
        # E0/E1 deliberately use batch=1, so balance can only be enforced over
        # the epoch. E2 is the intervention that first balances each batch.
        sampler = GroupBalancedEpochSampler(
            train_records, int(config.get("seed", 42)),
            per_group,
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
    optimizer = torch.optim.Adam(
        parameters, lr=float(config["train"].get("learning_rate", 1e-4)))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=float(config["train"].get("lr_gamma", 0.95)))
    criterion = nn.BCEWithLogitsLoss()
    accumulation = int(config["train"].get("gradient_accumulation_steps", 1))
    use_amp = config["train"].get("precision", "amp") == "amp" and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    clip_setting = config["train"].get("gradient_clip_norm", 5.0)
    clip_norm = None if clip_setting is None else float(clip_setting)

    checkpoint_dir, log_dir = ensure_dir(run_dir / "checkpoints"), ensure_dir(run_dir / "logs")
    metadata = runtime_metadata(config, ROOT)
    clips_per_epoch = (len(sampler) * physical
                       if isinstance(sampler, (BalancedBatchSampler, PoolPassBatchSampler))
                       else len(sampler))
    batches_per_epoch = int(np.ceil(clips_per_epoch / float(physical)))
    metadata.update({
        "optimizer": "Adam", "learning_rate": optimizer.param_groups[0]["lr"],
        "lr_gamma": float(config["train"].get("lr_gamma", 0.95)),
        "manifest": str(args.manifest),
        "validation_paths_sha256": hashlib.sha256("\n".join(sorted(
            row["path"] for row in val_records)).encode()).hexdigest(),
        "test_paths_sha256": hashlib.sha256("\n".join(sorted(
            row["path"] for row in select_records(records, "test"))).encode()).hexdigest(),
        "batch_protocol": batch_protocol,
        "training_pool_videos_by_class": {
            name: sum(row["class_name"] == name for row in train_records)
            for name in ("real", "fake")},
        "selection_validation_videos_by_class": {
            name: sum(row["class_name"] == name for row in val_records)
            for name in ("real", "fake")},
        "development_validation_videos_by_class": {
            name: sum(row["class_name"] == name for row in full_val_records)
            for name in ("real", "fake")},
        "clips_per_epoch": int(clips_per_epoch),
        "optimizer_steps_per_epoch": int(np.ceil(batches_per_epoch / float(accumulation))),
        "experiment": args.experiment,
        "validation_scope": ("full-validation" if config["train"].get("epoch_protocol") == "balanced_pool_pass"
                             else "selection-validation"),
        "selection_validation_videos": len(val_records),
    })
    save_json(run_dir / "config.json", json_safe(config))
    save_json(run_dir / "runtime.json", metadata)
    print("Phase A on {}: {} training / {} validation videos; physical={} accum={} "
          "effective={} precision={}.".format(device, len(train_records), len(val_records),
          physical, accumulation, physical * accumulation, metadata["precision"]))
    print("Sampling protocol: {}.".format(batch_protocol))

    history, best, best_epoch = [], -float("inf"), 0
    early_enabled = bool(config["train"].get("early_stopping_enabled", False))
    early_patience = int(config["train"].get("early_stopping_patience", 10))
    early_min_epochs = int(config["train"].get("early_stopping_min_epochs", 20))
    early_min_delta = float(config["train"].get("early_stopping_min_delta", 0.0))
    if early_enabled and (early_patience < 1 or early_min_epochs < 1 or early_min_delta < 0):
        raise ValueError("Invalid early-stopping settings.")
    best_for_stopping, stale_epochs = -float("inf"), 0
    epochs = 1 if args.smoke_test else int(config["train"].get("epochs", 60))
    for epoch in range(1, epochs + 1):
        started = time.time()
        epoch_lr = optimizer.param_groups[0]["lr"]
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        extractor.train()
        head.train()
        optimizer.zero_grad(set_to_none=True)
        running, seen, optimizer_updates = 0.0, 0, 0
        train_correct_online = torch.zeros((), dtype=torch.long, device=device)
        skipped_training_paths = []
        progress = tqdm(train_loader, desc="Phase A {}/{}".format(epoch, epochs))
        last_step = -1
        # Optional, off by default: how much of the epoch is spent waiting for
        # the loader rather than computing. This is the number that decides
        # whether a smaller input would buy anything -- a run that is starved
        # of data gets nothing from fewer pixels or a cheaper conv1, and FLOPs
        # cannot answer the question. It costs a CUDA sync per step, so it is
        # only for the feasibility run.
        data_wait, compute_time = 0.0, 0.0
        marker = time.time()
        for step, batch in enumerate(progress):
            if args.profile_input_pipeline:
                data_wait += time.time() - marker
                marker = time.time()
            if args.smoke_test is not None and step >= args.smoke_test:
                break
            last_step = step
            if batch is None or batch.get("_skip_only", False):
                skipped_training_paths.extend(
                    [] if batch is None else batch.get("_skipped_paths", []))
                continue
            skipped_training_paths.extend(batch.get("_skipped_paths", []))
            clips = batch["clip"].to(device, non_blocking=True)
            targets = batch["label"].to(device, non_blocking=True).float()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = head(extractor(clips))
                loss = criterion(logits, targets) / accumulation
            train_correct_online += ((logits.detach() > 0) ==
                                     (targets > 0.5)).sum()
            scaler.scale(loss).backward()
            if (step + 1) % accumulation == 0:
                scaler.unscale_(optimizer)
                if clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(parameters, clip_norm)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_updates += int(scaler.get_scale() >= scale_before)
                optimizer.zero_grad(set_to_none=True)
            running += float(loss.detach()) * accumulation * targets.numel()
            seen += targets.numel()
            progress.set_postfix(bce="{:.4f}".format(running / max(1, seen)))
            if args.profile_input_pipeline:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                compute_time += time.time() - marker
                marker = time.time()
        if last_step >= 0 and (last_step + 1) % accumulation:
            scaler.unscale_(optimizer)
            if clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(parameters, clip_norm)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_updates += int(scaler.get_scale() >= scale_before)
            optimizer.zero_grad(set_to_none=True)
        # Split the epoch here. Projecting a 60-epoch cost from a two-epoch run
        # needs the training and validation halves separately: they scale with
        # different things -- clips per epoch against validation videos -- and
        # a throughput computed over their sum belongs to neither.
        train_seconds = time.time() - started
        if config["train"].get("epoch_protocol") in ("pool_pass", "balanced_pool_pass") and skipped_training_paths:
            raise RuntimeError("Incomplete pool-pass epoch: {} training videos "
                               "could not be decoded; first paths: {}."
                               .format(len(skipped_training_paths),
                                       skipped_training_paths[:10]))
        scored = score_deterministic(
            extractor, head, val_loader, device,
            clip_chunk_size=int(config["data"].get("eval_clip_chunk_size", 4)),
            use_amp=use_amp,
            limit=None)
        if config["train"].get("epoch_protocol") in ("pool_pass", "balanced_pool_pass") and (
                len(scored["labels"]) != len(val_records) or
                scored["skipped_unreadable"]):
            raise RuntimeError("Incomplete selection validation: scored {} of {} "
                               "videos; skipped {}."
                               .format(len(scored["labels"]), len(val_records),
                                       scored["skipped_paths"][:10]))
        labels, anomaly = scored["labels"], -scored["logits"]
        threshold = calibrate_threshold(anomaly[labels == 1],
                                        config["train"].get("calibration_fpr", 0.05))
        metrics = detection_metrics(labels, anomaly, threshold)
        if config["train"].get("epoch_protocol") == "balanced_pool_pass":
            metrics["at_fake_probability_0_5"] = probability_half_metrics(labels, anomaly)
        video_logits = torch.as_tensor(scored["logits"], dtype=torch.float32)
        video_labels = torch.as_tensor(labels, dtype=torch.float32)
        validation_loss = float(criterion(video_logits, video_labels))
        metrics["evaluation_scope"] = metadata["validation_scope"]
        metrics["num_videos"] = int(len(labels))
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
               "train_clip_accuracy_online_at_0_5": train_correct_online.item() / max(1, seen),
               "validation_loss": validation_loss,
               "batch_real_count_histogram": (
                   {str(count): sampler.last_real_counts.count(count)
                    for count in sorted(set(sampler.last_real_counts))}
                   if isinstance(sampler, PoolPassBatchSampler) else None),
               "skipped_training_clips": len(skipped_training_paths),
               "skipped_training_paths": skipped_training_paths,
               "skipped_validation_videos": scored["skipped_unreadable"],
               "skipped_validation_paths": scored["skipped_paths"],
               "learning_rate": epoch_lr,
               "selection_metric": "auroc", "selection_value": metrics["auroc"],
               "epoch_duration_seconds": duration,
               "train_seconds": train_seconds,
               "validation_seconds": duration - train_seconds,
               "train_clips_per_second": (seen / train_seconds) if train_seconds else None,
               "input_pipeline": ({"data_wait_seconds": data_wait,
                                   "compute_seconds": compute_time,
                                   "data_wait_fraction": data_wait / max(1e-9, data_wait + compute_time),
                                   "scope": ("training loop only; the fraction is of "
                                             "train_seconds, not of epoch_duration_seconds, "
                                             "which also contains validation"),
                                   "reading": ("data_wait_fraction near 1 means the loader is the "
                                               "bottleneck, where fewer pixels help little and more "
                                               "workers help; near 0 means compute, where the input "
                                               "scale is the lever")}
                                  if args.profile_input_pipeline else None),
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
        if metrics["auroc"] > best_for_stopping + early_min_delta:
            best_for_stopping, stale_epochs = metrics["auroc"], 0
        else:
            stale_epochs += 1
        save_history(history, log_dir)
        if optimizer_updates:
            scheduler.step()
        print("{} AUROC={:.4f}, accuracy={:.4f}, balanced accuracy={:.4f}, "
              "fake AP={:.4f} ({:.2f}x), real AP={:.4f} "
              "({:.2f}x), Macro-AP(real/fake)={:.4f}, embedding-var={:.3g}".format(
              metadata["validation_scope"], metrics["auroc"], metrics["accuracy"], metrics["balanced_accuracy"],
              metrics["fake_average_precision"],
              metrics["fake_ap_lift"], metrics["real_average_precision"],
              metrics["real_ap_lift"], metrics["macro_average_precision_real_fake"],
              metrics["embedding_variance_mean"]))
        if config["train"].get("epoch_protocol") == "balanced_pool_pass":
            half = metrics["at_fake_probability_0_5"]
            print("at fake probability 0.5: validation accuracy={:.4f}, "
                  "balanced accuracy={:.4f}, fake F1={:.4f}; "
                  "online training-clip accuracy={:.4f}".format(
                      half["accuracy"], half["balanced_accuracy"], half["fake_f1"],
                      row["train_clip_accuracy_online_at_0_5"]))
        if early_enabled and epoch >= early_min_epochs and stale_epochs >= early_patience:
            print("Early stopping at epoch {}: validation AUROC has not improved "
                  "for {} epochs; best checkpoint is epoch {}.".format(
                      epoch, stale_epochs, best_epoch))
            break
    print("Best Phase-A epoch {} AUROC {:.4f}: {}".format(
        best_epoch, best, (checkpoint_dir / "best.pt").resolve()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
