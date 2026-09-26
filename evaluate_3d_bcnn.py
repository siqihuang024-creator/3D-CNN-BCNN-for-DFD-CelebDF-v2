"""Evaluate a V2 Phase-A or Phase-C checkpoint at video level."""

import argparse
import copy
import hashlib
import sys
from pathlib import Path

import numpy as np
import pyro
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from video_bcnn.data import (load_manifest, preflight_face_cache, seed_worker,
                             skip_unreadable_collate)
from video_bcnn.evaluation import score_bayesian, score_deterministic
from video_bcnn.experiment import active_records, make_dataset, select_records
from video_bcnn.metrics import (calibrate_threshold, clustered_auroc_interval,
                                detection_metrics)
from video_bcnn.model import DeterministicHead, build_feature_extractor, freeze_extractor
from video_bcnn.reporting import save_evaluation_report
from video_bcnn.utils import (load_checkpoint, load_config, override_dataset_roots,
                              override_num_workers, resolve_device, seed_everything)
from train_3d_bcnn import build_model


class ShuffledFrameLoader:
    """Yield the same clips with their frames permuted in time.

    This perturbs the whole model, including the trunk's temporal convolutions;
    it cannot isolate the TCN. Small drops do not prove order independence.
    Orders are bound to video/clip IDs, independent of loader traversal.
    """

    def __init__(self, loader, seed):
        self.loader, self.seed = loader, int(seed)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            if batch is None or batch.get("_skip_only", False):
                yield batch
                continue
            batch = dict(batch)
            clips = batch["clips"].clone()
            frame_indices = batch.get("clip_frame_indices")
            if frame_indices is not None:
                frame_indices = frame_indices.clone()
            steps = clips.shape[3]
            for video in range(clips.shape[0]):
                for clip in range(clips.shape[1]):
                    paths = batch.get("relative_path", batch.get("path", [str(video)]))
                    datasets = batch.get("dataset", [""] * clips.shape[0])
                    key = "{}|{}::{}|{}".format(self.seed, datasets[video],
                        str(paths[video]).replace("\\", "/"), clip).encode("utf-8")
                    value = int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2 ** 63)
                    generator = torch.Generator().manual_seed(value)
                    order = torch.randperm(steps, generator=generator)
                    clips[video, clip] = clips[video, clip][:, order]
                    if frame_indices is not None:
                        frame_indices[video, clip] = frame_indices[video, clip][order]
            batch["clips"] = clips
            if frame_indices is not None:
                batch["clip_frame_indices"] = frame_indices
            yield batch


def evaluate(values, threshold, draws, seed):
    labels, scores = values["labels"], values["scores"]
    metrics = detection_metrics(labels, scores, threshold)
    fake_labels = 1 - labels
    def interval(clusters):
        report = clustered_auroc_interval(
            fake_labels, scores, clusters, draws=draws, seed=seed)
        groups = {}
        for label, cluster in zip(labels, clusters):
            groups.setdefault(str(cluster), set()).add(int(label))
        report["clusters_with_real"] = sum(1 in classes for classes in groups.values())
        report["clusters_with_fake"] = sum(0 in classes for classes in groups.values())
        return report
    metrics["identity_bootstrap"] = interval(values["target_ids"])
    metrics["source_family_bootstrap"] = interval(values["source_clips"])
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
    parser.add_argument("--expected-videos", type=int, default=None,
                        help="Require this many manifest videos for the requested split.")
    parser.add_argument("--recalibrate-threshold", action="store_true")
    parser.add_argument("--export-embeddings", action="store_true")
    parser.add_argument("--report-name", default=None,
                        help="Output prefix. Defaults to full_val for a complete val run, "
                             "test for test, and <split>_debug with --max-videos.")
    parser.add_argument("--experiment", default=None,
                        help="Human-readable experiment label stored with every score row.")
    parser.add_argument("--shuffle-sequence", action="store_true",
                        help="Order control for the temporal head only: permute the "
                             "per-frame feature sequence between the trunk and the TCN, "
                             "leaving each frame feature untouched. TCN checkpoints only.")
    parser.add_argument("--shuffle-frames", action="store_true",
                        help="Whole-model order diagnostic: permute raw frames per clip. "
                             "Writes a separate report; not a TCN-only intervention.")
    args = parser.parse_args()
    if args.shuffle_frames and args.shuffle_sequence:
        parser.error("Use only one shuffle intervention per evaluation.")
    if (args.shuffle_frames or args.shuffle_sequence) and args.report_name in ("full_val", "test", "val"):
        parser.error("A shuffle control cannot overwrite the ordered report name.")

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
    if args.expected_videos is not None and len(records) != args.expected_videos:
        raise ValueError("Expected {} {} videos in manifest, found {}.".format(
            args.expected_videos, args.split, len(records)))
    preflight_face_cache(records, config)
    dataset = make_dataset(records, config, training=False,
                           clips_per_video=config["data"].get("eval_clips_per_video", 8))
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=int(config["data"].get("num_workers", 0)),
                        pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
                        collate_fn=skip_unreadable_collate)
    if args.shuffle_frames:
        loader = ShuffledFrameLoader(loader, seed)

    stage = checkpoint.get("stage", "phase_c")
    if args.shuffle_sequence and stage != "phase_a":
        raise ValueError("Sequence shuffling is only supported for Phase-A TCN checkpoints.")
    if stage == "phase_a":
        extractor = build_feature_extractor(config["model"]).to(device)
        extractor.load_state_dict(checkpoint["extractor"])
        head = DeterministicHead(extractor.feature_dim,
                                 config["model"].get("deterministic_hidden_dim", 0),
                                 config["model"].get("dropout", 0.2)).to(device)
        head.load_state_dict(checkpoint["head"])
        if args.shuffle_sequence:
            extractor.set_sequence_shuffle(seed)
        raw = score_deterministic(
            extractor, head, loader, device,
            int(config["data"].get("eval_clip_chunk_size", 4)), use_amp=True)
        values = dict(raw)
        values.update({"scores": -raw["logits"], "means": raw["logits"],
                       "stds": np.zeros_like(raw["logits"]),
                       "embedding_norms": np.full(len(raw["labels"]), np.nan),
                       "clip_scores": [-np.asarray(item) for item in raw["clip_logits"]],
                       "clip_means": [np.asarray(item) for item in raw["clip_logits"]],
                       "clip_stds": [np.zeros_like(item) for item in raw["clip_logits"]]})
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

    if args.max_videos is None and len(values["labels"]) != len(records):
        raise RuntimeError(
            "Incomplete {} evaluation: requested {} videos, scored {}; "
            "skipped paths: {}".format(args.split, len(records), len(values["labels"]),
                                      values.get("skipped_paths", [])[:10]))

    if args.expected_videos is not None and len(values["labels"]) != args.expected_videos:
        raise RuntimeError("Scored {} of {} videos; refusing a partial evaluation.".format(
            len(values["labels"]), args.expected_videos))
    threshold = checkpoint.get("threshold")
    if args.recalibrate_threshold or threshold is None:
        threshold = calibrate_threshold(values["scores"][values["labels"] == 1],
                                        config["train"].get("calibration_fpr", 0.05))
    metrics = evaluate(values, float(threshold), args.bootstrap_draws, seed)
    if args.report_name:
        report_name = args.report_name
    elif args.max_videos is not None:
        report_name = "{}_debug".format(args.split)
    elif args.split == "val":
        report_name = "full_val"
    else:
        report_name = args.split
    if not args.report_name:
        # Never overwrite the ordered report: the pair is the measurement.
        if args.shuffle_frames:
            report_name = "{}_shuffled".format(report_name)
        if args.shuffle_sequence:
            report_name = "{}_seqshuffled".format(report_name)
    experiment = args.experiment or Path(args.checkpoint).resolve().parent.parent.name
    metrics.update({"split": args.split, "stage": stage,
                    "checkpoint": str(Path(args.checkpoint).resolve()),
                    "experiment": experiment, "seed": seed,
                    "evaluation_name": report_name,
                    "evaluation_scope": ("full-validation" if args.split == "val" and
                                         args.max_videos is None else
                                         "debug-subset" if args.max_videos is not None else
                                         "final-test"),
                    "frame_order": ("frames-shuffled" if args.shuffle_frames else
                                    "sequence-shuffled" if args.shuffle_sequence else
                                    "natural"),
                    "threshold_source": ("evaluation_reals" if args.recalibrate_threshold
                                         else "checkpoint"),
                    "eval_batch_size": 1,
                    "eval_clip_chunk_size": int(config["data"].get("eval_clip_chunk_size", 4)),
                    "num_videos_requested": len(records),
                    "num_videos_scored": len(values["labels"]),
                    "skipped_videos": values.get("skipped_paths", [])})
    checkpoint_dir = Path(args.checkpoint).resolve().parent
    report_dir = checkpoint_dir.parent / "reports"
    report = save_evaluation_report(values, metrics, report_dir, args.split,
                                    report_name=report_name,
                                    write_legacy=not (args.shuffle_frames or args.shuffle_sequence))
    if args.export_embeddings and "embeddings" in values:
        np.savez_compressed(report_dir / "{}_embeddings.npz".format(args.split),
                            embeddings=values["embeddings"], labels=values["labels"],
                            paths=np.asarray(values["paths"]))
    print("{} AUROC={:.4f}, identity-clustered 95% CI [{:.4f}, {:.4f}]".format(
        args.split, metrics["auroc"], metrics["identity_bootstrap"]["low"],
        metrics["identity_bootstrap"]["high"]))
    print("AP-fake={:.4f}, AP-real={:.4f}, Macro-AP (real/fake)={:.4f}".format(
        metrics["fake_average_precision"], metrics["real_average_precision"],
        metrics["macro_average_precision_real_fake"]))
    print("Report: {}".format(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
