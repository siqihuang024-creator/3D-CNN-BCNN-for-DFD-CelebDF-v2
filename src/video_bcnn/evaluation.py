"""Memory-bounded video scoring shared by Phase A, Phase C and evaluation."""

import numpy as np
import torch
from tqdm import tqdm


def _metadata(result, batch, index):
    result["paths"].append(batch["path"][index])
    result["datasets"].append(batch["dataset"][index])
    result["methods"].append(batch["method"][index])
    result["target_ids"].append(batch["target_id"][index])
    result["donor_ids"].append(batch["donor_id"][index])


@torch.no_grad()
def score_deterministic(extractor, head, loader, device, clip_chunk_size=4,
                        use_amp=True, limit=None):
    """Return one real-positive logit per video without loading all clips at once."""
    extractor.eval()
    head.eval()
    labels, logits, paths, sources, datasets, methods, target_ids, donor_ids = (
        [], [], [], [], [], [], [], [])
    embeddings = []
    skipped = 0
    for step, batch in enumerate(tqdm(loader, desc="Phase A scoring", leave=False)):
        if limit is not None and step >= limit:
            break
        if batch is None:
            skipped += 1
            continue
        if batch["clips"].shape[0] != 1:
            raise ValueError("Evaluation DataLoader must use batch_size=1.")
        clips = batch["clips"][0]
        current = []
        for start in range(0, clips.shape[0], int(clip_chunk_size)):
            part = clips[start:start + int(clip_chunk_size)].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=bool(use_amp and device.type == "cuda")):
                features = extractor(part)
                current.append(head(features).float().cpu())
                embeddings.append(features.float().cpu())
        logits.append(float(torch.cat(current).mean()))
        labels.append(int(batch["label"][0]))
        paths.append(batch["path"][0])
        datasets.append(batch["dataset"][0])
        methods.append(batch["method"][0])
        target_ids.append(batch["target_id"][0])
        donor_ids.append(batch["donor_id"][0])
        source = batch.get("source_clip", [batch["target_id"][0]])
        sources.append(source[0])
    matrix = torch.cat(embeddings) if embeddings else torch.empty(0, 0)
    return {"labels": np.asarray(labels), "logits": np.asarray(logits),
            "paths": paths, "datasets": np.asarray(datasets),
            "methods": np.asarray(methods), "target_ids": target_ids,
            "donor_ids": donor_ids, "source_clips": np.asarray(sources),
            "skipped_unreadable": skipped,
            "embedding_variance_mean": (float(matrix.var(0, unbiased=False).mean())
                                        if len(matrix) else float("nan"))}


@torch.no_grad()
def extract_video_features(extractor, clips, device, clip_chunk_size=4, use_amp=False):
    """Extract clip features in bounded chunks and average to one video vector."""
    outputs = []
    for start in range(0, clips.shape[0], int(clip_chunk_size)):
        part = clips[start:start + int(clip_chunk_size)].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=bool(use_amp and device.type == "cuda")):
            outputs.append(extractor(part).float())
    return torch.cat(outputs, 0).mean(0, keepdim=True)


@torch.no_grad()
def score_bayesian(model, loader, device, mc_samples=30, clip_chunk_size=4,
                   collect_embeddings=False):
    """Score videos by negative posterior predictive mean (high means fake)."""
    model.feature_extractor.eval()
    result = {key: [] for key in (
        "labels", "scores", "means", "stds", "paths", "datasets", "methods",
        "target_ids", "donor_ids", "source_clips", "embedding_norms")}
    embeddings, skipped = [], 0
    for batch in tqdm(loader, desc="Phase C scoring", leave=False):
        if batch is None:
            skipped += 1
            continue
        if batch["clips"].shape[0] != 1:
            raise ValueError("Evaluation DataLoader must use batch_size=1.")
        feature = extract_video_features(model.feature_extractor, batch["clips"][0],
                                         device, clip_chunk_size, use_amp=False)
        if int(mc_samples) > 0:
            mean, std = model.posterior_from_features(feature, mc_samples)
        else:
            mean = model.posterior_loc_from_features(feature)
            std = torch.zeros_like(mean)
        result["labels"].append(int(batch["label"][0]))
        result["means"].append(float(mean[0].cpu()))
        result["stds"].append(float(std[0].cpu()))
        result["scores"].append(float(-mean[0].cpu()))
        result["embedding_norms"].append(float(feature.norm().cpu()))
        _metadata(result, batch, 0)
        source = batch.get("source_clip", [batch["target_id"][0]])
        result["source_clips"].append(source[0])
        embeddings.append(feature.cpu())
    matrix = torch.cat(embeddings, 0) if embeddings else torch.empty(0, 0)
    result["embedding_variance_mean"] = (
        float(matrix.var(0, unbiased=False).mean()) if len(matrix) else float("nan"))
    result["embedding_norm_mean"] = (
        float(np.mean(result["embedding_norms"])) if result["embedding_norms"] else float("nan"))
    result["skipped_unreadable"] = skipped
    for key in ("labels", "scores", "means", "stds", "datasets", "methods",
                "embedding_norms", "source_clips"):
        result[key] = np.asarray(result[key])
    if collect_embeddings:
        result["embeddings"] = matrix.numpy()
    return result
