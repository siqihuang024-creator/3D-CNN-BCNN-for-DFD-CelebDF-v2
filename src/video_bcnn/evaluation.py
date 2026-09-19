"""Memory-bounded video scoring shared by Phase A, Phase C and evaluation."""

import numpy as np
import torch
from tqdm import tqdm


def _batch_skips(batch):
    if batch is None:
        return ["<unknown unreadable video>"], True
    paths = list(batch.get("_skipped_paths", []))
    return paths, bool(batch.get("_skip_only", False))


def _metadata(result, batch, index):
    result["paths"].append(batch["path"][index])
    relative = batch.get("relative_path", batch["path"])
    result["relative_paths"].append(relative[index])
    result["datasets"].append(batch["dataset"][index])
    result["methods"].append(batch["method"][index])
    result["target_ids"].append(batch["target_id"][index])
    result["donor_ids"].append(batch["donor_id"][index])
    source = batch.get("source_clip", [batch["target_id"][index]])
    result["source_clips"].append(source[index])


@torch.no_grad()
def extract_clip_features(extractor, clips, device, clip_chunk_size=4, use_amp=False):
    """Extract one feature row per clip in bounded GPU chunks."""
    outputs = []
    for start in range(0, clips.shape[0], int(clip_chunk_size)):
        part = clips[start:start + int(clip_chunk_size)].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=bool(use_amp and device.type == "cuda")):
            outputs.append(extractor(part).float())
    return torch.cat(outputs, 0)


@torch.no_grad()
def extract_video_features(extractor, clips, device, clip_chunk_size=4, use_amp=False):
    """Compatibility helper: mean of independently extracted clip features."""
    return extract_clip_features(extractor, clips, device, clip_chunk_size,
                                 use_amp).mean(0, keepdim=True)


@torch.no_grad()
def score_deterministic(extractor, head, loader, device, clip_chunk_size=4,
                        use_amp=True, limit=None):
    """Score every clip with the Phase-A head, then average scores by video."""
    extractor.eval()
    head.eval()
    result = {key: [] for key in ("labels", "logits", "paths", "relative_paths", "datasets",
                                   "methods", "target_ids", "donor_ids",
                                   "source_clips")}
    feature_sum = feature_square_sum = None
    feature_count, skipped_paths = 0, []
    for step, batch in enumerate(tqdm(loader, desc="Phase A scoring", leave=False)):
        if limit is not None and step >= limit:
            break
        skips, empty = _batch_skips(batch)
        skipped_paths.extend(skips)
        if empty:
            continue
        if batch["clips"].shape[0] != 1:
            raise ValueError("Evaluation DataLoader must use batch_size=1.")
        features = extract_clip_features(extractor, batch["clips"][0], device,
                                         clip_chunk_size, use_amp)
        with torch.cuda.amp.autocast(enabled=bool(use_amp and device.type == "cuda")):
            logits = head(features).float()
        result["logits"].append(float(logits.mean().cpu()))
        result["labels"].append(int(batch["label"][0]))
        _metadata(result, batch, 0)
        video_feature = features.mean(0).double().cpu()
        feature_sum = (video_feature.clone() if feature_sum is None
                       else feature_sum + video_feature)
        square = video_feature.square()
        feature_square_sum = (square if feature_square_sum is None
                              else feature_square_sum + square)
        feature_count += 1
    if feature_count:
        mean = feature_sum / float(feature_count)
        variance = feature_square_sum / float(feature_count) - mean.square()
        variance_mean = float(variance.clamp_min(0).mean())
    else:
        variance_mean = float("nan")
    for key in ("labels", "logits", "datasets", "methods", "source_clips"):
        result[key] = np.asarray(result[key])
    result.update({"skipped_unreadable": len(skipped_paths),
                   "skipped_paths": skipped_paths,
                   "embedding_variance_mean": variance_mean})
    return result


@torch.no_grad()
def cache_bayesian_loader_features(model, loader, device, clip_chunk_size=4):
    """Decode a deterministic validation split once and retain per-clip features."""
    model.feature_extractor.eval()
    result = {key: [] for key in ("labels", "paths", "relative_paths", "datasets", "methods",
                                   "target_ids", "donor_ids", "source_clips")}
    matrices, offsets, skipped_paths, position = [], [], [], 0
    for batch in tqdm(loader, desc="Caching validation features", leave=False):
        skips, empty = _batch_skips(batch)
        skipped_paths.extend(skips)
        if empty:
            continue
        if batch["clips"].shape[0] != 1:
            raise ValueError("Evaluation DataLoader must use batch_size=1.")
        features = extract_clip_features(model.feature_extractor, batch["clips"][0],
                                         device, clip_chunk_size, use_amp=False).cpu()
        matrices.append(features)
        offsets.append((position, position + len(features)))
        position += len(features)
        result["labels"].append(int(batch["label"][0]))
        _metadata(result, batch, 0)
    result["features"] = (torch.cat(matrices, 0) if matrices
                          else torch.empty(0, model.feature_extractor.feature_dim))
    result["offsets"] = offsets
    result["skipped_paths"] = skipped_paths
    return result


@torch.no_grad()
def score_bayesian_cached(model, cached, device, mc_samples=30,
                          collect_embeddings=False):
    """Apply the nonlinear Bayesian head per clip, then average clip predictions."""
    result = {key: list(cached[key]) for key in (
        "labels", "paths", "relative_paths", "datasets", "methods", "target_ids", "donor_ids",
        "source_clips")}
    result.update({"scores": [], "means": [], "stds": [], "embedding_norms": []})
    video_embeddings = []
    for start, end in cached["offsets"]:
        features = cached["features"][start:end].to(device, non_blocking=True)
        if int(mc_samples) > 0:
            means, stds = model.posterior_from_features(features, mc_samples)
        else:
            means = model.posterior_loc_from_features(features)
            stds = torch.zeros_like(means)
        # The head is nonlinear: E[f(h)] is not f(E[h]). Always aggregate scores.
        result["means"].append(float(means.mean().cpu()))
        result["stds"].append(float(stds.mean().cpu()))
        result["scores"].append(float(-means.mean().cpu()))
        video_feature = features.mean(0).cpu()
        video_embeddings.append(video_feature)
        result["embedding_norms"].append(float(video_feature.norm()))
    matrix = (torch.stack(video_embeddings) if video_embeddings
              else torch.empty(0, model.feature_extractor.feature_dim))
    result["embedding_variance_mean"] = (
        float(matrix.var(0, unbiased=False).mean()) if len(matrix) else float("nan"))
    result["embedding_norm_mean"] = (
        float(np.mean(result["embedding_norms"])) if result["embedding_norms"]
        else float("nan"))
    result["skipped_paths"] = list(cached.get("skipped_paths", []))
    result["skipped_unreadable"] = len(result["skipped_paths"])
    for key in ("labels", "scores", "means", "stds", "datasets", "methods",
                "embedding_norms", "source_clips"):
        result[key] = np.asarray(result[key])
    if collect_embeddings:
        result["embeddings"] = matrix.numpy()
    return result


@torch.no_grad()
def score_bayesian(model, loader, device, mc_samples=30, clip_chunk_size=4,
                   collect_embeddings=False):
    cached = cache_bayesian_loader_features(model, loader, device, clip_chunk_size)
    return score_bayesian_cached(model, cached, device, mc_samples,
                                 collect_embeddings)
