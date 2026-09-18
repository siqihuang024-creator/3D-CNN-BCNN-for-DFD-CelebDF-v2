"""Dataset selection, balanced sampling, clip scoring, and grouped evaluation."""

import hashlib
import math

import numpy as np
import torch
from torch.utils.data import Sampler
from tqdm import tqdm

from .data import VideoClipDataset
from .metrics import calibrate_threshold, detection_metrics


def active_records(records, config):
    active = set(config["data"].get("active_datasets", []))
    return [row for row in records if not active or row["dataset"] in active]


def select_records(records, split, label=None):
    selected = [row for row in records if row["split"] == split]
    if label is not None:
        selected = [row for row in selected if int(row["label"]) == int(label)]
    return selected


# The detector's positive class is always "fake", so the anomaly score has to
# rank fakes high whichever class the likelihood was anchored to.
#   train_label      manifest label kept for backpropagation (None keeps both)
#   constant_target  fixed regression target, or None to use the real label
#   score_sign       multiplies the posterior mean to build the anomaly score
TRAINING_OBJECTIVES = {
    "one_class_real": {"train_label": 1, "constant_target": 1.0, "score_sign": -1.0},
    "one_class_fake": {"train_label": 0, "constant_target": 1.0, "score_sign": 1.0},
    "supervised": {"train_label": None, "constant_target": None, "score_sign": -1.0},
}


def resolve_objective(config):
    """Read `train.objective`, defaulting to the original real-only protocol."""
    name = config["train"].get("objective", "one_class_real")
    if name not in TRAINING_OBJECTIVES:
        raise ValueError(
            "Unknown train.objective {!r}; expected one of {}.".format(
                name, sorted(TRAINING_OBJECTIVES)
            )
        )
    return dict(TRAINING_OBJECTIVES[name], name=name)


def training_records(records, objective):
    """Select the training videos an objective backpropagates through."""
    return select_records(records, "train", label=objective["train_label"])


def make_dataset(records, config, training, clips_per_video=None):
    data_config = dict(config["data"])
    data_config.update({
        "input_resize": config["model"]["input_resize"],
        "center_crop": config["model"]["center_crop"],
    })
    return VideoClipDataset(
        records,
        config["data"]["dataset_roots"],
        data_config,
        training=training,
        clips_per_video=clips_per_video,
    )


class GroupBalancedEpochSampler(Sampler):
    """Draw an equal clip budget from every balance group, with fresh positions.

    `group_keys` names the manifest columns that define a balance group. The
    one-class protocol balances `["dataset"]`; supervised training balances
    `["dataset", "class_name"]` so the 30k CelebDFv3 fakes cannot drown out the
    613 reals. `stratify_key` round-robins inside a group, which keeps all 23
    CelebDFv3 forgery methods present in every epoch instead of letting the
    larger method families dominate by chance.
    """

    def __init__(self, records, seed, samples_per_group="max",
                 group_keys=("dataset",), stratify_key=None):
        self.group_keys = tuple(group_keys)
        if not self.group_keys:
            raise ValueError("group_keys must name at least one manifest column.")
        self.stratify_key = stratify_key
        self.indices, self.strata = {}, {}
        for index, row in enumerate(records):
            group = tuple(row[key] for key in self.group_keys)
            self.indices.setdefault(group, []).append(index)
            if stratify_key is not None:
                self.strata.setdefault(group, {}).setdefault(row[stratify_key], []).append(index)
        if not self.indices:
            raise ValueError("Cannot sample an empty training record list.")
        if samples_per_group == "max":
            self.samples_per_group = max(len(values) for values in self.indices.values())
        else:
            self.samples_per_group = int(samples_per_group)
        if self.samples_per_group < 1:
            raise ValueError("samples_per_group must be positive or 'max'.")
        self.generator = torch.Generator()
        self.generator.manual_seed(int(seed))

    # Retained so existing call sites and logs keep reading a familiar name.
    @property
    def samples_per_dataset(self):
        return self.samples_per_group

    def __len__(self):
        return self.samples_per_group * len(self.indices)

    def _resample(self, values, quota):
        """Cycle through a shuffled pool until the quota is met."""
        result = []
        while len(result) < quota:
            order = torch.randperm(len(values), generator=self.generator).tolist()
            remaining = quota - len(result)
            result.extend(values[position] for position in order[:remaining])
        return result

    def _group_epoch_indices(self, group):
        if self.stratify_key is None:
            return self._resample(self.indices[group], self.samples_per_group)
        strata = self.strata[group]
        names = sorted(strata)
        base, extra = divmod(self.samples_per_group, len(names))
        result = []
        for position, name in enumerate(names):
            quota = base + (1 if position < extra else 0)
            if quota:
                result.extend(self._resample(strata[name], quota))
        return result

    def __iter__(self):
        combined = []
        for group in sorted(self.indices):
            combined.extend(self._group_epoch_indices(group))
        order = torch.randperm(len(combined), generator=self.generator).tolist()
        return iter(combined[position] for position in order)


class DatasetBalancedEpochSampler(GroupBalancedEpochSampler):
    """Backwards-compatible dataset-only balancing used by the one-class runs."""

    def __init__(self, records, seed, samples_per_dataset="max"):
        super(DatasetBalancedEpochSampler, self).__init__(
            records, seed, samples_per_dataset, group_keys=("dataset",)
        )


class BalancedBatchSampler(Sampler):
    """Yield physical batches balanced over dataset/class groups."""

    def __init__(self, records, batch_size, seed=42,
                 group_keys=("dataset", "class_name"), batches_per_epoch=None):
        self.batch_size, self.seed, self.epoch = int(batch_size), int(seed), 0
        groups = {}
        for index, row in enumerate(records):
            groups.setdefault(tuple(row[key] for key in group_keys), []).append(index)
        if not groups:
            raise ValueError("Cannot balance an empty record list.")
        if self.batch_size % len(groups):
            raise ValueError("physical batch {} must be divisible by {} groups."
                             .format(self.batch_size, len(groups)))
        self.groups = {key: np.asarray(value, dtype=np.int64)
                       for key, value in sorted(groups.items())}
        self.per_group = self.batch_size // len(groups)
        default = int(np.ceil(max(len(v) for v in self.groups.values()) /
                              float(self.per_group)))
        self.batches_per_epoch = int(default if batches_per_epoch is None
                                     else batches_per_epoch)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        streams = {}
        for key, values in self.groups.items():
            needed = self.batches_per_epoch * self.per_group
            streams[key] = rng.choice(values, size=needed,
                                      replace=needed > len(values))
        for batch_index in range(self.batches_per_epoch):
            start, batch = batch_index * self.per_group, []
            for key in sorted(streams):
                batch.extend(int(value) for value in
                             streams[key][start:start + self.per_group])
            rng.shuffle(batch)
            yield batch


def dataset_balanced_sampler(records, seed, samples_per_dataset="max",
                            group_keys=("dataset",), stratify_key=None):
    return GroupBalancedEpochSampler(
        records, seed, samples_per_dataset, group_keys, stratify_key
    )


def _stable_rng(seed, *parts):
    key = ":".join([str(seed)] + [str(part) for part in parts]).encode("utf-8")
    value = int(hashlib.sha256(key).hexdigest()[:8], 16)
    return np.random.RandomState(value)


def _update_feature_stat(stats, key, vector):
    if key not in stats:
        stats[key] = {
            "count": 0,
            "sum": torch.zeros_like(vector, dtype=torch.float64),
            "square_sum": torch.zeros_like(vector, dtype=torch.float64),
            "norm_sum": 0.0,
        }
    item = stats[key]
    item["count"] += 1
    item["sum"] += vector
    item["square_sum"] += vector.square()
    item["norm_sum"] += float(vector.norm().item())


def _feature_diagnostics(stats):
    groups, centroids = {}, {}
    for key, item in stats.items():
        count = float(item["count"])
        centroid = item["sum"] / count
        variance = item["square_sum"] / count - centroid.square()
        centroids[key] = centroid
        groups[key] = {
            "count": int(item["count"]),
            "mean_norm": float(item["norm_sum"] / count),
            "within_group_variance_mean": float(variance.clamp_min(0.0).mean().item()),
        }
    separations = {}
    prefixes = [""] + sorted({key.rsplit("/", 1)[0] for key in stats if "/" in key})
    for prefix in prefixes:
        real_key = "{}/real".format(prefix) if prefix else "real"
        fake_key = "{}/fake".format(prefix) if prefix else "fake"
        if real_key not in centroids or fake_key not in centroids:
            continue
        real, fake = centroids[real_key], centroids[fake_key]
        denominator = float(real.norm().item() * fake.norm().item())
        cosine_distance = 0.0 if denominator == 0.0 else 1.0 - float(torch.dot(real, fake).item()) / denominator
        separations[prefix or "overall"] = {
            "centroid_l2_distance": float((real - fake).norm().item()),
            "centroid_cosine_distance": float(cosine_distance),
        }
    return {"groups": groups, "real_fake_separation": separations}


def overfit_records(records, per_class, seed):
    """A small, deterministic, class-balanced slice for the fit diagnostic.

    A model that cannot drive the loss down on a handful of videos it trains
    on directly has an architecture or optimisation problem, not a
    generalisation one -- and that question is worth minutes, not hours.
    """
    groups = {}
    for row in records:
        groups.setdefault((row["dataset"], row["class_name"]), []).append(row)
    selected = []
    for key in sorted(groups):
        values = sorted(groups[key], key=lambda row: row["path"])
        order = _stable_rng(seed, "overfit", *key).permutation(len(values))
        selected.extend(values[int(index)] for index in order[: int(per_class)])
    return selected


def filter_forgery_methods(records, patterns):
    """Keep every real video and only the fakes whose method matches a pattern.

    CelebDFv3's fakes carry the generator's own output size: all eight FaceSwap
    methods write native wide frames like the real videos, while six of seven
    FaceReenact methods and all seven TalkingFace methods write 256x256 or
    512x512 squares. Every real video is native wide, so "square and small"
    identifies 61% of the fakes with no reference to content at all, and a
    detector can reach most of its score by measuring blur.

    Restricting training to the native-resolution generators removes the
    shortcut from the gradient, at the cost of a model that has never seen
    reenactment or talking-face forgeries.
    """
    wanted = tuple(patterns)
    kept, dropped = [], {}
    for row in records:
        if int(row["label"]) == 1:
            kept.append(row)
            continue
        method = row.get("method", "")
        if any(pattern in method for pattern in wanted):
            kept.append(row)
        else:
            dropped[method] = dropped.get(method, 0) + 1
    if not any(int(row["label"]) == 0 for row in kept):
        raise ValueError(
            "No fake videos match {!r}; check the method names in the "
            "manifest.".format(list(wanted))
        )
    return kept, dropped


def subset_training_identities(records, count, seed):
    """Keep `count` training identities, dropping every row that leaves the set.

    DFD reaches only chance with 20 training identities while CelebDFv3 reaches
    0.777 with 251, but those two runs also differ in resolution, face-detection
    failure rate and forgery-method count. Cutting CelebDFv3's identities to the
    same 20 isolates identity diversity from all of them.

    Identities are drawn from those that appear in fakes, because the donor-safe
    rule keeps a fake only when its target and its donor both survive: sampling
    from all 251 identities instead leaves 20 identities with zero fakes.
    """
    per_dataset, kept, counts = {}, [], {}
    for row in records:
        if int(row["label"]) == 0:
            names = per_dataset.setdefault(row["dataset"], set())
            names.add(row["target_id"])
            if row["donor_id"]:
                names.add(row["donor_id"])
    chosen = {}
    for dataset in sorted(per_dataset):
        names = sorted(per_dataset[dataset])
        if int(count) >= len(names):
            chosen[dataset] = set(names)
            continue
        rng = _stable_rng(seed, "identities", dataset, count)
        order = rng.permutation(len(names))
        chosen[dataset] = {names[int(index)] for index in order[: int(count)]}
    for row in records:
        allowed = chosen.get(row["dataset"])
        if allowed is None:
            continue
        if int(row["label"]) == 1:
            keep = row["target_id"] in allowed
        else:
            # Mirror the manifest's donor-safe rule: an empty donor carries no
            # second identity, so only the target has to survive.
            keep = row["target_id"] in allowed and (
                not row["donor_id"] or row["donor_id"] in allowed
            )
        if keep:
            kept.append(row)
            key = (row["dataset"], row["class_name"])
            counts[key] = counts.get(key, 0) + 1
    if not kept:
        raise ValueError(
            "Keeping {} identities left no training videos.".format(count)
        )
    summary = {
        dataset: {"identities": len(names)} for dataset, names in sorted(chosen.items())
    }
    for (dataset, class_name), value in sorted(counts.items()):
        summary[dataset][class_name] = value
    return kept, summary


def capped_validation_records(records, seed, max_fakes_per_dataset):
    """Keep all real videos and a deterministic method-stratified fake subset."""
    if max_fakes_per_dataset is None:
        return list(records)
    selected = [row for row in records if int(row["label"]) == 1]
    for dataset in sorted({row["dataset"] for row in records}):
        groups = {}
        for row in records:
            if row["dataset"] == dataset and int(row["label"]) == 0:
                groups.setdefault(row["method"], []).append(row)
        shuffled = {}
        for method, values in groups.items():
            order = _stable_rng(seed, dataset, method).permutation(len(values))
            shuffled[method] = [values[int(index)] for index in order]
        chosen, position = [], 0
        methods = sorted(shuffled)
        while len(chosen) < int(max_fakes_per_dataset):
            added = False
            for method in methods:
                if position < len(shuffled[method]):
                    chosen.append(shuffled[method][position])
                    added = True
                    if len(chosen) >= int(max_fakes_per_dataset):
                        break
            if not added:
                break
            position += 1
        selected.extend(chosen)
    return selected


@torch.no_grad()
def score_loader(bayesian_model, loader, device, mc_samples=0, collect_embeddings=False,
                 score_sign=-1.0):
    bayesian_model.feature_extractor.eval()
    labels, scores, means, stds = [], [], [], []
    paths, datasets, methods, target_ids, donor_ids = [], [], [], [], []
    embedding_norms, exported_embeddings = [], []
    fps_values, face_any_miss, face_miss_fraction = [], [], []
    center_x_jitter, center_y_jitter, width_jitter, height_jitter = [], [], [], []
    feature_stats = {}
    feature_sum = None
    feature_square_sum = None
    feature_count = 0
    skipped = 0
    for batch in tqdm(loader, desc="Scoring clips", leave=False):
        if batch is None:
            skipped += 1
            continue
        clips = batch["clips"].to(device, non_blocking=True)
        batch_size, clip_count, channels, frame_count, height, width = clips.shape
        if bayesian_model.feature_extractor.input_mode == "clip":
            model_inputs = clips.flatten(0, 1)
            units_per_video = clip_count
        else:
            model_inputs = clips.permute(0, 1, 3, 2, 4, 5).reshape(
                batch_size * clip_count * frame_count, channels, height, width
            )
            units_per_video = clip_count * frame_count
        features = bayesian_model.feature_extractor(model_inputs)
        posterior_loc = bayesian_model.posterior_loc_from_features(features)
        if int(mc_samples) > 0:
            _, posterior_std = bayesian_model.posterior_from_features(features, mc_samples)
        else:
            posterior_std = torch.zeros_like(posterior_loc)
        video_features = features.reshape(batch_size, units_per_video, -1).mean(dim=1)
        cpu_features = video_features.detach().double().cpu()
        batch_labels = batch["label"].cpu().tolist()
        for index in range(batch_size):
            class_name = "real" if int(batch_labels[index]) == 1 else "fake"
            dataset_name = batch["dataset"][index]
            for key in ("all", class_name, dataset_name, "{}/{}".format(dataset_name, class_name)):
                _update_feature_stat(feature_stats, key, cpu_features[index])
        current_sum = cpu_features.sum(dim=0)
        current_square_sum = cpu_features.square().sum(dim=0)
        feature_sum = current_sum if feature_sum is None else feature_sum + current_sum
        feature_square_sum = current_square_sum if feature_square_sum is None else feature_square_sum + current_square_sum
        feature_count += batch_size
        embedding_norms.extend(video_features.norm(dim=1).cpu().tolist())
        if collect_embeddings:
            exported_embeddings.append(video_features.float().cpu().numpy())
        unit_anomaly = float(score_sign) * posterior_loc.reshape(batch_size, units_per_video)
        scores.extend(unit_anomaly.mean(dim=1).cpu().tolist())
        means.extend(posterior_loc.reshape(batch_size, units_per_video).mean(dim=1).cpu().tolist())
        stds.extend(posterior_std.reshape(batch_size, units_per_video).mean(dim=1).cpu().tolist())
        labels.extend(batch_labels)
        paths.extend(batch["path"])
        datasets.extend(batch["dataset"])
        methods.extend(batch["method"])
        target_ids.extend(batch["target_id"])
        donor_ids.extend(batch["donor_id"])
        fps_values.extend(batch["fps"].cpu().tolist())
        face_any_miss.extend(batch["clip_face_any_miss"].max(dim=1).values.cpu().tolist())
        face_miss_fraction.extend(batch["clip_face_miss_fraction"].mean(dim=1).cpu().tolist())
        center_x_jitter.extend(batch["clip_box_center_x_jitter"].mean(dim=1).cpu().tolist())
        center_y_jitter.extend(batch["clip_box_center_y_jitter"].mean(dim=1).cpu().tolist())
        width_jitter.extend(batch["clip_box_width_jitter"].mean(dim=1).cpu().tolist())
        height_jitter.extend(batch["clip_box_height_jitter"].mean(dim=1).cpu().tolist())
    if skipped:
        print("Scored {} videos; skipped {} unreadable.".format(len(labels), skipped))
    feature_mean = feature_sum / float(max(1, feature_count))
    feature_variance = feature_square_sum / float(max(1, feature_count)) - feature_mean.square()
    result = {
        "skipped_unreadable": int(skipped),
        "labels": np.asarray(labels),
        "scores": np.asarray(scores),
        "means": np.asarray(means),
        "stds": np.asarray(stds),
        "paths": paths,
        "datasets": np.asarray(datasets),
        "methods": np.asarray(methods),
        "target_ids": target_ids,
        "donor_ids": donor_ids,
        "embedding_norms": np.asarray(embedding_norms),
        "embedding_variance_mean": float(feature_variance.clamp_min(0.0).mean().item()),
        "embedding_norm_mean": float(np.mean(embedding_norms)),
        "embedding_diagnostics": _feature_diagnostics(feature_stats),
        "fps": np.asarray(fps_values),
        "face_any_miss": np.asarray(face_any_miss),
        "face_miss_fraction": np.asarray(face_miss_fraction),
        "center_x_jitter": np.asarray(center_x_jitter),
        "center_y_jitter": np.asarray(center_y_jitter),
        "width_jitter": np.asarray(width_jitter),
        "height_jitter": np.asarray(height_jitter),
    }
    if collect_embeddings:
        result["embeddings"] = np.concatenate(exported_embeddings, axis=0)
    return result


def per_dataset_metrics(values, global_threshold, false_positive_rate):
    result = {}
    for dataset in sorted(set(values["datasets"].tolist())):
        mask = values["datasets"] == dataset
        labels, scores = values["labels"][mask], values["scores"][mask]
        metrics = detection_metrics(labels, scores, global_threshold)
        real_scores = scores[labels == 1]
        local_threshold = calibrate_threshold(real_scores, false_positive_rate)
        local = detection_metrics(labels, scores, local_threshold)
        metrics.update({
            "num_videos": int(mask.sum()),
            "num_real_videos": int((labels == 1).sum()),
            "num_fake_videos": int((labels == 0).sum()),
            "mean_real_anomaly": float(scores[labels == 1].mean()),
            "mean_fake_anomaly": float(scores[labels == 0].mean()),
            "domain_calibrated_threshold": float(local_threshold),
            "domain_calibrated_accuracy": local["accuracy"],
            "domain_calibrated_balanced_accuracy": local["balanced_accuracy"],
        })
        result[dataset] = metrics
    return result


def per_method_metrics(values, threshold):
    result = {}
    for dataset in sorted(set(values["datasets"].tolist())):
        dataset_mask = values["datasets"] == dataset
        real_mask = dataset_mask & (values["labels"] == 1)
        fake_methods = sorted(set(values["methods"][dataset_mask & (values["labels"] == 0)].tolist()))
        for method in fake_methods:
            fake_mask = dataset_mask & (values["methods"] == method) & (values["labels"] == 0)
            mask = real_mask | fake_mask
            metrics = detection_metrics(values["labels"][mask], values["scores"][mask], threshold)
            metrics["num_real_videos"] = int(real_mask.sum())
            metrics["num_fake_videos"] = int(fake_mask.sum())
            result["{}/{}".format(dataset, method)] = metrics
    return result


def preprocessing_diagnostics(values):
    result = {}
    groups = [("overall", np.ones(len(values["labels"]), dtype=bool))]
    for dataset in sorted(set(values["datasets"].tolist())):
        dataset_mask = values["datasets"] == dataset
        groups.append((dataset, dataset_mask))
        groups.append(("{}/real".format(dataset), dataset_mask & (values["labels"] == 1)))
        groups.append(("{}/fake".format(dataset), dataset_mask & (values["labels"] == 0)))
    for name, mask in groups:
        if not mask.any():
            continue
        valid_fps = values["fps"][mask]
        valid_fps = valid_fps[valid_fps > 0.0]
        result[name] = {
            "num_videos": int(mask.sum()),
            "fps_mean": float(valid_fps.mean()) if len(valid_fps) else 0.0,
            "fps_std": float(valid_fps.std()) if len(valid_fps) else 0.0,
            "clip_any_face_detection_failure_rate": float(values["face_any_miss"][mask].mean()),
            "frame_face_detection_failure_fraction": float(values["face_miss_fraction"][mask].mean()),
            "box_center_x_jitter_mean": float(values["center_x_jitter"][mask].mean()),
            "box_center_y_jitter_mean": float(values["center_y_jitter"][mask].mean()),
            "box_width_jitter_mean": float(values["width_jitter"][mask].mean()),
            "box_height_jitter_mean": float(values["height_jitter"][mask].mean()),
        }
    return result


def evaluate_values(values, false_positive_rate, threshold=None, include_methods=False):
    if threshold is None:
        threshold = calibrate_threshold(
            values["scores"][values["labels"] == 1], false_positive_rate
        )
    metrics = detection_metrics(values["labels"], values["scores"], threshold)
    metrics["mean_real_anomaly"] = float(values["scores"][values["labels"] == 1].mean())
    metrics["mean_fake_anomaly"] = float(values["scores"][values["labels"] == 0].mean())
    metrics["mean_real_posterior"] = float(values["means"][values["labels"] == 1].mean())
    metrics["mean_fake_posterior"] = float(values["means"][values["labels"] == 0].mean())
    metrics["embedding_variance_mean"] = float(values["embedding_variance_mean"])
    metrics["embedding_norm_mean"] = float(values["embedding_norm_mean"])
    metrics["embedding_diagnostics"] = values["embedding_diagnostics"]
    metrics["preprocessing_diagnostics"] = preprocessing_diagnostics(values)
    metrics["per_dataset"] = per_dataset_metrics(values, threshold, false_positive_rate)
    auc_values = [item["auroc"] for item in metrics["per_dataset"].values() if not math.isnan(item["auroc"])]
    metrics["macro_dataset_auroc"] = float(np.mean(auc_values))
    if include_methods:
        metrics["per_method"] = per_method_metrics(values, threshold)
    return metrics, float(threshold)
