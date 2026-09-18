"""Threshold calibration and real-vs-fake detection metrics."""

import math

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def clustered_auroc_interval(labels, scores, clusters, draws=2000, seed=42,
                             confidence=0.95):
    """Bootstrap an AUROC interval by resampling clusters, not videos.

    Resampling rows treats every video as an independent observation. They are
    not: on CelebDFv3 one real clip yields a median of 84 forgeries that share
    its background, wardrobe, lighting and camera motion, so a test split of
    5433 fakes carries about 125 independent units. Resampling rows understates
    the spread several-fold. ``clusters`` is the unit that is actually
    independent -- the source clip, or the identity.

    Draws that end up with one class are skipped rather than scored, which
    happens easily when the cluster count is small.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    clusters = np.asarray(clusters)
    groups = {}
    for index, name in enumerate(clusters):
        groups.setdefault(name, []).append(index)
    keys = sorted(groups)
    members = [np.asarray(groups[key]) for key in keys]
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(int(draws)):
        picked = rng.integers(0, len(members), size=len(members))
        rows = np.concatenate([members[index] for index in picked])
        if len(np.unique(labels[rows])) < 2:
            continue
        samples.append(roc_auc_score(labels[rows], scores[rows]))
    tail = (1.0 - float(confidence)) / 2.0
    point = (float(roc_auc_score(labels, scores))
             if len(np.unique(labels)) == 2 else math.nan)
    if not samples:
        return {"auroc": point, "low": math.nan, "high": math.nan,
                "standard_error": math.nan, "clusters": len(keys), "draws": 0}
    samples = np.asarray(samples)
    return {
        "auroc": point,
        "low": float(np.quantile(samples, tail)),
        "high": float(np.quantile(samples, 1.0 - tail)),
        "standard_error": float(samples.std(ddof=1)),
        "clusters": len(keys),
        "draws": int(samples.size),
    }


def calibrate_threshold(real_anomaly_scores, false_positive_rate):
    scores = np.asarray(real_anomaly_scores, dtype=np.float64)
    if not len(scores):
        raise ValueError("At least one real validation score is required for calibration.")
    return float(np.quantile(scores, 1.0 - float(false_positive_rate)))


def detection_metrics(real_labels, anomaly_scores, threshold, target_fpr=0.05):
    labels = np.asarray(real_labels, dtype=np.int64)
    scores = np.asarray(anomaly_scores, dtype=np.float64)
    fake_labels = 1 - labels
    predicted_fake = (scores >= float(threshold)).astype(np.int64)
    result = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(fake_labels, predicted_fake)),
        "balanced_accuracy": float(balanced_accuracy_score(fake_labels, predicted_fake)),
        "confusion_matrix": confusion_matrix(fake_labels, predicted_fake, labels=[0, 1]).tolist(),
        "observed_false_positive_rate": (float(predicted_fake[fake_labels == 0].mean())
                                         if np.any(fake_labels == 0) else math.nan),
        "observed_true_positive_rate": (float(predicted_fake[fake_labels == 1].mean())
                                        if np.any(fake_labels == 1) else math.nan),
    }
    fake_rate = float(fake_labels.mean()) if len(fake_labels) else math.nan
    real_rate = 1.0 - fake_rate if len(fake_labels) else math.nan
    result.update({
        "fake_base_rate": fake_rate,
        "real_base_rate": real_rate,
    })
    if np.unique(fake_labels).size < 2:
        # Every key still comes back, so a caller can read the dict without
        # knowing whether the split degenerated. A truncated smoke-test
        # validation hits this, and so would a split that lost one class.
        return dict(
            result, auroc=math.nan, average_precision=math.nan,
            fake_average_precision=math.nan, fake_ap_lift=math.nan,
            real_average_precision=math.nan, real_ap_lift=math.nan, eer=math.nan,
            target_false_positive_rate=float(target_fpr),
            tpr_at_target_fpr=math.nan,
        )
    result["auroc"] = float(roc_auc_score(fake_labels, scores))
    fake_ap = float(average_precision_score(fake_labels, scores))
    # A low anomaly score means real, hence -scores for the real-positive AP.
    real_ap = float(average_precision_score(labels, -scores))
    result["average_precision"] = fake_ap  # compatibility with V1 reports
    result["fake_average_precision"] = fake_ap
    result["fake_ap_lift"] = fake_ap / max(fake_rate, 1e-12)
    result["real_average_precision"] = real_ap
    result["real_ap_lift"] = real_ap / max(real_rate, 1e-12)
    fpr, tpr, _ = roc_curve(fake_labels, scores)
    result["target_false_positive_rate"] = float(target_fpr)
    result["tpr_at_target_fpr"] = float(np.interp(float(target_fpr), fpr, tpr))
    fnr = 1.0 - tpr
    index = int(np.nanargmin(np.abs(fpr - fnr)))
    result["eer"] = float((fpr[index] + fnr[index]) / 2.0)
    return result
