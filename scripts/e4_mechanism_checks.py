"""Three post-hoc checks on a trained TCN head. Nothing here retrains anything.

The order control left one question open. Permuting the pre-TCN sequence did
not move AUROC, but two very different mechanisms produce that result, and the
distinction matters for what may be written down:

A. score agreement. Are the per-video scores themselves unchanged, or do they
   move while the real/fake ranking survives? The first would mean the detector
   is close to permutation invariant. The second means order does change the
   representation, and only the discriminative part of that change is missing.
   AUROC alone cannot tell these apart, and the claim "the TCN does not use
   order" is only defensible under the first.

B. temporal tap norms. A kernel whose centre tap dominates behaves almost like
   a pointwise map, which would explain A without appealing to the data at all.
   This is circumstantial in both directions. A dominant centre tap is not
   proof of a 1x1 equivalence, because the residual path and the normalisation
   also shape the output. And equal tap norms say only that the neighbouring
   taps contribute terms of comparable size: ||W_L|| = ||W_C|| does not make
   W_L = W_C, the two can point anywhere relative to each other, and neither
   norm says whether the mixing carries class information. What the measurement
   can rule out is a trivial pointwise degeneration, and that is all it should
   be asked to do.

C. per-position aggregation screen. With a GAP aggregator and a single linear
   classifier the clip logit is exactly the mean of the per-position logits,

       w.(1/T sum_t y_t) + b = (1/T) sum_t (w.y_t + b),

   so max, top-k and min over t can be scored on the trained decision function
   without retraining. This is what separates the screen from the tcn_bypass
   diagnostic: there the classifier is fed a feature vector from a
   distribution it never saw, whereas here it is only ever applied to the
   features it was trained on, and the mean branch has to reproduce the
   published scores. What does change is which statistic of those per-position
   decisions becomes the clip score, and the statistic has its own
   distribution -- a max sits higher than a mean by construction. AUROC is
   rank-based so that shift does not bias the comparison, but any thresholded
   reading of these numbers would need recalibrating. The identity holds only
   for hidden_dim=0 with GAP, so both are enforced rather than assumed.

   What it cannot do is predict what a model *trained* with max pooling would
   learn. It decides whether that 5.5-hour run is worth starting.

A and B cost seconds. C needs one pass of the trunk unless a sequence cache
from temporal_order_control.py is supplied, in which case it is also seconds.

Usage:
    python scripts/e4_mechanism_checks.py \
        --checkpoint artifacts/v2/celebdfv3_e4_gap_seed42/checkpoints/best.pt \
        --config configs/v2/phase_a_celeb.yaml \
        --manifest artifacts/manifests/combined_manifest_p05.csv \
        --split val --expected-videos 8205 \
        --order-control-scores results/v2/E4_temporal_order_control_video_scores.csv \
        --sequence-cache artifacts/v2/e4_val_sequences.pt \
        --output results/v2/E4_mechanism_checks.json
"""

import argparse
import copy
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from compare_experiments import (choose_clusters, paired_cluster_bootstrap,  # noqa: E402
                                 ranking_metrics)
from temporal_order_control import (cache_sequences, load_sequence_cache,  # noqa: E402
                                    save_sequence_cache, score, verify_reference)
from video_bcnn.data import (load_manifest, preflight_face_cache,  # noqa: E402
                             seed_worker, skip_unreadable_collate)
from video_bcnn.experiment import active_records, make_dataset, select_records  # noqa: E402
from video_bcnn.model import DeterministicHead, build_feature_extractor  # noqa: E402
from video_bcnn.reporting import json_safe  # noqa: E402
from video_bcnn.utils import (load_checkpoint, load_config,  # noqa: E402
                              override_dataset_roots, override_num_workers,
                              resolve_device, save_json, seed_everything)

METADATA_COLUMNS = ("video_id", "label_real", "identity", "source_family_id",
                    "forgery_method", "relative_path", "dataset")

# Aggregations over the T positions inside one clip. All are permutation
# invariant, so none of them can recover temporal order; they ask the separate
# question of whether the evidence is concentrated in a few positions.
TOP_K = (2, 4, 8)


# --------------------------------------------------------------------------- A

def read_condition_scores(path):
    """Read the order control's per-video CSV: metadata rows plus one column per condition."""
    with open(path, "r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("No rows in {}.".format(path))
    names = [name for name in rows[0] if name not in METADATA_COLUMNS]
    if "ordered" not in names:
        raise ValueError("{} has no 'ordered' column; is it a temporal order control export?"
                         .format(path))
    metadata = [{key: row[key] for key in rows[0] if key in METADATA_COLUMNS} for row in rows]
    for row in metadata:
        row["label_real"] = int(row["label_real"])
    scores = {name: np.asarray([float(row[name]) for row in rows]) for name in names}
    return metadata, scores


def agreement(ordered, other):
    """Pearson, Spearman and absolute differences between two score vectors.

    The absolute numbers are meaningless without a scale, so they are also
    reported relative to the spread of the ordered scores: an MAE of 0.01 is
    nothing next to a standard deviation of 2 and everything next to 0.02.
    """
    spread = float(np.std(ordered))
    difference = np.abs(ordered - other)
    result = {"pearson": float("nan"), "spearman": float("nan"),
              "mae": float(np.mean(difference)),
              "max_abs_difference": float(np.max(difference)),
              "ordered_score_std": spread,
              "mae_over_ordered_std": float(np.mean(difference) / spread) if spread else None,
              "videos": int(len(ordered))}
    if len(ordered) > 1 and np.std(ordered) > 0 and np.std(other) > 0:
        result["pearson"] = float(np.corrcoef(ordered, other)[0, 1])
        result["spearman"] = float(np.corrcoef(rankdata(ordered), rankdata(other))[0, 1])
    return result


def score_agreement(path, reference=None, atol=1e-4):
    """Compare the ordered branch with every perturbed one, per class.

    The CSV is checked against the run's own published scores first. Any file
    with an 'ordered' column parses, so without that check a report could be
    assembled from another experiment's export and would look entirely normal.
    """
    metadata, scores = read_condition_scores(path)
    labels = np.asarray([row["label_real"] for row in metadata], dtype=np.int64)
    provenance = {"checked": False,
                  "reason": "no reference scores supplied; provenance unverified"}
    if reference is not None:
        provenance = verify_reference(metadata, scores["ordered"], reference, atol)
        provenance["checked"] = True
    masks = {"all": np.ones(len(labels), dtype=bool),
             "real": labels == 1, "fake": labels == 0}
    report = {"source": str(Path(path).resolve()), "videos": int(len(labels)),
              "ordered_matches_published_scores": provenance, "conditions": {},
              "reading": ("A correlation near 1 with a tiny MAE means the detector is "
                          "close to permutation invariant. A visibly lower correlation "
                          "with unchanged AUROC means order moves the score but not the "
                          "real/fake ranking, which is the more interesting outcome and "
                          "forbids writing that the TCN ignores order. The per-class rows "
                          "compare the SIZE of the perturbation between real and fake "
                          "videos and nothing else: equal MAE does not establish that the "
                          "two classes move together or in the same direction, only that "
                          "they move by a similar amount, and a class-dependent shift "
                          "that leaves the ranking intact would look the same here.")}
    for name, values in scores.items():
        if name == "ordered":
            continue
        report["conditions"][name] = {
            group: agreement(scores["ordered"][mask], values[mask])
            for group, mask in masks.items() if mask.any()}
    shuffles = [name for name in report["conditions"] if name.startswith("shuffled_")]
    if shuffles:
        report["shuffle_summary"] = {
            key: {"mean": float(np.mean([report["conditions"][name]["all"][key]
                                         for name in shuffles])),
                  "min": float(np.min([report["conditions"][name]["all"][key]
                                       for name in shuffles])),
                  "max": float(np.max([report["conditions"][name]["all"][key]
                                       for name in shuffles]))}
            for key in ("pearson", "spearman", "mae", "max_abs_difference")}
    return report


# --------------------------------------------------------------------------- B

def temporal_tap_norms(state_dict, prefix="tcn."):
    """Frobenius norm of each temporal tap of every TCN Conv1d, plus the residual gate.

    ResidualTemporalBlock computes x + act(norm(conv(x))), so the branch can be
    quiet either because the convolution is small or because the batch norm
    scale is. Both are reported; neither alone settles the mechanism.
    """
    blocks = {}
    for key in sorted(state_dict):
        if not key.startswith(prefix):
            continue
        weight = state_dict[key]
        if key.endswith("conv.weight") and weight.dim() == 3:
            taps = [float(torch.linalg.norm(weight[:, :, index].float()))
                    for index in range(weight.shape[-1])]
            total = float(sum(taps))
            centre = weight.shape[-1] // 2
            blocks.setdefault(key[len(prefix):].rsplit(".", 2)[0], {}).update({
                "kernel_size": int(weight.shape[-1]),
                "tap_frobenius_norms": taps,
                "centre_index": centre,
                "centre_share": (taps[centre] / total) if total else None,
                "uniform_share": 1.0 / weight.shape[-1]})
        if key.endswith("norm.weight") and weight.dim() == 1:
            blocks.setdefault(key[len(prefix):].rsplit(".", 2)[0], {}).update({
                "residual_norm_weight_abs_mean": float(weight.abs().float().mean()),
                "residual_norm_weight_max": float(weight.abs().float().max())})
    return blocks


def tap_norm_report(checkpoint_state, config):
    """Trained taps beside the taps of a fresh build of the same architecture.

    At initialisation the three taps are drawn from one distribution, so their
    shares sit near 1/3 by construction; the reference makes that explicit
    instead of leaving the reader to assume it.
    """
    trained = temporal_tap_norms(checkpoint_state)
    if not trained:
        raise ValueError("No TCN Conv1d weights in this checkpoint; is it a TCN-head run?")
    reference = temporal_tap_norms(build_feature_extractor(config["model"]).state_dict())
    return {"trained": trained, "fresh_initialisation": reference,
            "reading": ("A centre share far above the uniform value is consistent with "
                        "the block behaving almost pointwise, which would explain an "
                        "order-insensitive score. A share near the uniform value rules "
                        "that degeneration out and nothing more: equal Frobenius norms "
                        "mean the neighbouring taps contribute terms of comparable size, "
                        "not that they learned comparable things -- ||W_L|| = ||W_C|| "
                        "does not make W_L = W_C -- and no norm says whether the mixing "
                        "carries class information. Report it as 'no centre-tap "
                        "dominance', never as evidence of effective temporal dynamics.")}


# --------------------------------------------------------------------------- C

def aggregate_positions(anomalies, name):
    """[clips, T] per-position anomalies -> [clips]; higher means more fake."""
    if name == "mean":
        return anomalies.mean(axis=1)
    if name == "max":
        return anomalies.max(axis=1)
    if name == "min":
        return anomalies.min(axis=1)
    if name == "median":
        return np.median(anomalies, axis=1)
    if name.startswith("top"):
        k = int(name[3:])
        if not 0 < k <= anomalies.shape[1]:
            raise ValueError("top-{} needs 1..{} positions.".format(k, anomalies.shape[1]))
        return np.sort(anomalies, axis=1)[:, -k:].mean(axis=1)
    raise ValueError("Unknown aggregation {!r}.".format(name))


@torch.no_grad()
def per_position_anomalies(sequences, extractor, head, device, chunk=4,
                           precision="float32"):
    """Per-position anomalies, the video score, and the per-clip logits beside them.

    The feature path (aggregate, then classify) is what the training and the
    published evaluation ran, so it is the branch compared against full-val.
    The position path (classify, then aggregate) is the new one, and the two
    are equal in real arithmetic, so any gap between them is rounding.

    Which is why the screen runs in float32 by default while the published
    evaluation ran under autocast. The clip logits of the E4 run reach 13.7 in
    absolute value, and a single fp16 step at that magnitude is 1.3e-2, so the
    two paths can disagree by a few times 1e-3 out of pure rounding -- of the
    same order as the AUROC differences the screen is looking for, and enough
    to change which of 32 positions wins a max. float32 removes both problems
    and costs minutes on cached sequences. Pass precision="amp" to reproduce
    the published numerics instead, which is what the reference check needs.

    The per-clip logits are returned unaveraged on purpose. Comparing the two
    paths only at video level would let a positive error on one clip cancel a
    negative one on another and report agreement that does not hold anywhere.
    """
    use_amp = precision == "amp" and device.type == "cuda"
    if precision not in ("amp", "float32"):
        raise ValueError("precision must be 'amp' or 'float32'.")
    extractor.eval()
    head.eval()
    positions, feature_path, clip_logits = [], [], []
    for item in tqdm(sequences, desc="Per-position logits ({})".format(precision)):
        # The cache is fp16, as the published run produced it. Widening it is
        # exact, so float32 reads the same inputs at higher working precision.
        batch = item.to(device) if use_amp else item.to(device).float()
        clip_anomalies, clip_features = [], []
        for start in range(0, len(batch), chunk):
            part = batch[start:start + chunk]
            with torch.cuda.amp.autocast(enabled=use_amp):
                temporal = extractor.tcn_sequence(part)          # [clips, T, C]
                steps = head(temporal)                           # [clips, T]
                pooled = head(extractor.aggregator(temporal))    # [clips]
            clip_anomalies.append(-steps.float().cpu().numpy())
            clip_features.append(-pooled.float().cpu().numpy())
        positions.append(np.concatenate(clip_anomalies, axis=0))
        clip_logits.append(np.concatenate(clip_features, axis=0))
        feature_path.append(float(clip_logits[-1].mean()))
    return positions, np.asarray(feature_path), clip_logits


@torch.no_grad()
def linear_path_equivalence(sequences, extractor, head, device, chunk=4,
                            precision="amp"):
    """Both orderings of the linear map, on one and the same post-TCN tensor.

    The surgical form of the identity check, and the one that separates a
    numerical explanation from an implementation error. The TCN runs exactly as
    the published evaluation ran it, its output is widened to float32 -- which
    is exact, not a recomputation -- and only then are "classify each position,
    then average" and "average, then classify" compared. The TCN's own
    precision is therefore held fixed and cannot contribute, so whatever is
    left is the arithmetic of the two orderings alone, and in float32 that is
    rounding at 1e-7. A gap here would mean the two paths are not the same
    computation, which is a bug rather than a precision effect.

    This is deliberately not the same measurement as the screen's own float32
    check, where the TCN is re-run at higher precision and its output differs
    slightly from the published one. Reporting them separately is the point.
    """
    use_amp = precision == "amp" and device.type == "cuda"
    extractor.eval()
    head.eval()
    gaps = []
    for item in tqdm(sequences, desc="Linear-path equivalence ({})".format(precision)):
        batch = item.to(device) if use_amp else item.to(device).float()
        for start in range(0, len(batch), chunk):
            with torch.cuda.amp.autocast(enabled=use_amp):
                temporal = extractor.tcn_sequence(batch[start:start + chunk])
            widened = temporal.float()
            with torch.cuda.amp.autocast(enabled=False):
                per_position = head(widened).mean(1)
                pooled = head(widened.mean(1))
            gaps.append((per_position - pooled).abs().cpu().numpy())
    return np.concatenate(gaps)


def dispersion(positions, labels):
    """How spiky the per-position evidence is, by class.

    A sparse-anomaly account predicts fake clips whose positions scatter more
    than real ones. This is descriptive only: no interval, and it says nothing
    about whether the spread is discriminative.
    """
    def summarise(mask):
        chosen = [item for item, keep in zip(positions, mask) if keep]
        if not chosen:
            return None
        spreads = np.asarray([float(item.std(axis=1).mean()) for item in chosen])
        ranges = np.asarray([float((item.max(axis=1) - item.min(axis=1)).mean())
                             for item in chosen])
        return {"within_clip_std_mean": float(spreads.mean()),
                "within_clip_range_mean": float(ranges.mean()),
                "videos": int(len(chosen))}
    return {"real": summarise(labels == 1), "fake": summarise(labels == 0)}


def position_screen(sequences, metadata, extractor, head, device, chunk,
                    reference, reference_atol, decomposition_atol, draws, seed,
                    precision="float32"):
    labels = np.asarray([row["label_real"] for row in metadata], dtype=np.int64)
    if np.unique(labels).size != 2:
        raise ValueError("The screen needs both real and fake videos.")
    positions, feature_path, clip_logits = per_position_anomalies(
        sequences, extractor, head, device, chunk, precision)
    steps = int(positions[0].shape[1])

    # A top-k equal to the sequence length is the mean again, so it is dropped.
    names = ["mean", "max"] + ["top{}".format(k) for k in TOP_K if k < steps] + \
            ["median", "min"]
    within = {name: np.asarray([float(aggregate_positions(item, name).mean())
                                for item in positions]) for name in names}
    pooled = {name: np.asarray([float(aggregate_positions(item.reshape(1, -1), name)[0])
                                for item in positions]) for name in names}

    # The identity the whole screen rests on, measured on real data and clip by
    # clip. The video-level figure is kept as well, but it is the weaker of the
    # two: averaging eight clips can hide per-clip errors that cancel.
    #
    # The whole distribution is reported, not only the maximum over 65,640
    # clips. A maximum alone cannot distinguish the tail of a rounding
    # distribution from a handful of clips where something is actually wrong,
    # and that distinction is exactly what a reader needs when the gate trips.
    # The errors are also scaled by the logits, because an absolute tolerance
    # silently assumes a logit magnitude and this model's reach 13.7.
    gaps = np.concatenate([np.abs(item.mean(axis=1) - logits)
                           for item, logits in zip(positions, clip_logits)])
    magnitude = float(np.sqrt(np.mean(np.concatenate(clip_logits) ** 2)))
    per_clip_error = float(gaps.max())
    video_error = float(np.max(np.abs(within["mean"] - feature_path)))
    decomposition = {
        "per_clip_max_abs_error": per_clip_error,
        "per_clip_median_abs_error": float(np.median(gaps)),
        "per_clip_p99_abs_error": float(np.quantile(gaps, 0.99)),
        "per_video_max_abs_error": video_error,
        "clip_logit_rms": magnitude,
        "per_clip_max_relative_to_logit_rms": per_clip_error / magnitude if magnitude else None,
        "clips_checked": int(len(gaps)),
        "tolerance": float(decomposition_atol), "precision": precision,
    }
    if not np.isfinite(per_clip_error) or per_clip_error > decomposition_atol:
        raise ValueError(
            "Mean of per-position logits differs from the pooled logit by {:.3g} on at "
            "least one clip (median {:.3g}, RMS logit {:.3g}), above the tolerance {}. "
            "In float32 the two paths are the same computation, so a gap this large is "
            "not rounding and the max/top-k numbers must not be read."
            .format(per_clip_error, float(np.median(gaps)), magnitude, decomposition_atol))
    pooled_error = float(np.max(np.abs(pooled["mean"] - within["mean"])))

    # The published scores were produced under autocast, so the branch that has
    # to reproduce them is the autocast one. Running it costs a second pass over
    # the cached sequences and buys the provenance guarantee plus a number worth
    # having on its own: how far the precision alone moves a video score, which
    # is the floor under any small AUROC difference read off this screen.
    # The branch that has to reproduce the published scores is not a second
    # implementation of the published path -- it is that path. score() is the
    # function whose ordered branch already matched full_val_video_scores.csv
    # to the last bit, so it is called rather than imitated. Re-deriving it
    # here once cost a run: an arithmetically identical rewrite still landed
    # one fp16 ulp away on the remote card, because how a chunk is shaped
    # decides which GEMM kernel runs and therefore how it rounds. Nothing
    # below may substitute the screen's own scores for these.
    reference_scores = score(sequences, extractor, head, device, chunk=chunk)
    numerics = {"screen_precision": precision,
                "published_scores_precision": "cuda-amp",
                "reference_branch": "temporal_order_control.score",
                "amp_mean_auroc": ranking_metrics(labels, reference_scores)["auroc"],
                "screen_versus_published_video_score_max_abs_difference": float(
                    np.max(np.abs(feature_path - reference_scores)))}

    # Third and separate check: the two orderings on one identical post-TCN
    # tensor. If the screen's float32 numbers were ever to disagree, this is
    # what says whether the cause is precision or a mistake in the code.
    surgical = linear_path_equivalence(sequences, extractor, head, device, chunk,
                                       "amp" if device.type == "cuda" else "float32")
    equivalence = {
        "definition": ("published-precision TCN output widened to float32, then both "
                       "orderings of the linear classifier compared per clip"),
        "max_abs_error": float(surgical.max()),
        "median_abs_error": float(np.median(surgical)),
        "clips_checked": int(surgical.size),
        "tolerance": float(decomposition_atol)}
    if not np.isfinite(surgical.max()) or surgical.max() > decomposition_atol:
        raise ValueError(
            "On an identical post-TCN tensor in float32 the two linear orderings "
            "differ by {:.3g}, above {}. Precision cannot explain this: it is the same "
            "computation on the same input, so the implementation is wrong."
            .format(float(surgical.max()), decomposition_atol))
    check = (verify_reference(metadata, reference_scores, reference, reference_atol)
             if reference else {"passed": None, "scope": "no reference supplied"})
    cluster_key, clusters = choose_clusters(metadata)
    report = {
        "steps_per_clip": steps, "videos": int(len(labels)),
        "real": int((labels == 1).sum()), "fake": int((labels == 0).sum()),
        "aggregations": names, "cluster_key": cluster_key,
        "mean_branch_reference_check": check,
        "numerics": numerics,
        "decomposition": decomposition,
        "linear_path_equivalence": equivalence,
        "pooled_versus_within_clip_mean_max_abs_error": pooled_error,
        "within_clip": {name: ranking_metrics(labels, values)
                        for name, values in within.items()},
        "pooled_over_all_positions": {
            "scope": ("Exploratory, and not part of the position-level comparison: it "
                      "changes the clip-level aggregation at the same time as the "
                      "position-level one, so a difference here cannot be attributed to "
                      "either. No intervals are computed, and nothing should be "
                      "concluded from this block without them."),
            "metrics": {name: ranking_metrics(labels, values)
                        for name, values in pooled.items()}},
        "dispersion": dispersion(positions, labels),
        "scope": ("Post-hoc aggregation on a checkpoint trained and selected under the "
                  "mean protocol. It measures whether this trained decision function "
                  "puts its evidence in a few positions; it does not predict what a "
                  "model trained with another aggregator would learn."),
        "within_clip_definition": "aggregate the T positions inside a clip, then average clips",
        "pooled_definition": "aggregate all clips x T positions of the video at once",
    }
    report["bootstrap"] = paired_cluster_bootstrap(labels, within, clusters, draws, seed)
    return report, positions


def load_or_build_sequences(args, config, extractor, device):
    """Reuse the order control's trunk pass when it exists, otherwise run it once."""
    cache = Path(args.sequence_cache) if args.sequence_cache else None
    if cache is not None and cache.is_file():
        sequences, metadata, payload = load_sequence_cache(cache, args.expected_videos)
        return sequences, metadata, {"reused": True, "path": str(cache.resolve()),
                                     "written_for_checkpoint": payload.get("checkpoint"),
                                     "written_for_split": payload.get("split")}
    if not args.manifest:
        raise ValueError("No sequence cache and no --manifest: nothing to run the trunk on.")
    records = select_records(active_records(load_manifest(args.manifest), config), args.split)
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
    print("caching trunk sequences for {} videos ...".format(len(records)))
    sequences, metadata = cache_sequences(
        extractor, loader, device, int(config["data"].get("eval_clip_chunk_size", 4)))
    if args.expected_videos is not None and len(sequences) != args.expected_videos:
        raise RuntimeError("Cached {} of {} videos; refusing a partial screen.".format(
            len(sequences), args.expected_videos))
    info = {"reused": False}
    if cache is not None:
        info.update(save_sequence_cache(cache, sequences, metadata,
                                        args.checkpoint, args.split))
    return sequences, metadata, info


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None,
                        help="Only needed for the per-position screen; the checkpoint "
                             "carries the model configuration for the tap norms.")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--dataset-root", action="append", default=None, metavar="NAME=PATH")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--expected-videos", type=int, default=None)
    parser.add_argument("--order-control-scores", default=None,
                        help="Per-video CSV written by temporal_order_control.py. "
                             "Omit to skip check A.")
    parser.add_argument("--sequence-cache", default=None,
                        help="Read the cached pre-TCN sequences from here, or write them "
                             "here after running the trunk once.")
    parser.add_argument("--skip-position-screen", action="store_true",
                        help="Run only the cheap checks A and B.")
    parser.add_argument("--reference-scores", default=None,
                        help="Ordered full-val CSV. Defaults to the checkpoint's own "
                             "run/reports export; the mean branch must reproduce it.")
    parser.add_argument("--reference-atol", type=float, default=1e-4)
    parser.add_argument("--allow-unverified-scores", action="store_true",
                        help="Run check A without confirming that the order control CSV "
                             "belongs to this checkpoint. For inspecting an export whose "
                             "run directory is not to hand, not for results.")
    parser.add_argument("--screen-precision", choices=["float32", "amp"],
                        default="float32",
                        help="Working precision of the per-position screen. float32 by "
                             "default: the two paths are equal in real arithmetic, and "
                             "fp16 rounding on logits that reach 13.7 is the same size "
                             "as the differences being measured. 'amp' reproduces the "
                             "published numerics instead.")
    parser.add_argument("--decomposition-atol", type=float, default=1e-4,
                        help="How far the mean of per-position logits may sit from the "
                             "pooled logit, per clip. Not zero because float32 is not "
                             "exact either; raise it to about 2e-2 only with "
                             "--screen-precision amp, where a single fp16 step at this "
                             "model's logit magnitude is already 1.3e-2.")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--dump-step-logits", default=None,
                        help="Optional .npz of the per-clip, per-position anomalies.")
    parser.add_argument("--output", default="results/v2/E4_mechanism_checks.json")
    args = parser.parse_args()
    if args.draws < 1 or args.reference_atol < 0 or args.decomposition_atol <= 0:
        parser.error("draws must be positive and the tolerances nonnegative.")

    device = resolve_device("cpu" if args.skip_position_screen else "cuda")
    checkpoint = load_checkpoint(args.checkpoint, device)
    if checkpoint.get("stage") != "phase_a":
        raise ValueError("These checks read a Phase-A checkpoint.")
    config = copy.deepcopy(checkpoint["config"])
    config.setdefault("data", {})
    if config["model"].get("temporal_head") != "tcn":
        raise ValueError("A mean head has no TCN and no per-position logits to read.")
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    if not args.skip_position_screen:
        # What makes the decomposition exact rather than approximate. Checked
        # before anything is read from disk, because an unsuitable checkpoint
        # is unsuitable whatever the run directory looks like.
        if config["model"].get("temporal_aggregation") != "gap":
            raise ValueError("The per-position decomposition assumes GAP; this run used "
                             "{!r}.".format(config["model"].get("temporal_aggregation")))
        if int(config["model"].get("deterministic_hidden_dim", 0)) != 0:
            raise ValueError("The per-position decomposition assumes a single linear "
                             "classifier; this run has a hidden layer.")

    # One reference for both checks: the run's own published scores are what
    # tie every number below to this checkpoint rather than to some other run.
    reference = (Path(args.reference_scores) if args.reference_scores else
                 Path(args.checkpoint).parent.parent / "reports" /
                 ("full_val_video_scores.csv" if args.split == "val"
                  else "test_video_scores.csv"))
    if not reference.is_file():
        if not args.allow_unverified_scores:
            raise FileNotFoundError(
                "Published scores not found at {}. Both checks verify against them; "
                "pass --reference-scores, or --allow-unverified-scores to run check A "
                "without provenance (the position screen always requires them)."
                .format(reference))
        reference = None

    report = {"checkpoint": str(Path(args.checkpoint).resolve()), "split": args.split,
              "seed": seed, "temporal_aggregation": config["model"].get("temporal_aggregation"),
              "deterministic_hidden_dim": config["model"].get("deterministic_hidden_dim", 0),
              "reference_scores": str(reference) if reference else None}
    report["score_agreement"] = (
        score_agreement(args.order_control_scores, reference, args.reference_atol)
        if args.order_control_scores else None)
    report["temporal_tap_norms"] = tap_norm_report(checkpoint["extractor"], config)

    if not args.skip_position_screen:
        runtime = override_dataset_roots(load_config(args.config), args.dataset_root) \
            if args.config else {"data": {}}
        override_num_workers(runtime, args.num_workers)
        config["data"]["dataset_roots"] = runtime["data"].get(
            "dataset_roots", config["data"].get("dataset_roots"))
        config["data"]["num_workers"] = runtime["data"].get("num_workers", 0)
        extractor = build_feature_extractor(config["model"]).to(device)
        extractor.load_state_dict(checkpoint["extractor"])
        head = DeterministicHead(extractor.feature_dim, 0,
                                 config["model"].get("dropout", 0.2)).to(device)
        head.load_state_dict(checkpoint["head"])
        sequences, metadata, cache_info = load_or_build_sequences(
            args, config, extractor, device)
        if reference is None:
            raise FileNotFoundError(
                "The mean branch has to be checked against the published scores; "
                "--allow-unverified-scores does not extend to the position screen.")
        screen, positions = position_screen(
            sequences, metadata, extractor, head, device,
            int(config["data"].get("eval_clip_chunk_size", 4)), reference,
            args.reference_atol, args.decomposition_atol, args.draws, seed,
            args.screen_precision)
        screen["sequence_cache"] = cache_info
        report["position_screen"] = screen
        if args.dump_step_logits:
            # Every video carries the same clip count under a fixed evaluation
            # protocol, so this stacks into one [videos, clips, positions] array.
            Path(args.dump_step_logits).parent.mkdir(parents=True, exist_ok=True)
            shapes = {item.shape for item in positions}
            if len(shapes) != 1:
                raise ValueError("Videos returned different clip counts {}; the dump "
                                 "assumes a fixed evaluation protocol.".format(shapes))
            np.savez_compressed(
                args.dump_step_logits,
                anomalies=np.stack(positions),
                video_ids=np.asarray([row["video_id"] for row in metadata]),
                label_real=np.asarray([row["label_real"] for row in metadata]))
    else:
        report["position_screen"] = None

    save_json(args.output, json_safe(report))
    print_report(report)
    print("wrote {}".format(args.output))
    return 0


def print_report(report):
    agreements = report.get("score_agreement")
    if agreements:
        provenance = agreements["ordered_matches_published_scores"]
        print("\nA. ordered vs perturbed per-video scores (all videos)")
        print("   provenance: %s" % ("ordered column matches the published scores"
                                     if provenance.get("passed")
                                     else provenance.get("reason", "UNVERIFIED")))
        print("%-14s %9s %9s %10s %10s" % ("condition", "pearson", "spearman", "MAE", "MAE/std"))
        for name in sorted(agreements["conditions"]):
            row = agreements["conditions"][name]["all"]
            print("%-14s %9.5f %9.5f %10.5f %10.4f" % (
                name, row["pearson"], row["spearman"], row["mae"],
                row["mae_over_ordered_std"] if row["mae_over_ordered_std"] else float("nan")))
    print("\nB. TCN temporal taps (share of the kernel's Frobenius norm)")
    trained = report["temporal_tap_norms"]["trained"]
    fresh = report["temporal_tap_norms"]["fresh_initialisation"]
    for name in sorted(trained):
        block = trained[name]
        if "centre_share" not in block:
            continue
        print("%-10s taps %s  centre %.3f (uniform %.3f, at init %.3f)" % (
            name, " ".join("%.3f" % value for value in block["tap_frobenius_norms"]),
            block["centre_share"], block["uniform_share"],
            fresh.get(name, {}).get("centre_share", float("nan"))))
    screen = report.get("position_screen")
    if screen:
        print("\nC. aggregation over the %d positions inside a clip" % screen["steps_per_clip"])
        print("%-10s %8s   %s" % ("aggregate", "AUROC", "95% CI vs mean"))
        deltas = screen["bootstrap"]["deltas"]
        for name in screen["aggregations"]:
            interval = deltas.get("{}-mean".format(name))
            if name == "mean" or interval is None:
                print("%-10s %8.4f" % (name, screen["within_clip"][name]["auroc"]))
                continue
            auroc = interval["auroc"]
            print("%-10s %8.4f   [%+0.4f, %+0.4f]%s" % (
                name, screen["within_clip"][name]["auroc"], auroc["low"], auroc["high"],
                "" if auroc["high"] < 0 or auroc["low"] > 0 else "  (covers 0)"))
        decomposition = screen["decomposition"]
        print("mean branch reproduces full-val: %s" %
              screen["mean_branch_reference_check"].get("passed"))
        print("decomposition in %s over %d clips: median %.2g, p99 %.2g, max %.2g "
              "(RMS logit %.3g)" % (
                  decomposition["precision"], decomposition["clips_checked"],
                  decomposition["per_clip_median_abs_error"],
                  decomposition["per_clip_p99_abs_error"],
                  decomposition["per_clip_max_abs_error"],
                  decomposition["clip_logit_rms"]))
        surgical = screen["linear_path_equivalence"]
        print("same post-TCN tensor, both linear orderings in float32: max %.2g "
              "over %d clips" % (surgical["max_abs_error"], surgical["clips_checked"]))
        shift = screen["numerics"].get(
            "screen_versus_published_video_score_max_abs_difference")
        if shift is not None:
            print("screen precision moves a video score by at most %.3g against the "
                  "published path; treat AUROC differences near that scale as numerics"
                  % shift)


if __name__ == "__main__":
    raise SystemExit(main())
