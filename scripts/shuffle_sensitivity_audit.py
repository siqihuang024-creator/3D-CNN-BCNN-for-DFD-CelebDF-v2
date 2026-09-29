"""Is the order sensitivity of E4's scores class-dependent, or a score-level effect?

The order control established that permuting the pre-TCN sequence leaves AUROC
where it was while moving individual video scores by about 13% of their spread.
Splitting that movement by class then showed it is not symmetric: fakes move
more than reals (MAE 0.269 against 0.194), and because reals actually have the
wider score distribution, normalising by each class's own spread widens the gap
rather than closing it (0.134 against 0.082, a ratio of 1.63).

That asymmetry has two very different readings, and this script separates them.

  Class effect. Fake representations are genuinely more sensitive to temporal
  scrambling. Then there is order-derived, class-related information in the
  representation that the detector is not converting into ranking -- a much
  stronger statement than "no temporal signal here", because the signal would
  be present and unused.

  Score-level effect. The perturbation simply grows with the score, and the two
  classes sit at different score levels (real -1.78, fake +0.82). Then the
  asymmetry says nothing about classes at all.

Separating them means comparing the classes at the same score, which is harder
than it sounds and is where an earlier version of this script failed. Binning
the score and checking that both classes appear in a bin is not enough: a bin
holding reals at its bottom and fakes at its top has conditioned on nothing,
and on a synthetic case whose perturbation was a pure function of the score --
no class effect whatever -- that version still reported a ratio of 1.08 at
three bins. Every comparison here is therefore restricted to the score range
the two classes actually share inside a bin, and the counts, the ratio and the
coverage are all computed on that subset.

Nothing needs a GPU or the checkpoint: it reads the per-video CSV the order
control already wrote. Six measurements, in the order they constrain each
other:

  1. the perturbation averaged over every shuffle seed, not one permutation,
     so a single unlucky draw cannot carry the result;
  2. split by class;
  3. normalised by each class's own score spread;
  4. conditioned on the score, inside shared ranges only;
  5. an identity-clustered interval on the conditioned quantities, because
     there are only 134 real videos and a point estimate is not enough;
  6. the perturbation magnitude scored as a classifier, raw and conditioned.
     The raw figure inherits discrimination from the score it correlates with,
     so only the conditioned one answers the question.

Usage:
    python scripts/shuffle_sensitivity_audit.py \
        --scores results/v2/E4_temporal_order_control_video_scores.csv \
        --reference-scores artifacts/v2/celebdfv3_e4_gap_seed42/reports/full_val_video_scores.csv \
        --output results/v2/E4_shuffle_sensitivity_audit.json
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from compare_experiments import choose_clusters, ranking_metrics  # noqa: E402
from temporal_order_control import verify_reference  # noqa: E402
from video_bcnn.reporting import json_safe  # noqa: E402
from video_bcnn.utils import save_json  # noqa: E402


def read_scores(path):
    with open(path, "r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "ordered" not in rows[0]:
        raise ValueError("{} is not a temporal order control export.".format(path))
    shuffles = sorted(name for name in rows[0] if name.startswith("shuffled_"))
    if not shuffles:
        raise ValueError("No shuffled_* columns in {}.".format(path))
    ordered = np.asarray([float(row["ordered"]) for row in rows])
    # Averaging the seeds first: the question is how much a permutation moves a
    # score in general, not how much one particular permutation happened to.
    perturbation = np.mean([[abs(float(row["ordered"]) - float(row[name]))
                             for name in shuffles] for row in rows], axis=1)
    labels = np.asarray([int(row["label_real"]) for row in rows], dtype=np.int64)
    metadata = [{"video_id": row.get("video_id", ""), "label_real": int(row["label_real"]),
                 "identity": row.get("identity", ""),
                 "source_family_id": row.get("source_family_id", "")} for row in rows]
    return ordered, perturbation, labels, metadata, shuffles


def per_class(ordered, perturbation, labels):
    """The raw observation, before any conditioning: how far each class moves."""
    result = {}
    for name, mask in (("real", labels == 1), ("fake", labels == 0)):
        spread = float(ordered[mask].std())
        result[name] = {"videos": int(mask.sum()),
                        "mean_abs_perturbation": float(perturbation[mask].mean()),
                        "score_mean": float(ordered[mask].mean()),
                        "score_std": spread,
                        "perturbation_over_score_std": (
                            float(perturbation[mask].mean() / spread) if spread else None)}
    real, fake = result["real"], result["fake"]
    result["fake_over_real_raw"] = fake["mean_abs_perturbation"] / real["mean_abs_perturbation"]
    result["fake_over_real_normalised"] = (fake["perturbation_over_score_std"] /
                                           real["perturbation_over_score_std"])
    result["caveat"] = ("Uncontrolled for the score level, so it cannot distinguish a "
                        "class effect from the two classes sitting at different scores. "
                        "That is what the conditioned section is for.")
    return result


def shared_range_bins(ordered, labels, bins, minimum):
    """Per bin, the videos inside the score range the two classes actually share.

    Checking only that the classes' ranges intersect somewhere in a bin is not
    enough. A bin whose reals sit at -2.0 to -0.1 and whose fakes sit at -0.2 to
    +2.0 does intersect, yet comparing every video in it still compares
    low-scoring reals against high-scoring fakes, which is the confound the
    conditioning exists to remove. So the comparison is restricted to the
    intersection itself.

    Bins come from quantiles of the pooled scores rather than of one class.
    Cutting on the real quantiles would put reals in their own typical range
    and fakes in their atypical low one, which answers a different question.
    """
    edges = np.quantile(ordered, np.linspace(0, 1, int(bins) + 1))
    edges[-1] += 1e-9
    empty = np.empty(0, dtype=np.int64)
    for low, high in zip(edges[:-1], edges[1:]):
        inside = np.flatnonzero((ordered >= low) & (ordered < high))
        real = inside[labels[inside] == 1]
        fake = inside[labels[inside] == 0]
        if real.size == 0 or fake.size == 0:
            yield low, high, inside, empty, (None, None)
            continue
        lower = max(float(ordered[real].min()), float(ordered[fake].min()))
        upper = min(float(ordered[real].max()), float(ordered[fake].max()))
        if upper < lower:
            yield low, high, inside, empty, (None, None)
            continue
        shared = inside[(ordered[inside] >= lower) & (ordered[inside] <= upper)]
        if ((labels[shared] == 1).sum() < minimum or
                (labels[shared] == 0).sum() < minimum):
            yield low, high, inside, empty, (lower, upper)
            continue
        yield low, high, inside, shared, (lower, upper)


def conditioned_statistics(ordered, perturbation, labels, bins, minimum):
    """Both conditioned statistics, from one pass over the same comparable videos.

    The ratio and the stratified AUROC are produced together on purpose. They
    answer the same question two ways, and an earlier version computed them on
    different subsets and bootstrapped a third, which made it possible for a
    point estimate to be undefined while its interval was not.
    """
    rows, ratios, aurocs, weights, covered = [], [], [], [], 0
    for low, high, inside, shared, (lower, upper) in shared_range_bins(
            ordered, labels, bins, minimum):
        row = {"score_low": float(low), "score_high": float(high),
               "videos_in_bin": int(inside.size),
               "real_in_bin": int((labels[inside] == 1).sum()),
               "fake_in_bin": int((labels[inside] == 0).sum()),
               "shared_score_low": lower, "shared_score_high": upper,
               "comparable": bool(shared.size)}
        if shared.size:
            real, fake = shared[labels[shared] == 1], shared[labels[shared] == 0]
            row.update({
                "real_compared": int(real.size), "fake_compared": int(fake.size),
                "real_mean_abs_perturbation": float(perturbation[real].mean()),
                "fake_mean_abs_perturbation": float(perturbation[fake].mean()),
                "fake_over_real": float(perturbation[fake].mean() /
                                        perturbation[real].mean()),
                "magnitude_auroc": ranking_metrics(labels[shared],
                                                   perturbation[shared])["auroc"]})
            ratios.append(row["fake_over_real"])
            aurocs.append(row["magnitude_auroc"])
            weights.append(int(shared.size))
            covered += int(shared.size)
        rows.append(row)
    determinable = bool(ratios)
    return {
        "minimum_per_class": int(minimum), "bin_count": int(bins), "bins": rows,
        "comparable_bins": len(ratios),
        # Counted on the shared ranges, not on whole bins. Classes that barely
        # share a score range leave only a sliver where they can be compared,
        # and a ratio measured on a sliver must not be read as if it covered
        # the experiment.
        "comparable_coverage": float(covered / len(labels)) if len(labels) else 0.0,
        "determinable": determinable,
        "fake_over_real_conditioned": (float(np.average(ratios, weights=weights))
                                       if determinable else None),
        "magnitude_stratified_auroc": (float(np.average(aurocs, weights=weights))
                                       if determinable else None),
        "reading": ("Secondary to the matched comparison, and known to over-report. A "
                    "shared range can still be wide, leaving reals at its bottom and "
                    "fakes at its top: on a construction whose perturbation is a pure "
                    "function of the score, with no class effect at all, this statistic "
                    "returned 1.03 with an interval excluding 1, while the matched one "
                    "returned 1.000. So a value here that disagrees with the matched "
                    "result is residual confounding, not evidence. Read "
                    "comparable_coverage, and treat determinable=false as 'cannot be "
                    "decided from this sample' rather than as evidence either way."),
    }


def matched_comparison(ordered, perturbation, labels, caliper, min_matches):
    """Compare each real video against the fakes that scored almost the same.

    Binning was the first attempt and it is too blunt here. With 134 real
    videos, bins fine enough to remove the score dependence keep too few of
    them, and bins coarse enough to keep them leave reals at the bottom of each
    shared range and fakes at the top. On a construction whose perturbation is
    a pure function of the score -- no class effect at all -- ten bins still
    reported a ratio of 1.034 and an AUROC of 0.584 while covering 37% of the
    videos. Matching each real to the fakes within a caliper of its own score
    brings the same construction to 1.000 and 0.510 while keeping 131 of the
    134 reals, because it conditions on the score itself rather than on a
    bucket of it.

    The ratio is a ratio of means, not a mean of per-real ratios, and that
    distinction cost a run. Dividing each real's own perturbation into its
    partners' average puts a high-variance, right-skewed quantity in the
    denominator while the numerator is an average over roughly 140 videos, and
    the mismatch biases the result upward whatever the data say: on a
    construction where both classes are drawn from the *same* skewed
    distribution, the mean of per-real ratios reported 2.09 against a truth of
    1.00, the geometric mean 1.41 and the median 1.40, while the ratio of means
    gave 1.02 and the matched AUROC 0.51. Only the last two are usable, and
    both still recover an injected 1.5x effect (1.35 and 0.62). The biased
    figure is still reported, under a name that says so, because an earlier run
    quoted it.

    Every real is weighted equally in the AUROC, however many partners it
    found, so the sample size of that statistic is the number of matched reals
    and the identity bootstrap resamples the same unit.
    """
    real = np.flatnonzero(labels == 1)
    fake = np.flatnonzero(labels == 0)
    if real.size == 0 or fake.size == 0:
        return {"determinable": False, "reason": "one class is absent"}
    order = np.argsort(ordered[fake])
    fake_sorted, scores_sorted = fake[order], ordered[fake][order]
    partner_means, own, per_real_ratios = [], [], []
    win_rates, partners_used, gaps = [], [], []
    for index in real:
        low = np.searchsorted(scores_sorted, ordered[index] - caliper, "left")
        high = np.searchsorted(scores_sorted, ordered[index] + caliper, "right")
        partners = fake_sorted[low:high]
        if partners.size < min_matches or perturbation[index] <= 0:
            continue
        partner_means.append(float(perturbation[partners].mean()))
        own.append(float(perturbation[index]))
        per_real_ratios.append(partner_means[-1] / own[-1])
        larger = float((perturbation[partners] > perturbation[index]).sum())
        tied = float((perturbation[partners] == perturbation[index]).sum())
        win_rates.append((larger + 0.5 * tied) / partners.size)
        partners_used.append(int(partners.size))
        gaps.append(float(np.abs(ordered[partners] - ordered[index]).mean()))
    if not own:
        return {"determinable": False,
                "reason": "no real video found {} fakes within the caliper".format(
                    min_matches)}
    return {"determinable": True, "caliper": float(caliper),
            "min_matches": int(min_matches),
            "matched_reals": len(own), "real_videos": int(real.size),
            "matched_fraction_of_reals": float(len(own) / real.size),
            "median_partners_per_real": float(np.median(partners_used)),
            "mean_abs_score_gap": float(np.mean(gaps)),
            "fake_over_real": float(np.mean(partner_means) / np.mean(own)),
            "mean_paired_difference": float(np.mean(partner_means) - np.mean(own)),
            "mean_of_per_real_ratios_biased": float(np.mean(per_real_ratios)),
            "matched_auroc": float(np.mean(win_rates)),
            "reading": ("fake_over_real is a ratio of means: the matched fakes' mean "
                        "perturbation over the matched reals' mean. matched_auroc is the "
                        "mean probability that a matched fake moves more than its real. "
                        "The nulls are 1 and 0.5, and mean_paired_difference says the "
                        "same thing in score units with a null of 0. Ignore "
                        "mean_of_per_real_ratios_biased: it divides by a single real's "
                        "perturbation and reads about 2.1 even when the classes are "
                        "drawn from one distribution. mean_abs_score_gap says how tight "
                        "the matching actually was.")}


def magnitude_discrimination(ordered, perturbation, labels, bins, minimum):
    """The perturbation size as a classifier, raw and conditioned.

    The raw figure is not enough on its own: the perturbation is correlated
    with the score, and the score already separates the classes, so part of any
    raw AUROC is borrowed from it. The conditioned figures come from
    conditioned_statistics, on shared score ranges, and are reported at several
    bin counts because stratification is only as fine as its bins -- the score
    still varies inside one, so some confound survives and shrinks as the bins
    narrow.
    """
    trend = {}
    for multiple in (1, 2, 4):
        count = int(bins) * multiple
        current = conditioned_statistics(ordered, perturbation, labels, count, minimum)
        trend[str(count)] = {"auroc": current["magnitude_stratified_auroc"],
                             "fake_over_real": current["fake_over_real_conditioned"],
                             "comparable_bins": current["comparable_bins"],
                             "comparable_coverage": current["comparable_coverage"]}
    return {"raw_auroc": float(ranking_metrics(labels, perturbation)["auroc"]),
            "score_auroc_for_reference": ranking_metrics(labels, ordered)["auroc"],
            "conditioned_by_bin_count": trend,
            "reading": ("raw_auroc borrows from the score it correlates with. Read the "
                        "conditioned values, and read them across bin counts: one that "
                        "keeps falling towards 0.5 as the bins narrow was residual "
                        "confounding, one that holds steady is not.")}


def cluster_intervals(ordered, perturbation, labels, clusters, bins, minimum,
                      caliper, min_matches, draws, seed):
    """Identity-clustered intervals for the conditioned statistics.

    Each interval covers the same quantity as its point estimate, recomputed by
    the same function on each resample. Draws that leave no comparable videos
    contribute nothing and are counted, so a thin interval cannot pass for a
    firm one.
    """
    members = [np.flatnonzero(clusters == key) for key in sorted(set(clusters))]
    rng = np.random.default_rng(seed)
    ratios, aurocs, raw = [], [], []
    matched_ratios, matched_aurocs, matched_differences = [], [], []
    for _ in range(int(draws)):
        picked = np.concatenate([members[index] for index in
                                 rng.integers(len(members), size=len(members))])
        drawn_labels = labels[picked]
        if np.unique(drawn_labels).size < 2:
            continue
        drawn, drawn_scores = perturbation[picked], ordered[picked]
        raw.append(ranking_metrics(drawn_labels, drawn)["auroc"])
        pairs = matched_comparison(drawn_scores, drawn, drawn_labels,
                                   caliper, min_matches)
        if pairs["determinable"]:
            matched_ratios.append(pairs["fake_over_real"])
            matched_aurocs.append(pairs["matched_auroc"])
            matched_differences.append(pairs["mean_paired_difference"])
        current = conditioned_statistics(drawn_scores, drawn, drawn_labels, bins, minimum)
        if not current["determinable"]:
            continue
        ratios.append(current["fake_over_real_conditioned"])
        aurocs.append(current["magnitude_stratified_auroc"])

    def interval(values, null, requested):
        if not values:
            return {"draws": 0, "determinable": False,
                    "reason": "no resample held enough overlapping videos of both classes"}
        low, high = float(np.quantile(values, .025)), float(np.quantile(values, .975))
        return {"draws": len(values), "determinable": bool(len(values) >= 0.5 * requested),
                "mean": float(np.mean(values)), "low": low, "high": high,
                "excludes_null": bool(low > null or high < null)}

    return {"clusters": len(members), "draws_requested": int(draws),
            "matched_fake_over_real": interval(matched_ratios, 1.0, draws),
            "matched_auroc": interval(matched_aurocs, 0.5, draws),
            "matched_difference": interval(matched_differences, 0.0, draws),
            "binned_fake_over_real": interval(ratios, 1.0, draws),
            "binned_magnitude_auroc": interval(aurocs, 0.5, draws),
            "unconditioned_magnitude_auroc": interval(raw, 0.5, draws),
            "reading": ("The matched pair is the one the verdict rests on; the binned "
                        "pair is the coarser check beside it and the unconditioned "
                        "figure is kept only for contrast. An interval whose "
                        "determinable is false rests on too few usable resamples to be "
                        "read.")}


def verdict(matched, intervals):
    """One sentence, so the report cannot be skimmed into the wrong conclusion.

    Two traps, both of which an earlier version fell into.

    Direction. An interval excluding its null can sit on either side of it, so
    "excludes the null" does not mean "fakes move more". A ratio interval of
    0.60 to 0.72 excludes 1 while saying the opposite, and the earlier version
    reported it as fakes moving more. The direction is therefore read off the
    bounds, and the reversed case gets its own sentence rather than being
    folded into one of the others.

    Absence of evidence. Two intervals covering their nulls mean no class
    difference was detected at matched scores. They do not mean the raw
    asymmetry was entirely a score-level effect: that is the same failure to
    distinguish "no significant difference" from "equivalence" that this
    project has had to correct elsewhere. The bounds go into the sentence so
    that whatever effect is still compatible stays visible.
    """
    ratio = intervals["matched_fake_over_real"]
    auroc = intervals["matched_auroc"]
    if (not matched.get("determinable") or not ratio.get("determinable")
            or not auroc.get("determinable")):
        return ("undetermined: too few real videos found score-matched fakes, so neither "
                "a class effect nor a score-level explanation is supported")
    bounds = ("ratio {:.3f} to {:.3f}, AUROC {:.3f} to {:.3f}".format(
        ratio["low"], ratio["high"], auroc["low"], auroc["high"]))
    if ratio["low"] > 1.0 and auroc["low"] > 0.5:
        return ("class effect, fakes move more: at matched scores both intervals stay "
                "above their nulls ({})".format(bounds))
    if ratio["high"] < 1.0 and auroc["high"] < 0.5:
        return ("class effect in the opposite direction, reals move more: at matched "
                "scores both intervals stay below their nulls ({})".format(bounds))
    if not ratio["excludes_null"] and not auroc["excludes_null"]:
        return ("no class difference detected after matching ({}). This is consistent "
                "with the raw asymmetry having followed the score rather than the class, "
                "but it does not establish that: effects up to those bounds remain "
                "compatible with the data".format(bounds))
    # A larger average alongside a rank comparison that covers chance is worth
    # naming, but only as what the intervals say. A heavy tail would produce
    # it; so would other shapes, and an AUROC interval covering 0.5 means no
    # per-sample discrimination was detected, not that there is none. Neither
    # inference belongs in the sentence.
    if ratio["low"] > 1.0 and not auroc["excludes_null"]:
        difference = intervals.get("matched_difference", {})
        also = (" and the paired difference agrees ({:.4f} to {:.4f})".format(
            difference["low"], difference["high"])
            if difference.get("determinable") and difference.get("excludes_null") else "")
        return ("larger average movement for fakes, with no reliable per-sample "
                "discrimination detected: the mean ratio stays above 1{} while the "
                "matched AUROC covers 0.5 ({}). Whether the gap is carried by a minority "
                "of videos is not settled by these numbers and needs its own "
                "check".format(also, bounds))
    if auroc["low"] > 0.5 and not ratio["excludes_null"]:
        return ("matched fakes move more often than not, while the mean ratio covers 1 "
                "({}): the direction is consistent, and the size of the average "
                "difference is not pinned down by this interval".format(bounds))
    return ("mixed: the two matched statistics point in incompatible directions ({}); "
            "report both and claim neither reading".format(bounds))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", required=True,
                        help="Per-video CSV written by temporal_order_control.py.")
    parser.add_argument("--reference-scores", default=None,
                        help="Published full-val CSV. The ordered column is checked "
                             "against it so the audit cannot run on another experiment.")
    parser.add_argument("--reference-atol", type=float, default=1e-4)
    parser.add_argument("--bins", type=int, default=10,
                        help="Score bins for the conditioned comparison.")
    parser.add_argument("--min-per-class", type=int, default=10,
                        help="How many of each class a shared range needs to be compared.")
    parser.add_argument("--caliper-fraction", type=float, default=0.05,
                        help="Matching caliper as a fraction of the pooled score "
                             "standard deviation. Smaller matches more tightly and "
                             "keeps fewer reals; the report prints both.")
    parser.add_argument("--min-matches", type=int, default=3,
                        help="How many score-matched fakes a real video needs to count.")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/v2/E4_shuffle_sensitivity_audit.json")
    args = parser.parse_args()
    if args.bins < 2 or args.min_per_class < 1 or args.draws < 1:
        parser.error("bins must be at least 2, min-per-class and draws positive.")

    ordered, perturbation, labels, metadata, shuffles = read_scores(args.scores)
    if np.unique(labels).size != 2:
        raise ValueError("The audit needs both real and fake videos.")
    provenance = ({"checked": False, "reason": "no reference supplied"}
                  if args.reference_scores is None else
                  dict(verify_reference(metadata, ordered, args.reference_scores,
                                        args.reference_atol), checked=True))
    cluster_key, clusters = choose_clusters(metadata)
    caliper = float(args.caliper_fraction) * float(ordered.std())
    conditioned = conditioned_statistics(ordered, perturbation, labels,
                                         args.bins, args.min_per_class)
    matched = matched_comparison(ordered, perturbation, labels, caliper, args.min_matches)
    intervals = cluster_intervals(ordered, perturbation, labels, clusters,
                                  args.bins, args.min_per_class, caliper,
                                  args.min_matches, args.draws, args.seed)
    report = {
        "scores": str(Path(args.scores).resolve()),
        "videos": int(len(labels)), "shuffle_seeds_averaged": len(shuffles),
        "ordered_matches_published_scores": provenance,
        "cluster_key": cluster_key,
        "per_class": per_class(ordered, perturbation, labels),
        "score_matched": matched,
        # The caliper decides how much of the score dependence survives, so the
        # point estimate is shown at several of them rather than at one.
        "score_matched_by_caliper": {
            "{:.3f}".format(float(fraction) * float(ordered.std())):
                matched_comparison(ordered, perturbation, labels,
                                   float(fraction) * float(ordered.std()),
                                   args.min_matches)
            for fraction in (args.caliper_fraction * 4, args.caliper_fraction * 2,
                             args.caliper_fraction, args.caliper_fraction / 2)},
        "score_conditioned": conditioned,
        "magnitude_discrimination": magnitude_discrimination(
            ordered, perturbation, labels, args.bins, args.min_per_class),
        "bootstrap": intervals,
        "verdict": verdict(matched, intervals),
        "scope": ("Post-hoc analysis of a fixed checkpoint. It describes how this "
                  "trained detector's scores respond to permutation; it does not show "
                  "what a detector trained to exploit that response would achieve."),
    }
    save_json(args.output, json_safe(report))
    print_report(report)
    print("\nwrote {}".format(args.output))
    return 0


def print_report(report):
    classes = report["per_class"]
    print("\nperturbation averaged over %d shuffle seeds (uncontrolled)" %
          report["shuffle_seeds_averaged"])
    print("%-6s %7s %10s %10s %10s" % ("class", "videos", "MAE", "score std", "MAE/std"))
    for name in ("real", "fake"):
        row = classes[name]
        print("%-6s %7d %10.4f %10.3f %10.4f" % (
            name, row["videos"], row["mean_abs_perturbation"],
            row["score_std"], row["perturbation_over_score_std"]))
    print("fake/real: raw %.3f, normalised %.3f" % (
        classes["fake_over_real_raw"], classes["fake_over_real_normalised"]))

    matched = report["score_matched"]
    print("\nscore-matched pairs (nulls: ratio 1.000, AUROC 0.500)")
    if matched.get("determinable"):
        print("  caliper %.3f  matched %d of %d reals  median %d partners  "
              "mean score gap %.4f" % (
                  matched["caliper"], matched["matched_reals"], matched["real_videos"],
                  matched["median_partners_per_real"], matched["mean_abs_score_gap"]))
        print("  fake/real %.4f (ratio of means)   difference %+.4f   "
              "matched AUROC %.4f" % (
                  matched["fake_over_real"], matched["mean_paired_difference"],
                  matched["matched_auroc"]))
        print("  (mean of per-real ratios %.4f is structurally biased upward; "
              "it reads ~2.1 with no effect at all)" %
              matched["mean_of_per_real_ratios_biased"])
        print("  across calipers:")
        for key in sorted(report["score_matched_by_caliper"], key=float):
            row = report["score_matched_by_caliper"][key]
            if not row.get("determinable"):
                print("    %-7s undetermined (%s)" % (key, row.get("reason", "")))
                continue
            print("    %-7s ratio %.4f  AUROC %.4f  reals %d/%d" % (
                key, row["fake_over_real"], row["matched_auroc"],
                row["matched_reals"], row["real_videos"]))
    else:
        print("  undetermined: %s" % matched.get("reason", ""))

    conditioned = report["score_conditioned"]
    print("\ncoarser check, binned on the shared score ranges "
          "(%d of %d bins usable, covering %.1f%% of videos)" % (
        conditioned["comparable_bins"], len(conditioned["bins"]),
        100 * conditioned["comparable_coverage"]))
    for row in conditioned["bins"]:
        if not row["comparable"]:
            continue
        print("  shared [%7.2f,%7.2f]  real %-4d %.4f   fake %-5d %.4f   "
              "ratio %.3f  AUROC %.3f" % (
                  row["shared_score_low"], row["shared_score_high"],
                  row["real_compared"], row["real_mean_abs_perturbation"],
                  row["fake_compared"], row["fake_mean_abs_perturbation"],
                  row["fake_over_real"], row["magnitude_auroc"]))
    if conditioned["determinable"]:
        print("  conditioned ratio %.3f, conditioned magnitude AUROC %.4f" % (
            conditioned["fake_over_real_conditioned"],
            conditioned["magnitude_stratified_auroc"]))
    else:
        print("  no usable shared range: the conditioned question cannot be answered here")

    trend = report["magnitude_discrimination"]["conditioned_by_bin_count"]
    print("\nstability across bin counts (raw AUROC %.4f, the score itself %.4f)" % (
        report["magnitude_discrimination"]["raw_auroc"],
        report["magnitude_discrimination"]["score_auroc_for_reference"]))
    for count in sorted(trend, key=int):
        row = trend[count]
        print("  %-4s bins  ratio %-7s AUROC %-7s coverage %5.1f%%" % (
            count,
            "%.3f" % row["fake_over_real"] if row["fake_over_real"] is not None else "n/a",
            "%.4f" % row["auroc"] if row["auroc"] is not None else "n/a",
            100 * row["comparable_coverage"]))

    print("")
    for name in ("matched_fake_over_real", "matched_auroc",
                 "binned_fake_over_real", "binned_magnitude_auroc"):
        interval = report["bootstrap"][name]
        if not interval.get("draws"):
            print("%-30s undetermined (%s)" % (name, interval.get("reason", "no draws")))
            continue
        print("%-30s %.3f  95%% CI [%.3f, %.3f]%s%s" % (
            name, interval["mean"], interval["low"], interval["high"],
            "" if interval["excludes_null"] else "   (covers the null)",
            "" if interval["determinable"] else "   (too few usable draws)"))
    print("\nverdict: %s" % report["verdict"])


if __name__ == "__main__":
    raise SystemExit(main())
