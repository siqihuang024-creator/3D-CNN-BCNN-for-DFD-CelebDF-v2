"""Contracts for full-validation score export and paired comparison."""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.compare_experiments import (align_tables, alignment_diagnostics,
                                         choose_clusters,
                                         paired_cluster_bootstrap)
from video_bcnn.reporting import save_evaluation_report


class ScoreExportTests(unittest.TestCase):
    @staticmethod
    def values():
        return {
            "labels": np.asarray([1, 0]),
            "scores": np.asarray([0.1, 0.9]),
            "means": np.asarray([-0.1, -0.9]),
            "stds": np.asarray([0.0, 0.0]),
            "paths": ["/data/real.mp4", "/data/fake.mp4"],
            "relative_paths": ["real.mp4", "fake.mp4"],
            "datasets": np.asarray(["CelebDFv3", "CelebDFv3"]),
            "methods": np.asarray(["real", "FaceSwap"]),
            "target_ids": ["id0", "id1"], "donor_ids": ["", "id2"],
            "source_clips": np.asarray(["family0", "family1"]),
            "embedding_norms": np.asarray([1.0, 2.0]),
            "clip_scores": [np.asarray([0.0, 0.2]), np.asarray([0.8, 1.0])],
            "clip_means": [np.asarray([0.0, -0.2]), np.asarray([-0.8, -1.0])],
            "clip_stds": [np.zeros(2), np.zeros(2)],
            "clip_frame_indices": [np.asarray([[0, 2], [10, 12]]),
                                   np.asarray([[1, 3], [11, 13]])],
        }

    def test_video_and_clip_csv_row_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics = {"split": "val", "checkpoint": "best.pt",
                       "experiment": "E0", "seed": 42}
            save_evaluation_report(self.values(), metrics, directory, "val",
                                   report_name="full_val")
            video_path = Path(directory) / "full_val_video_scores.csv"
            clip_path = Path(directory) / "full_val_clip_scores.csv"
            with open(video_path, newline="", encoding="utf-8") as handle:
                videos = list(csv.DictReader(handle))
            with open(clip_path, newline="", encoding="utf-8") as handle:
                clips = list(csv.DictReader(handle))
            self.assertEqual(len(videos), 2)
            self.assertEqual(len(clips), 4)
            self.assertEqual(clips[0]["clip_start_frame"], "0")
            self.assertEqual(clips[0]["clip_end_frame"], "2")
            self.assertEqual(videos[0]["source_family_id"], "family0")
            self.assertTrue((Path(directory) / "val_scores.csv").is_file())


class FrameIndexCopyTests(unittest.TestCase):
    def test_frame_indices_do_not_alias_the_batch_tensor(self):
        import torch
        from video_bcnn.evaluation import _frame_indices
        batch = {"clip_frame_indices": torch.arange(12).reshape(1, 3, 4)}
        values = _frame_indices(batch, 0, 3)
        self.assertEqual(values.shape, (3, 4))
        self.assertFalse(np.shares_memory(values, batch["clip_frame_indices"].numpy()))


class FrameOrderControlTests(unittest.TestCase):
    def test_shuffle_permutes_time_keeps_frames_and_is_reproducible(self):
        import torch
        from evaluate_3d_bcnn import ShuffledFrameLoader
        # [video, clip, channel, time, height, width]; each frame is a constant
        # plane carrying its own index, so a permutation is easy to read off.
        clips = torch.arange(6, dtype=torch.float32).reshape(1, 1, 1, 6, 1, 1)
        batch = {"clips": clips.clone(), "label": torch.tensor([1])}
        order = [b["clips"].flatten().tolist() for b in ShuffledFrameLoader([batch], 42)]
        self.assertEqual(sorted(order[0]), list(range(6)))   # same frames
        self.assertNotEqual(order[0], list(range(6)))        # different order
        again = {"clips": clips.clone(), "label": torch.tensor([1])}
        repeat = [b["clips"].flatten().tolist() for b in ShuffledFrameLoader([again], 42)]
        self.assertEqual(order[0], repeat[0])                # reproducible
        skipped = [b for b in ShuffledFrameLoader([None, {"_skip_only": True}], 42)]
        self.assertEqual(skipped, [None, {"_skip_only": True}])

    def test_shuffle_does_not_mutate_source_and_tracks_frame_indices(self):
        import torch
        from evaluate_3d_bcnn import ShuffledFrameLoader
        clips = torch.arange(6.).reshape(1, 1, 1, 6, 1, 1)
        indices = torch.arange(6).reshape(1, 1, 6)
        first = {"clips": clips, "clip_frame_indices": indices,
                 "relative_path": ["x.mp4"], "dataset": ["CelebDFv3"]}
        other = {"clips": clips.clone(), "relative_path": ["other.mp4"],
                 "dataset": ["CelebDFv3"]}
        shuffled = list(ShuffledFrameLoader([first], 42))[0]
        repeat = list(ShuffledFrameLoader([other, first], 42))[1]
        self.assertTrue(torch.equal(first["clips"], clips))
        self.assertTrue(torch.equal(indices.flatten(), torch.arange(6)))
        self.assertTrue(torch.equal(shuffled["clips"].flatten().long(),
                                    shuffled["clip_frame_indices"].flatten()))
        self.assertTrue(torch.equal(shuffled["clips"], repeat["clips"]))


class SequenceShuffleTests(unittest.TestCase):
    @staticmethod
    def extractor(temporal_head):
        from video_bcnn.model import Stable3DFeatureExtractor
        return Stable3DFeatureExtractor(spatial_output_size=4, temporal_steps=8,
                                        temporal_head=temporal_head).eval()

    def test_sequence_shuffle_changes_a_tcn_output_and_is_reproducible(self):
        import torch
        model = self.extractor("tcn")
        clips = torch.randn(2, 3, 8, 64, 64)
        with torch.no_grad():
            ordered = model(clips)
            model.set_sequence_shuffle(42)
            first = model(clips)
            model.set_sequence_shuffle(42)
            second = model(clips)
        self.assertFalse(torch.allclose(ordered, first))
        self.assertTrue(torch.allclose(first, second))
        model.set_sequence_shuffle(None)
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(clips), ordered))

    def test_mean_head_refuses_the_vacuous_control(self):
        with self.assertRaisesRegex(ValueError, "permutation invariant"):
            self.extractor("mean").set_sequence_shuffle(42)

    def test_cached_two_step_path_matches_the_single_forward(self):
        # The order control caches trunk_sequence and re-runs only the head, so
        # the split must reproduce forward() exactly or every drop it reports
        # would be an artefact of the refactor.
        import torch
        model = self.extractor("tcn")
        clips = torch.randn(2, 3, 8, 64, 64)
        with torch.no_grad():
            direct = model(clips)
            staged = model.temporal_head_forward(model.trunk_sequence(clips))
        self.assertTrue(torch.allclose(direct, staged, atol=1e-6))

    def test_block_shuffle_keeps_windows_contiguous(self):
        import torch
        from scripts.temporal_order_control import block_order, reversed_order
        generator = torch.Generator().manual_seed(0)
        order = block_order(4)(12, generator).tolist()
        self.assertEqual(sorted(order), list(range(12)))
        blocks = [order[i:i + 4] for i in range(0, 12, 4)]
        for block in blocks:                      # each window stays in sequence
            self.assertEqual(block, list(range(block[0], block[0] + 4)))
        self.assertEqual(reversed_order(5, None).tolist(), [4, 3, 2, 1, 0])

    def test_block16_always_swaps_halves(self):
        import torch
        from scripts.temporal_order_control import block_order
        for seed in range(10):
            result = block_order(16)(32, torch.Generator().manual_seed(seed))
            self.assertEqual(result.tolist(), list(range(16, 32)) + list(range(16)))
        with self.assertRaises(ValueError):
            block_order(32)(32, torch.Generator())

    def test_refactored_forward_matches_original_formula(self):
        import torch
        for kind in ("mean", "tcn"):
            model = self.extractor(kind)
            clips = torch.randn(2, 3, 8, 64, 64)
            with torch.no_grad():
                values = clips
                for conv, pool, norm in ((model.conv1, model.pool1, model.norm1),
                                         (model.conv2, model.pool2, model.norm2),
                                         (model.conv3, model.pool3, model.norm3)):
                    values = model.activation(norm(pool(conv(values))))
                values = model._spatial_reduce(values)
                if kind == "mean":
                    original = model.output_norm(values.mean(2)).flatten(1)
                else:
                    values = model.output_norm(values)
                    sequence = values.permute(0, 2, 1, 3, 4).reshape(2, 8, -1)
                    original = model.aggregator(model.tcn(sequence.transpose(1, 2)).transpose(1, 2))
                self.assertTrue(torch.equal(original, model(clips)))


class CachedOrderControlTests(unittest.TestCase):
    def test_order_control_main_writes_all_conditions_and_reference_check(self):
        import torch
        from unittest.mock import patch
        from scripts import temporal_order_control as control
        from video_bcnn.model import DeterministicHead
        config = {"device": "cpu", "seed": 42,
                  "data": {"dataset_roots": {"CelebDFv3": "unused"}, "num_workers": 0,
                           "eval_clip_chunk_size": 2, "eval_clips_per_video": 2},
                  "model": {"temporal_head": "tcn", "temporal_steps": 8,
                            "spatial_output_size": 4, "feature_dim": 512}}
        model = SequenceShuffleTests.extractor("tcn")
        head = DeterministicHead(512, 0, .2).eval()
        batches = [{"clips": torch.randn(1, 2, 3, 8, 64, 64), "label": torch.tensor([label]),
                    "relative_path": ["{}.mp4".format(label)], "dataset": ["CelebDFv3"],
                    "method": ["real" if label else "fake"], "target_id": ["person"],
                    "source_clip": ["source"]} for label in (1, 0)]
        sequences, metadata = control.cache_sequences(model, batches, torch.device("cpu"), 2)
        scores = control.score(sequences, model, head, torch.device("cpu"), chunk=2)
        checkpoint = {"stage": "phase_a", "config": config,
                      "extractor": model.state_dict(), "head": head.state_dict()}
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "reference.csv"
            with reference.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["video_id", "label_real", "video_score"])
                for row, value in zip(metadata, scores):
                    writer.writerow([row["video_id"], row["label_real"], value])
            output = Path(directory) / "control.json"
            args = ["control", "--config", "unused", "--manifest", "unused",
                    "--checkpoint", "unused/best.pt", "--expected-videos", "2",
                    "--reference-scores", str(reference), "--output", str(output),
                    "--shuffle-seeds", "2", "--block-widths", "2", "4", "--draws", "5"]
            with patch.object(sys, "argv", args), patch.object(control, "load_config", return_value=config), \
                 patch.object(control, "load_checkpoint", return_value=checkpoint), \
                 patch.object(control, "load_manifest", return_value=[]), \
                 patch.object(control, "active_records", return_value=[]), \
                 patch.object(control, "select_records", return_value=[{}, {}]), \
                 patch.object(control, "preflight_face_cache"), patch.object(control, "make_dataset"), \
                 patch.object(control, "DataLoader", return_value=batches):
                self.assertEqual(control.main(), 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["ordered_reference_check"]["passed"])
            self.assertEqual(set(report["conditions"]),
                             {"ordered", "shuffled_0", "shuffled_1", "reversed", "block2", "block4", "tcn_bypass"})
            self.assertEqual(report["bootstrap"]["draws_valid"], 5)
            self.assertIn("mean_drop_ci", report["shuffle_summary"])
            with Path(report["video_scores_csv"]).open(newline="", encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)

    def test_per_sample_seeds_and_chunk_order_are_independent(self):
        from scripts.temporal_order_control import sample_generator, random_order
        first = random_order(32, sample_generator(42, "video1", 0))
        repeat = random_order(32, sample_generator(42, "video1", 0))
        other = random_order(32, sample_generator(42, "video2", 0))
        self.assertEqual(first.tolist(), repeat.tolist())
        self.assertNotEqual(first.tolist(), other.tolist())

    def test_cached_scores_match_full_evaluation_on_cpu_and_cuda(self):
        import torch
        from scripts.temporal_order_control import cache_sequences, score
        from video_bcnn.evaluation import score_deterministic
        from video_bcnn.model import DeterministicHead
        for device in [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else []):
            model = SequenceShuffleTests.extractor("tcn").to(device)
            head = DeterministicHead(512, 0, .2).to(device).eval()
            batch = {"clips": torch.randn(1, 3, 3, 8, 64, 64), "label": torch.tensor([1]),
                     "path": ["/data/x.mp4"], "relative_path": ["x.mp4"],
                     "dataset": ["CelebDFv3"], "method": ["real"], "target_id": ["id0"],
                     "donor_id": [""], "source_clip": ["source0"]}
            direct = -score_deterministic(model, head, [batch], device, 2, True)["logits"]
            sequences, metadata = cache_sequences(model, [batch], device, 2)
            cached = score(sequences, model, head, device, chunk=2)
            np.testing.assert_allclose(cached, direct, rtol=0, atol=1e-6)
            self.assertEqual(metadata[0]["video_id"], "CelebDFv3::x.mp4")

    def test_reference_mismatch_is_rejected(self):
        from scripts.temporal_order_control import verify_reference
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.csv"
            path.write_text("video_id,label_real,video_score\nv1,1,0.1\nv2,0,0.9\n", encoding="utf-8")
            meta = [{"video_id": "v1", "label_real": 1}, {"video_id": "v2", "label_real": 0}]
            self.assertTrue(verify_reference(meta, np.array([.1, .9]), path)["passed"])
            with self.assertRaisesRegex(ValueError, "differs"):
                verify_reference(meta, np.array([.2, .9]), path)

    def test_mean_shuffle_ci_uses_mean_metrics_not_mean_scores(self):
        from scripts.temporal_order_control import control_intervals, ranking_metrics
        labels = np.array([1, 0, 1, 0])
        conditions = {"ordered": np.array([0., 1., 0., 1.]),
                      "shuffled_0": np.array([0., 1., 0., 1.]),
                      "shuffled_1": np.array([1., 0., 1., 0.])}
        clusters = np.array(["a", "a", "b", "b"])
        deltas, mean, valid = control_intervals(labels, conditions, clusters, 20, 42)
        self.assertEqual(valid, 20)
        self.assertEqual(deltas["shuffled_1"]["auroc"]["high"], -1.)
        self.assertEqual(mean["auroc"]["low"], .5)
        self.assertEqual(mean["auroc"]["high"], .5)

    def test_shuffle_report_does_not_overwrite_legacy_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "val_scores.csv"
            path.write_text("ordered sentinel", encoding="utf-8")
            save_evaluation_report(ScoreExportTests.values(), {"split": "val"}, directory,
                                   "val", "full_val_shuffled", write_legacy=False)
            self.assertEqual(path.read_text(encoding="utf-8"), "ordered sentinel")


class ExtendedBudgetTests(unittest.TestCase):
    def test_extended_budget_cli_is_an_independent_separate_run(self):
        from unittest.mock import patch
        import pretrain_extractor as training
        with tempfile.TemporaryDirectory() as directory:
            config = {"seed": 42, "device": "cpu", "data": {
                "active_datasets": ["CelebDFv3"], "dataset_roots": {}},
                "model": {"temporal_aggregation": "gap"},
                "train": {"run_dir": str(Path(directory) / "celebdfv3_e4_gap_seed42"), "epochs": 60}}
            args = ["training", "--config", "unused", "--manifest", "unused",
                    "--max-epochs", "90", "--run-suffix", "E4_long"]
            with patch.object(sys, "argv", args), patch.object(training, "load_config", return_value=config), \
                 patch.object(training, "verify_dataset_roots", side_effect=RuntimeError("preflight sentinel")) as verify:
                with self.assertRaisesRegex(RuntimeError, "preflight sentinel"):
                    training.main()
                received = verify.call_args[0][0]
                self.assertEqual(received["train"]["epochs"], 90)
                self.assertEqual(Path(received["train"]["run_dir"]).name, "celebdfv3_e4_long_seed42")
                self.assertIn("not resume", received["train"]["optimization_scope"])
            target = Path(directory) / "celebdfv3_e4_long_seed42" / "checkpoints"
            target.mkdir(parents=True)
            (target / "best.pt").write_bytes(b"sentinel")
            with patch.object(sys, "argv", args), patch.object(training, "load_config", return_value=config):
                with self.assertRaises(FileExistsError):
                    training.main()
            self.assertEqual((target / "best.pt").read_bytes(), b"sentinel")


class ShortcutSplitTests(unittest.TestCase):
    def test_methods_split_by_what_the_control_alone_achieves(self):
        from scripts.shortcut_split_report import control_by_method
        table = {}
        # The control ranks "solved" fakes above every real and "neutral" fakes
        # interleaved with them.
        for index in range(4):
            table["r%d" % index] = {"label_real": 1, "video_score": 0.5 + index,
                                    "forgery_method": "real"}
        for index in range(4):
            table["s%d" % index] = {"label_real": 0, "video_score": 100.0 + index,
                                    "forgery_method": "solved"}
            table["n%d" % index] = {"label_real": 0, "video_score": 0.5 + index,
                                    "forgery_method": "neutral"}
        groups, per_method = control_by_method(table, 0.05, 0.95)
        self.assertEqual(groups["separable"], ["solved"])
        self.assertEqual(groups["neutral"], ["neutral"])
        self.assertEqual(per_method["solved"]["control_auroc"], 1.0)

    def test_frozen_groups_must_cover_each_method_exactly_once(self):
        from scripts.shortcut_split_report import validate_groups
        valid = {"neutral": ["a"], "separable": ["b"], "intermediate": []}
        validate_groups(valid, {"a", "b"})
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_groups(valid, {"a", "b", "c"})
        with self.assertRaisesRegex(ValueError, "multiple"):
            validate_groups({"neutral": ["a"], "separable": ["a"],
                             "intermediate": []}, {"a"})


class PairedComparisonTests(unittest.TestCase):
    @staticmethod
    def table(offset=0.0):
        rows = {}
        for index, (label, score) in enumerate(zip([1, 1, 0, 0],
                                                   [0.1, 0.2, 0.8, 0.9])):
            video_id = "v{}".format(index)
            rows[video_id] = {
                "video_id": video_id, "label_real": label,
                "video_score": score + offset,
                "source_family_id": "", "identity": "id{}".format(index % 2),
                "forgery_method": "real" if label else "FaceSwap",
            }
        return rows

    def test_score_merge_requires_identical_video_ids(self):
        left, right = self.table(), self.table(0.01)
        del right["v3"]
        with self.assertRaises(ValueError):
            align_tables({"E0": left, "E1": right})

    def test_intersect_compares_on_the_shared_videos(self):
        left, right = self.table(), self.table(0.01)
        del right["v3"]                      # a control that cannot score one video
        with self.assertRaises(ValueError):
            align_tables({"E0": left, "S": right})
        video_ids, _, scores = align_tables({"E0": left, "S": right}, intersect=True)
        self.assertEqual(video_ids, ["v0", "v1", "v2"])
        self.assertEqual(len(scores["E0"]), 3)
        audit = alignment_diagnostics({"E0": left, "S": right}, video_ids)
        self.assertFalse(audit["video_ids_identical"])
        self.assertEqual(audit["files"]["E0"]["videos_dropped"], 1)
        self.assertEqual(audit["files"]["E0"]["fake_dropped"], 1)
        self.assertEqual(audit["files"]["S"]["videos_dropped"], 0)

    def test_cluster_key_defaults_to_identity(self):
        _, metadata, _ = align_tables({"E0": self.table(), "E1": self.table(0.01)})
        key, clusters = choose_clusters(metadata)
        self.assertEqual(key, "identity")
        self.assertEqual(len(clusters), 4)

    def test_cluster_key_falls_back_to_source_family(self):
        rows = [dict(row, identity="", source_family_id="f{}".format(index))
                for index, row in enumerate(self.table().values())]
        key, _ = choose_clusters(rows)
        self.assertEqual(key, "source_family_id")

    def test_operating_points_are_reported_and_bootstrapped(self):
        from scripts.compare_experiments import ranking_metrics, BOOTSTRAP_METRICS
        # Twenty reals, twenty fakes, and a score that ranks every fake above
        # every real: both operating points must read 1.0.
        labels = np.asarray([1] * 20 + [0] * 20)
        scores = np.concatenate([np.linspace(0.0, 0.4, 20), np.linspace(0.6, 1.0, 20)])
        perfect = ranking_metrics(labels, scores)
        self.assertEqual(perfect["tpr_at_5pct_fpr"], 1.0)
        self.assertEqual(perfect["tpr_at_10pct_fpr"], 1.0)
        # A score carrying no information sits near the false-positive rate.
        uninformative = ranking_metrics(labels, np.tile(np.linspace(0, 1, 20), 2))
        self.assertLess(uninformative["tpr_at_5pct_fpr"], 0.2)
        self.assertLessEqual(uninformative["tpr_at_5pct_fpr"],
                             uninformative["tpr_at_10pct_fpr"])
        report = paired_cluster_bootstrap(
            labels, {"A": scores, "B": scores + 0.01},
            np.asarray(["c{}".format(index % 5) for index in range(40)]),
            draws=20, seed=42)
        for metric in BOOTSTRAP_METRICS:
            self.assertIn(metric, report["deltas"]["B-A"])

    def test_paired_cluster_bootstrap_runs(self):
        labels = np.asarray([1, 1, 0, 0])
        scores = {"E0": np.asarray([0.3, 0.4, 0.6, 0.7]),
                  "E1": np.asarray([0.1, 0.2, 0.8, 0.9])}
        clusters = np.asarray(["a", "b", "a", "b"])
        report = paired_cluster_bootstrap(labels, scores, clusters,
                                          draws=20, seed=42)
        self.assertEqual(report["draws_valid"], 20)
        self.assertIn("E1-E0", report["deltas"])
        self.assertIsNotNone(report["deltas"]["E1-E0"]["auroc"]["low"])


class MechanismCheckTests(unittest.TestCase):
    """The per-position screen is only meaningful if its identity actually holds."""

    @staticmethod
    def pieces():
        import torch
        from video_bcnn.model import DeterministicHead
        torch.manual_seed(0)
        model = SequenceShuffleTests.extractor("tcn")
        head = DeterministicHead(512, 0, .2).eval()
        return model, head

    def test_per_position_logits_average_to_the_pooled_logit(self):
        import torch
        model, head = self.pieces()
        sequence = torch.randn(3, 8, 512)
        with torch.no_grad():
            temporal = model.tcn_sequence(sequence)
            pooled = head(model.aggregator(temporal))
            per_step = head(temporal)
        self.assertEqual(tuple(per_step.shape), (3, 8))
        np.testing.assert_allclose(per_step.mean(1).numpy(), pooled.numpy(),
                                   rtol=0, atol=1e-5)

    def test_tcn_sequence_is_the_tensor_the_aggregator_receives(self):
        import torch
        model, _ = self.pieces()
        sequence = torch.randn(2, 8, 512)
        with torch.no_grad():
            np.testing.assert_allclose(
                model.temporal_head_forward(sequence).numpy(),
                model.aggregator(model.tcn_sequence(sequence)).numpy(),
                rtol=0, atol=0)

    def test_position_screen_mean_branch_reproduces_the_cached_score(self):
        import torch
        from scripts.e4_mechanism_checks import per_position_anomalies, position_screen
        from scripts.temporal_order_control import cache_sequences, score
        device = torch.device("cpu")
        model, head = self.pieces()
        batches = [{"clips": torch.randn(1, 2, 3, 8, 64, 64), "label": torch.tensor([label]),
                    "relative_path": ["{}.mp4".format(index)], "dataset": ["CelebDFv3"],
                    "method": ["real" if label else "fake"],
                    "target_id": ["person{}".format(index)], "source_clip": ["source"]}
                   for index, label in enumerate((1, 0, 1, 0))]
        sequences, metadata = cache_sequences(model, batches, device, 2)
        ordered = score(sequences, model, head, device, chunk=2)
        positions, feature_path, clip_logits = per_position_anomalies(
            sequences, model, head, device, 2)
        # Per clip, not only per video: video-level agreement can hide clip
        # errors of opposite sign that cancel when the clips are averaged.
        for item, logits in zip(positions, clip_logits):
            np.testing.assert_allclose(item.mean(axis=1), logits, rtol=0, atol=1e-5)
        np.testing.assert_allclose(feature_path, ordered, rtol=0, atol=1e-5)
        step_mean = np.asarray([float(item.mean()) for item in positions])
        np.testing.assert_allclose(step_mean, ordered, rtol=0, atol=1e-5)
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "reference.csv"
            with reference.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["video_id", "label_real", "video_score"])
                for row, value in zip(metadata, ordered):
                    writer.writerow([row["video_id"], row["label_real"], value])
            report, positions = position_screen(sequences, metadata, model, head,
                                                device, 2, reference, 1e-4, 1e-3, 5, 42)
        self.assertEqual(np.stack(positions).shape, (4, 2, 8))
        self.assertTrue(report["mean_branch_reference_check"]["passed"])
        self.assertLess(report["decomposition"]["per_clip_max_abs_error"], 1e-4)
        self.assertEqual(report["decomposition"]["clips_checked"], 8)
        self.assertEqual(report["decomposition"]["precision"], "float32")
        # Pooling every clip-position at once equals the within-clip mean.
        self.assertLess(report["pooled_versus_within_clip_mean_max_abs_error"], 1e-6)
        self.assertIn("max-mean", report["bootstrap"]["deltas"])

    def test_linear_orderings_agree_on_one_identical_post_tcn_tensor(self):
        """Separates precision from a bug: same tensor, same precision, two orderings."""
        import torch
        from scripts.e4_mechanism_checks import linear_path_equivalence
        model, head = self.pieces()
        sequences = [torch.randn(3, 8, 512) for _ in range(2)]
        gaps = linear_path_equivalence(sequences, model, head, torch.device("cpu"), 2,
                                       "float32")
        self.assertEqual(gaps.size, 6)
        self.assertLess(float(gaps.max()), 1e-5)

    def test_float32_screen_is_tight_where_autocast_is_not(self):
        """Regression for the first remote run, which tripped a tolerance set for fp32.

        The two paths are the same computation, so float32 agrees to rounding.
        Under autocast they can differ by far more, because a single fp16 step
        at this model's logit magnitude is already 1e-2. That is a property of
        the arithmetic, not of the checkpoint, which is why the screen runs in
        float32 and only the reference check uses autocast.
        """
        import torch
        from scripts.e4_mechanism_checks import per_position_anomalies
        if not torch.cuda.is_available():
            self.skipTest("no CUDA device; run this on the training host")
        device = torch.device("cuda")
        model, head = self.pieces()
        model, head = model.to(device), head.to(device)
        sequences = [torch.randn(8, 32, 512).half() for _ in range(16)]
        # Calibrate the head to the logit scale of the real run (clip scores
        # with a standard deviation near 2.15 and a reach past 10), because the
        # gap scales with it -- that is precisely what the first remote run
        # exposed and what a small-magnitude test cannot see.
        with torch.no_grad():
            probe = head(model.aggregator(model.tcn_sequence(
                sequences[0].float().to(device))))
            head.out.weight *= 2.15 / float(probe.std())

        def worst(precision):
            positions, _, clip_logits = per_position_anomalies(
                sequences, model, head, device, 4, precision)
            return max(float(np.max(np.abs(item.mean(axis=1) - logits)))
                       for item, logits in zip(positions, clip_logits))

        float32_error, amp_error = worst("float32"), worst("amp")
        self.assertLess(float32_error, 1e-4)
        # The incident: at this magnitude autocast alone exceeds the old gate.
        self.assertGreater(amp_error, 1e-3)
        self.assertGreater(amp_error, 100 * float32_error)

    def test_position_screen_rejects_a_broken_decomposition(self):
        import torch
        from scripts.e4_mechanism_checks import position_screen
        model, head = self.pieces()
        device = torch.device("cpu")
        sequences = [torch.randn(2, 8, 512) for _ in range(2)]
        metadata = [{"video_id": "v{}".format(index), "label_real": index,
                     "identity": "id{}".format(index)} for index in range(2)]
        with self.assertRaisesRegex(ValueError, "not rounding"):
            position_screen(sequences, metadata, model, head, device, 2,
                            None, 1e-4, -1.0, 5, 42)

    def test_aggregations_read_the_fake_end_of_the_positions(self):
        from scripts.e4_mechanism_checks import aggregate_positions
        values = np.asarray([[0.0, 1.0, 2.0, 3.0]])
        self.assertEqual(aggregate_positions(values, "max")[0], 3.0)
        self.assertEqual(aggregate_positions(values, "min")[0], 0.0)
        self.assertEqual(aggregate_positions(values, "top2")[0], 2.5)
        self.assertEqual(aggregate_positions(values, "mean")[0], 1.5)
        with self.assertRaises(ValueError):
            aggregate_positions(values, "top9")

    def test_score_agreement_separates_a_shift_from_a_reordering(self):
        from scripts.e4_mechanism_checks import score_agreement
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "conditions.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["video_id", "label_real", "identity",
                                 "ordered", "shuffled_0", "reversed"])
                for index, (label, ordered) in enumerate(zip([1, 1, 0, 0],
                                                             [0.0, 1.0, 2.0, 3.0])):
                    # shuffled_0 is a rigid shift: ranking intact, scores moved.
                    # reversed inverts the ranking outright.
                    writer.writerow(["v{}".format(index), label, "id{}".format(index),
                                     ordered, ordered + 0.5, -ordered])
            report = score_agreement(path)
            self.assertFalse(report["ordered_matches_published_scores"]["checked"])
        shifted = report["conditions"]["shuffled_0"]["all"]
        self.assertAlmostEqual(shifted["spearman"], 1.0, places=6)
        self.assertAlmostEqual(shifted["mae"], 0.5, places=6)
        self.assertAlmostEqual(shifted["max_abs_difference"], 0.5, places=6)
        self.assertAlmostEqual(report["conditions"]["reversed"]["all"]["spearman"],
                               -1.0, places=6)
        self.assertEqual(set(report["conditions"]["shuffled_0"]), {"all", "real", "fake"})
        self.assertIn("pearson", report["shuffle_summary"])

    def test_score_agreement_rejects_a_csv_from_another_run(self):
        """Any file with an 'ordered' column parses, so provenance has to be checked."""
        from scripts.e4_mechanism_checks import score_agreement
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            conditions = root / "conditions.csv"
            with conditions.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["video_id", "label_real", "identity", "ordered", "reversed"])
                for index, (label, value) in enumerate(zip([1, 0], [0.25, 0.75])):
                    writer.writerow(["v{}".format(index), label, "id0", value, value])
            published = root / "published.csv"
            published.write_text(
                "video_id,label_real,video_score\nv0,1,0.25\nv1,0,0.75\n", encoding="utf-8")
            report = score_agreement(conditions, published)
            self.assertTrue(report["ordered_matches_published_scores"]["passed"])
            other_run = root / "other.csv"
            other_run.write_text(
                "video_id,label_real,video_score\nv0,1,0.30\nv1,0,0.75\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs"):
                score_agreement(conditions, other_run)
            renamed = root / "renamed.csv"
            renamed.write_text(
                "video_id,label_real,video_score\nx0,1,0.25\nx1,0,0.75\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identical video IDs"):
                score_agreement(conditions, renamed)

    def test_tap_norms_flag_a_centre_dominated_kernel(self):
        import torch
        from scripts.e4_mechanism_checks import temporal_tap_norms
        state = {"tcn.0.conv.weight": torch.zeros(2, 2, 3),
                 "tcn.0.norm.weight": torch.full((2,), 0.25)}
        state["tcn.0.conv.weight"][:, :, 1] = 1.0
        blocks = temporal_tap_norms(state)
        self.assertAlmostEqual(blocks["0"]["centre_share"], 1.0)
        self.assertAlmostEqual(blocks["0"]["uniform_share"], 1 / 3)
        self.assertAlmostEqual(blocks["0"]["residual_norm_weight_abs_mean"], 0.25)
        balanced = {"tcn.1.conv.weight": torch.ones(2, 2, 3)}
        self.assertAlmostEqual(temporal_tap_norms(balanced)["1"]["centre_share"], 1 / 3)

    def test_tap_norm_report_needs_a_tcn_checkpoint(self):
        from scripts.e4_mechanism_checks import tap_norm_report
        with self.assertRaisesRegex(ValueError, "No TCN Conv1d weights"):
            tap_norm_report({}, {"model": {"temporal_head": "tcn"}})

    def test_sequence_cache_round_trips_and_checks_its_length(self):
        import torch
        from scripts.temporal_order_control import load_sequence_cache, save_sequence_cache
        sequences = [torch.randn(2, 8, 512), torch.randn(2, 8, 512)]
        metadata = [{"video_id": "v0"}, {"video_id": "v1"}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.pt"
            save_sequence_cache(path, sequences, metadata, __file__, "val")
            restored, rows, payload = load_sequence_cache(path, expected_videos=2)
            self.assertEqual(payload["split"], "val")
            self.assertEqual(payload["steps"], 8)
            self.assertEqual([row["video_id"] for row in rows], ["v0", "v1"])
            torch.testing.assert_close(restored[0], sequences[0])
            with self.assertRaisesRegex(ValueError, "expected 3"):
                load_sequence_cache(path, expected_videos=3)


    def test_main_runs_all_three_checks_from_a_cached_trunk_pass(self):
        import torch
        from unittest.mock import patch
        from scripts import e4_mechanism_checks as checks
        from scripts.temporal_order_control import cache_sequences, save_sequence_cache, score
        device = torch.device("cpu")
        model, head = self.pieces()
        config = {"seed": 42,
                  "data": {"dataset_roots": {"CelebDFv3": "unused"}, "num_workers": 0,
                           "eval_clip_chunk_size": 2, "eval_clips_per_video": 2},
                  "model": {"temporal_head": "tcn", "temporal_aggregation": "gap",
                            "temporal_steps": 8, "spatial_output_size": 4,
                            "feature_dim": 512, "deterministic_hidden_dim": 0}}
        batches = [{"clips": torch.randn(1, 2, 3, 8, 64, 64), "label": torch.tensor([label]),
                    "relative_path": ["{}.mp4".format(index)], "dataset": ["CelebDFv3"],
                    "method": ["real" if label else "fake"],
                    "target_id": ["person{}".format(index)], "source_clip": ["source"]}
                   for index, label in enumerate((1, 0, 1, 0))]
        sequences, metadata = cache_sequences(model, batches, device, 2)
        ordered = score(sequences, model, head, device, chunk=2)
        checkpoint = {"stage": "phase_a", "config": config,
                      "extractor": model.state_dict(), "head": head.state_dict()}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_sequence_cache(root / "cache.pt", sequences, metadata, "unused", "val")
            with (root / "reference.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["video_id", "label_real", "video_score"])
                for row, value in zip(metadata, ordered):
                    writer.writerow([row["video_id"], row["label_real"], value])
            with (root / "conditions.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["video_id", "label_real", "identity", "ordered", "shuffled_0"])
                for row, value in zip(metadata, ordered):
                    writer.writerow([row["video_id"], row["label_real"],
                                     row["identity"], value, value + 0.25])
            output = root / "checks.json"
            args = ["checks", "--checkpoint", str(root / "checkpoints" / "best.pt"),
                    "--order-control-scores", str(root / "conditions.csv"),
                    "--sequence-cache", str(root / "cache.pt"),
                    "--reference-scores", str(root / "reference.csv"),
                    "--expected-videos", "4", "--draws", "5", "--output", str(output)]
            with patch.object(sys, "argv", args), \
                 patch.object(checks, "load_checkpoint", return_value=checkpoint):
                self.assertEqual(checks.main(), 0)
            report = json.loads(output.read_text(encoding="utf-8"))
        self.assertAlmostEqual(
            report["score_agreement"]["conditions"]["shuffled_0"]["all"]["mae"], 0.25, places=6)
        self.assertIn("0", report["temporal_tap_norms"]["trained"])
        screen = report["position_screen"]
        self.assertTrue(screen["sequence_cache"]["reused"])
        self.assertTrue(screen["mean_branch_reference_check"]["passed"])
        self.assertIn("max", screen["within_clip"])
        # The pooled family is exploratory and must stay labelled as such.
        self.assertIn("scope", screen["pooled_over_all_positions"])
        self.assertIn("max", screen["pooled_over_all_positions"]["metrics"])
        self.assertIsNotNone(screen["dispersion"]["real"])

    def test_main_refuses_a_head_that_breaks_the_decomposition(self):
        from unittest.mock import patch
        from scripts import e4_mechanism_checks as checks
        model, head = self.pieces()
        config = {"seed": 42, "data": {},
                  "model": {"temporal_head": "tcn", "temporal_aggregation": "gap",
                            "deterministic_hidden_dim": 64}}
        checkpoint = {"stage": "phase_a", "config": config,
                      "extractor": model.state_dict(), "head": head.state_dict()}
        with tempfile.TemporaryDirectory() as directory:
            args = ["checks", "--checkpoint", "unused/best.pt",
                    "--output", str(Path(directory) / "out.json")]
            with patch.object(sys, "argv", args), \
                 patch.object(checks, "load_checkpoint", return_value=checkpoint):
                with self.assertRaisesRegex(ValueError, "single linear"):
                    checks.main()
            config["model"]["deterministic_hidden_dim"] = 0
            config["model"]["temporal_aggregation"] = "attention"
            with patch.object(sys, "argv", args), \
                 patch.object(checks, "load_checkpoint", return_value=checkpoint):
                with self.assertRaisesRegex(ValueError, "assumes GAP"):
                    checks.main()


class VramProbeTests(unittest.TestCase):
    def test_frame_size_follows_the_preprocessing_path(self):
        from vram_probe import frame_size
        face = {"data": {"frame_mode": "face", "center_crop": 256}, "model": {}}
        self.assertEqual(frame_size(face, None, 1080, 1920), (256, 256))
        # Decimation never resizes, so input_resize in the config is a red herring.
        decimate = {"data": {"frame_mode": "decimate", "decimate_step": 2},
                    "model": {"center_crop": 256}}
        self.assertEqual(frame_size(decimate, None, 1080, 1920), (540, 960))
        self.assertEqual(frame_size(decimate, 3, 1080, 1920), (360, 640))

    def test_variants_cover_the_three_rungs(self):
        from vram_probe import VARIANTS
        self.assertEqual(VARIANTS["e2"]["feature_dim"], 32 * 22 * 22)
        self.assertEqual(VARIANTS["e3"]["temporal_head"], "mean")
        self.assertEqual(VARIANTS["e4"]["temporal_head"], "tcn")


class DfdBatchProtocolTests(unittest.TestCase):
    def test_dfd_trains_with_the_same_batch_composition_as_celeb(self):
        """Accumulation does not change what BatchNorm sees, so the physical batch must match."""
        import yaml
        celeb = yaml.safe_load((ROOT / "configs/v2/phase_a_celeb.yaml").read_text(encoding="utf-8"))
        dfd = yaml.safe_load((ROOT / "configs/v2/phase_a_dfd.yaml").read_text(encoding="utf-8"))
        self.assertEqual(dfd["train"]["physical_batch_size"],
                         celeb["train"]["physical_batch_size"])
        self.assertEqual(dfd["train"].get("gradient_accumulation_steps", 1), 1)
        # Two balance groups (real, fake) and a batch of eight give 4 + 4.
        self.assertEqual(dfd["train"]["physical_batch_size"] % 2, 0)
        self.assertEqual(dfd["data"]["train_balance_keys"],
                         celeb["data"]["train_balance_keys"])

    def test_matrix_entries_do_not_quietly_restore_the_small_dfd_batch(self):
        """The matrix overrides the config, so a stale entry there would undo the above."""
        from video_bcnn.utils import apply_experiment, load_config
        matrix = str(ROOT / "configs/v2/experiment_matrix.yaml")
        base = str(ROOT / "configs/v2/phase_a_dfd.yaml")
        for experiment in ("E2_dfd", "E3", "E4_gap", "E4_attention", "E4_flatten"):
            config = apply_experiment(load_config(base), matrix, experiment)
            self.assertEqual(config["train"]["physical_batch_size"], 8, experiment)
            self.assertEqual(config["train"].get("gradient_accumulation_steps", 1), 1,
                             experiment)
        # E0 and E1 are the batch-size intervention itself and stay at one.
        for experiment in ("E0", "E1"):
            config = apply_experiment(load_config(base), matrix, experiment)
            self.assertEqual(config["train"]["physical_batch_size"], 1, experiment)


if __name__ == "__main__":
    unittest.main()
