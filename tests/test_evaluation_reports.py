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


if __name__ == "__main__":
    unittest.main()
