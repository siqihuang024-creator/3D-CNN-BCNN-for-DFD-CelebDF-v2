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
