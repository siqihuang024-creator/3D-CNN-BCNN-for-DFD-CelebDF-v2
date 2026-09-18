"""Fast, data-free tests for the V2 experiment contracts."""

import sys
import unittest
import tempfile
from pathlib import Path

import numpy as np
import pyro
import torch
from pyro.infer import SVI, Trace_ELBO
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_bcnn.data import VideoClipDataset, skip_unreadable_collate
from video_bcnn.experiment import BalancedBatchSampler
from video_bcnn.features import FeatureCacheDataset
from video_bcnn.metrics import calibrate_threshold, detection_metrics
from video_bcnn.model import (DeterministicHead, Stable3DFeatureExtractor,
                              TemporalAggregator, VideoBayesianCNN,
                              freeze_extractor)
from video_bcnn.protocols import assign_protocol, protocol_audit


class ModelTests(unittest.TestCase):
    def test_e3_mean_interface_is_512(self):
        model = Stable3DFeatureExtractor(temporal_head="mean", spatial_output_size=4)
        self.assertEqual(model.feature_dim, 512)
        output = model(torch.randn(2, 3, 4, 96, 96))
        self.assertEqual(tuple(output.shape), (2, 512))

    def test_e4_tcn_preserves_512_interface(self):
        model = Stable3DFeatureExtractor(temporal_head="tcn", temporal_steps=8,
                                         temporal_aggregation="attention")
        output = model(torch.randn(2, 3, 8, 96, 96))
        self.assertEqual(tuple(output.shape), (2, 512))
        self.assertGreater(model.last_temporal_stats["output_std"], 0)

    def test_non_square_pool_for_dfd_screen(self):
        model = Stable3DFeatureExtractor(temporal_head="mean",
                                         spatial_output_size=[4, 7])
        self.assertEqual(model.feature_dim, 32 * 4 * 7)

    def test_all_aggregation_options_return_512(self):
        values = torch.randn(2, 8, 512)
        for name in TemporalAggregator.CHOICES:
            aggregator = TemporalAggregator(name, 512, 8)
            self.assertEqual(tuple(aggregator(values).shape), (2, 512), name)

    def test_aggregation_parameter_budgets_match_the_plan(self):
        counts = {name: sum(parameter.numel() for parameter in
                            TemporalAggregator(name, 512, 32).parameters())
                  for name in TemporalAggregator.CHOICES}
        self.assertEqual(counts["gap"], 0)
        self.assertEqual(counts["max"], 0)
        self.assertEqual(counts["attention"], 65793)
        self.assertEqual(counts["flatten"], 8389120)
        self.assertEqual(counts["flatten64"], 1081920)
        self.assertTrue(2100000 <= counts["cls"] <= 2200000)

    def test_phase_a_default_head_is_one_linear_layer(self):
        head = DeterministicHead(512, hidden_dim=0)
        self.assertEqual(sum(isinstance(layer, nn.Linear) for layer in head.modules()), 1)

    def test_freeze_keeps_batchnorm_in_eval(self):
        extractor = Stable3DFeatureExtractor(temporal_head="mean", spatial_output_size=4)
        freeze_extractor(extractor)
        self.assertFalse(extractor.training)
        self.assertFalse(any(parameter.requires_grad for parameter in extractor.parameters()))


class BayesianTests(unittest.TestCase):
    class Tiny(nn.Module):
        feature_dim = 8
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(1))

    def test_v2_bayesian_head_and_posterior_diagnostics(self):
        pyro.clear_param_store()
        model = VideoBayesianCNN(self.Tiny())
        svi = SVI(model.model, model.guide, pyro.optim.Adam({"lr": 1e-3}),
                  loss=Trace_ELBO())
        loss = svi.step(torch.randn(4, 8), torch.ones(4), 20)
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(tuple(model.posterior_loc_from_features(torch.randn(3, 8)).shape), (3,))
        self.assertGreater(model.diagnostics()["sigma_mean"], 0)


class ProtocolTests(unittest.TestCase):
    @staticmethod
    def rows(count=12):
        rows = []
        for index in range(count):
            source = "id{}_0".format(index)
            rows.append({"dataset": "CelebDFv3", "path": "r{}.mp4".format(index),
                         "label": 1, "class_name": "real", "target_id": "id{}".format(index),
                         "donor_id": "", "source_clip": source,
                         "method": "real/Real_WEB"})
            for fake in range(2):
                rows.append({"dataset": "CelebDFv3", "path": "f{}_{}.mp4".format(index, fake),
                             "label": 0, "class_name": "fake", "target_id": "id{}".format(index),
                             "donor_id": "id{}".format((index + 1) % count),
                             "source_clip": source, "method": "FaceSwap"})
        return rows

    def test_p05_keeps_every_row_and_family_together(self):
        source = self.rows()
        assigned = assign_protocol(source, "p05", seed=42)
        audit = protocol_audit(assigned, "p05")
        self.assertEqual(audit["rows_total"], audit["rows_active"])
        self.assertFalse(any(audit["source_family_overlap"].values()))
        for family in set(row["source_clip"] for row in assigned):
            self.assertEqual(len({row["split"] for row in assigned
                                  if row["source_clip"] == family}), 1)

    def test_p0_is_video_level_control(self):
        assigned = assign_protocol(self.rows(), "p0", seed=42)
        self.assertEqual(len(assigned), len(self.rows()))
        self.assertEqual({row["protocol"] for row in assigned}, {"p0"})


class BatchAndMetricTests(unittest.TestCase):
    def test_every_physical_batch_is_class_balanced(self):
        records = ([{"dataset": "DFD", "class_name": "real"}] * 3 +
                   [{"dataset": "DFD", "class_name": "fake"}] * 20)
        sampler = BalancedBatchSampler(records, batch_size=4, batches_per_epoch=5)
        for batch in sampler:
            classes = [records[index]["class_name"] for index in batch]
            self.assertEqual(classes.count("real"), 2)
            self.assertEqual(classes.count("fake"), 2)

    def test_metrics_report_both_ap_directions_and_lift(self):
        labels = np.asarray([1, 1, 0, 0])
        scores = np.asarray([0.1, 0.2, 0.8, 0.9])
        metrics = detection_metrics(labels, scores,
                                    calibrate_threshold(scores[labels == 1], 0.05))
        self.assertEqual(metrics["auroc"], 1.0)
        self.assertEqual(metrics["fake_average_precision"], 1.0)
        self.assertEqual(metrics["real_average_precision"], 1.0)
        self.assertEqual(metrics["fake_ap_lift"], 2.0)


class DataTests(unittest.TestCase):
    def test_t32_stride2_indices(self):
        dataset = VideoClipDataset.__new__(VideoClipDataset)
        dataset.training = False
        dataset.clip_length = 32
        dataset.train_clip_strides = (1, 2)
        dataset.eval_clip_stride = 2
        dataset.clips_per_video = 8
        dataset.deterministic_clips = False
        clips = dataset.clip_indices(300)
        self.assertEqual(len(clips), 8)
        self.assertTrue(all(len(clip) == 32 for clip in clips))
        self.assertTrue(all(b - a == 2 for a, b in zip(clips[0], clips[0][1:])))

    def test_unreadable_collate(self):
        self.assertIsNone(skip_unreadable_collate([None]))
        batch = skip_unreadable_collate([None, {"label": torch.tensor(1)}])
        self.assertEqual(batch["label"].tolist(), [1])

    def test_feature_cache_matches_manifest_keys_not_row_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.npz"
            np.savez(path, features=np.asarray([[1, 2], [3, 4]], dtype=np.float32),
                     datasets=np.asarray(["DFD", "DFD"]),
                     paths=np.asarray(["a.mp4", "b.mp4"]))
            cache = FeatureCacheDataset(path, [
                {"dataset": "DFD", "path": "b.mp4"},
                {"dataset": "DFD", "path": "a.mp4"}])
            self.assertEqual(cache[0]["feature"].tolist(), [3.0, 4.0])


if __name__ == "__main__":
    unittest.main()
