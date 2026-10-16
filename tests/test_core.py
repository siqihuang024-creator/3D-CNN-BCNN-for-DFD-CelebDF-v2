"""Fast, data-free tests for the V2 experiment contracts."""

import copy
import sys
import unittest
import tempfile
from pathlib import Path

import numpy as np
import pyro
import torch
from pyro.infer import SVI, Trace_ELBO
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_bcnn.data import (MissingFaceCacheError, ShortVideoError, VideoClipDataset,
                             preflight_face_cache, skip_unreadable_collate)
from video_bcnn.evaluation import score_bayesian_cached
from video_bcnn.experiment import (BalancedBatchSampler, GroupBalancedEpochSampler,
                                   PoolPassBatchSampler)
from video_bcnn.features import FeatureCacheDataset
from video_bcnn.metrics import calibrate_threshold, detection_metrics
from video_bcnn.model import (DeterministicHead, InterleavedConv3d,
                              MC3FeatureExtractor, ResidualTemporalBlock,
                              Stable3DFeatureExtractor, TemporalAggregator,
                              VideoBayesianCNN, freeze_extractor,
                              standard_to_grouped_weight)
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

    @staticmethod
    def _interleaved_from(standard, stride):
        layer = InterleavedConv3d(standard.in_channels, standard.out_channels,
                                  standard.kernel_size, stride,
                                  standard.padding[0]).double()
        with torch.no_grad():
            layer.conv.weight.copy_(standard_to_grouped_weight(standard.weight, stride))
            layer.conv.bias.copy_(standard.bias)
        return layer

    def test_interleave_at_channel_stride_is_the_ordinary_conv(self):
        torch.manual_seed(0)
        standard = nn.Conv3d(4, 5, (3, 3, 3), padding=(1, 0, 0)).double()
        values = torch.randn(2, 4, 6, 7, 7, dtype=torch.float64)
        layer = self._interleaved_from(standard, 4)
        self.assertTrue(torch.allclose(layer(values), standard(values), atol=1e-12))

    def test_interleave_matches_literal_flattened_convolution(self):
        # The definition, written out: lay the channels of each position in
        # order, then slide a span-deep single-channel kernel `stride` planes.
        torch.manual_seed(0)
        standard = nn.Conv3d(4, 5, (3, 3, 3), padding=(1, 0, 0)).double()
        values = torch.randn(2, 4, 6, 7, 7, dtype=torch.float64)
        flat = values.permute(0, 2, 1, 3, 4).reshape(2, 1, 24, 7, 7)
        kernel = standard.weight.permute(0, 2, 1, 3, 4).reshape(5, 1, 12, 3, 3)
        for stride in (1, 2, 4):
            expected = F.conv3d(flat, kernel, standard.bias,
                                stride=(stride, 1, 1), padding=(4, 0, 0))
            layer = self._interleaved_from(standard, stride)
            output = layer(values)
            self.assertEqual(output.shape[2], layer.output_depth(6))
            self.assertTrue(torch.allclose(output, expected, atol=1e-12), stride)

    def test_e2_interleave_depths_and_parameter_count(self):
        self.assertEqual(InterleavedConv3d(16, 24, (3, 5, 5), 1, 1).output_depth(32), 497)
        self.assertEqual(InterleavedConv3d(24, 32, (3, 5, 5), 4, 1).output_depth(497), 2977)
        self.assertEqual(InterleavedConv3d(16, 24, (3, 5, 5), 2, 1).output_depth(32), 249)
        self.assertEqual(InterleavedConv3d(24, 32, (3, 5, 5), 8, 1).output_depth(249), 745)
        count = lambda model: sum(p.numel() for p in model.parameters())
        e2 = Stable3DFeatureExtractor(temporal_head="mean", spatial_output_size=22)
        interleaved = Stable3DFeatureExtractor(temporal_head="mean", spatial_output_size=22,
                                               interleave_strides=[1, 4])
        self.assertEqual(count(e2), 90280)
        self.assertEqual(count(interleaved), count(e2))
        with self.assertRaises(ValueError):
            Stable3DFeatureExtractor(temporal_head="tcn", interleave_strides=[1, 4])

    def test_full_e2_matches_interleaved_at_whole_position_stride(self):
        torch.manual_seed(3)
        ordinary = Stable3DFeatureExtractor(
            temporal_head="mean", spatial_output_size=2,
            conv_channels=(4, 6, 8)).double().eval()
        interleaved = Stable3DFeatureExtractor(
            temporal_head="mean", spatial_output_size=2,
            conv_channels=(4, 6, 8), interleave_strides=[4, 6]).double().eval()
        ordinary_state = ordinary.state_dict()
        transferred = interleaved.state_dict()
        for name in transferred:
            if name.startswith("conv2.conv.") or name.startswith("conv3.conv."):
                layer = name.split(".")[0]
                source = ordinary_state[layer + "." + name.split(".")[-1]]
                if name.endswith("weight"):
                    stride = 4 if layer == "conv2" else 6
                    source = standard_to_grouped_weight(source, stride)
                transferred[name] = source
            else:
                transferred[name] = ordinary_state[name]
        interleaved.load_state_dict(transferred)
        clips = torch.randn(2, 3, 4, 64, 64, dtype=torch.float64)
        with torch.no_grad():
            self.assertTrue(torch.allclose(ordinary(clips), interleaved(clips),
                                           atol=1e-10, rtol=1e-10))

    def test_interleaved_initialization_preserves_e2_weights_and_head_rng(self):
        def make_model(strides):
            extractor = Stable3DFeatureExtractor(
                temporal_head="mean", spatial_output_size=2,
                conv_channels=(4, 6, 8), interleave_strides=strides)
            return extractor, DeterministicHead(extractor.feature_dim)

        torch.manual_seed(42)
        ordinary, ordinary_head = make_model(None)
        torch.manual_seed(42)
        interleaved, interleaved_head = make_model([2, 3])
        self.assertTrue(torch.equal(ordinary.conv1.weight,
                                    interleaved.conv1.weight))
        for layer, stride in (("conv2", 2), ("conv3", 3)):
            reference = getattr(ordinary, layer)
            changed = getattr(interleaved, layer).conv
            self.assertTrue(torch.equal(changed.weight,
                standard_to_grouped_weight(reference.weight, stride)))
            self.assertTrue(torch.equal(changed.bias, reference.bias))
        self.assertTrue(torch.equal(ordinary_head.out.weight,
                                    interleaved_head.out.weight))

    def test_chunked_recompute_matches_whole_batch_and_bn_steps_once(self):
        torch.manual_seed(0)
        whole = Stable3DFeatureExtractor(conv_channels=(4, 8, 8), temporal_head="mean",
                                         spatial_output_size=2,
                                         interleave_strides=[1, 4]).double()
        chunked = copy.deepcopy(whole)
        chunked.interleave_chunk_size = 1
        clips = torch.randn(3, 3, 4, 64, 64, dtype=torch.float64)
        outputs = []
        for model in (whole, chunked):
            model.train()
            output = model(clips)
            output.square().sum().backward()
            outputs.append(output)
        self.assertTrue(torch.allclose(outputs[0], outputs[1], atol=1e-10))
        for (name, a), (_, b) in zip(whole.named_parameters(), chunked.named_parameters()):
            self.assertTrue(torch.allclose(a.grad, b.grad, atol=1e-10), name)
        for (name, a), (_, b) in zip(whole.named_buffers(), chunked.named_buffers()):
            if a.is_floating_point():
                self.assertTrue(torch.allclose(a, b, atol=1e-12), name)
            else:  # num_batches_tracked: one BN update per step, not two
                self.assertTrue(torch.equal(a, b), name)
        self.assertEqual(int(chunked.norm3.num_batches_tracked), 1)

    def test_non_square_pool_for_dfd_screen(self):
        model = Stable3DFeatureExtractor(temporal_head="mean",
                                         spatial_output_size=[4, 7])
        self.assertEqual(model.feature_dim, 32 * 4 * 7)

    def test_final_max_pool_does_not_change_stage_pooling(self):
        model = Stable3DFeatureExtractor(temporal_head="mean",
                                         stage_pool_type="avg",
                                         final_pool_type="max")
        self.assertIsInstance(model.pool1, nn.AvgPool3d)
        self.assertEqual(model.final_pool_type, "max")

    def test_zero_residual_branch_is_identity(self):
        block = ResidualTemporalBlock(8).eval()
        nn.init.zeros_(block.conv.weight)
        values = torch.randn(2, 8, 7)
        self.assertTrue(torch.equal(block(values), values))

    def test_e7_tcn_parameter_budget_and_e7b_control(self):
        e7 = MC3FeatureExtractor(pretrained=False, temporal_head="tcn")
        e7b = MC3FeatureExtractor(pretrained=False, temporal_head="none")
        self.assertEqual(sum(p.numel() for p in e7.parameters()), 13065152)
        self.assertFalse(hasattr(e7b, "tcn"))

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

    def test_nonlinear_head_runs_before_clip_averaging(self):
        class Extractor(nn.Module):
            feature_dim = 1
            def __init__(self):
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros(1))

        class Model:
            feature_extractor = Extractor()
            @staticmethod
            def posterior_loc_from_features(features):
                return features[:, 0].square()

        cached = {
            "features": torch.tensor([[-1.0], [3.0]]),
            "offsets": [(0, 2)], "labels": [1], "paths": ["a.mp4"],
            "relative_paths": ["a.mp4"], "datasets": ["DFD"],
            "methods": ["real"], "target_ids": ["id"], "donor_ids": [""],
            "source_clips": ["id"], "skipped_paths": [],
        }
        result = score_bayesian_cached(Model(), cached, torch.device("cpu"),
                                       mc_samples=0)
        self.assertEqual(result["means"].tolist(), [5.0])
        self.assertEqual(result["scores"].tolist(), [-5.0])


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

    def test_phase_a_epoch_budget_is_1000_clips_per_class(self):
        records = ([{"dataset": "DFD", "class_name": "real"}] * 3 +
                   [{"dataset": "DFD", "class_name": "fake"}] * 20)
        balanced = BalancedBatchSampler(records, batch_size=8,
                                        batches_per_epoch=250)
        baseline = GroupBalancedEpochSampler(
            records, seed=42, samples_per_group=1000,
            group_keys=("dataset", "class_name"))
        self.assertEqual(len(balanced) * 8, 2000)
        self.assertEqual(len(baseline), 2000)

    def test_pool_pass_covers_every_video_with_both_classes_per_batch(self):
        from collections import Counter
        records = [{"class_name": "real"}] * 622 + [{"class_name": "fake"}] * 1244
        sampler = PoolPassBatchSampler(records, batch_size=8, seed=42)
        mixes = set()
        for epoch in (1, 2, 3):
            sampler.set_epoch(epoch)
            batches = list(sampler)
            self.assertEqual(len(batches), 234)
            self.assertTrue(all(len(batch) == 8 and len(set(batch)) == 8 for batch in batches))
            uses = Counter(index for batch in batches for index in batch)
            self.assertEqual(set(uses), set(range(len(records))))
            self.assertEqual(sum(count - 1 for count in uses.values()), 6)
            repeated = [index for index, count in uses.items() if count == 2]
            self.assertEqual(sum(index < 622 for index in repeated), 2)
            self.assertEqual(sum(index >= 622 for index in repeated), 4)
            real_counts = [sum(index < 622 for index in batch) for batch in batches]
            self.assertTrue(all(1 <= count <= 7 for count in real_counts))
            self.assertEqual(sum(real_counts), 624)
            self.assertEqual(sampler.last_real_counts, real_counts)
            mixes.update(real_counts)
        self.assertGreater(len(mixes), 3)
        self.assertEqual(batches, list(sampler))
        sampler.set_epoch(4)
        self.assertNotEqual(batches, list(sampler))

    def test_pool_pass_repairs_a_batch_missing_a_class(self):
        records = [{"class_name": "real"}] * 3 + [{"class_name": "fake"}] * 21
        for seed in range(50):
            batches = list(PoolPassBatchSampler(records, batch_size=8, seed=seed))
            self.assertEqual(sorted(index for batch in batches for index in batch),
                             list(range(24)))
            self.assertTrue(all(sum(index < 3 for index in batch) == 1 for batch in batches))
        with self.assertRaises(ValueError):
            PoolPassBatchSampler(records[:2] + records[3:], batch_size=8)

    def test_metrics_report_both_ap_directions_and_lift(self):
        labels = np.asarray([1, 1, 0, 0])
        scores = np.asarray([0.1, 0.2, 0.8, 0.9])
        metrics = detection_metrics(labels, scores,
                                    calibrate_threshold(scores[labels == 1], 0.05))
        self.assertEqual(metrics["auroc"], 1.0)
        self.assertEqual(metrics["fake_average_precision"], 1.0)
        self.assertEqual(metrics["real_average_precision"], 1.0)
        self.assertEqual(metrics["macro_average_precision_real_fake"], 1.0)
        self.assertEqual(metrics["fake_ap_lift"], 2.0)
        self.assertEqual(metrics["fake_ap_gain"], 0.5)


class DataTests(unittest.TestCase):
    def test_revised_phase_a_uses_contiguous_32_frame_clips(self):
        import yaml
        config = yaml.safe_load((Path(__file__).resolve().parents[1] /
            "configs/v2/phase_a_celeb_revised.yaml").read_text(encoding="utf-8"))
        self.assertEqual(config["data"]["train_clip_strides"], [1])
        self.assertEqual(config["data"]["horizontal_flip_probability"], 0.0)
        self.assertEqual(config["data"]["eval_clip_stride"], 1)
        self.assertEqual(config["data"]["selection_clips_per_video"],
                         config["data"]["eval_clips_per_video"])
        self.assertTrue(config["data"]["require_full_clip"])
        dataset = VideoClipDataset.__new__(VideoClipDataset)
        dataset.clip_length = 32
        dataset.require_full_clip = True
        dataset.train_clip_strides = (1,)
        dataset.eval_clip_stride = 1
        dataset.clips_per_video = 8
        dataset.deterministic_clips = False
        for training in (True, False):
            dataset.training = training
            clips = dataset.clip_indices(100)
            self.assertTrue(all(len(clip) == 32 and
                all(b - a == 1 for a, b in zip(clip, clip[1:])) for clip in clips))
        with self.assertRaises(ShortVideoError):
            dataset.clip_indices(31)

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

    def test_missing_face_cache_is_fatal_not_an_unreadable_skip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "videos"
            cache = Path(directory) / "cache"
            dataset = VideoClipDataset.__new__(VideoClipDataset)
            dataset.records = [{"dataset": "DFD", "path": "missing.mp4"}]
            dataset.dataset_roots = {"DFD": root}
            dataset.face_box_cache = cache
            dataset.detection_store = {}
            dataset.unreadable = set()
            dataset._read_clips = lambda path: dataset._cached_detections(path)
            with self.assertRaises(MissingFaceCacheError):
                dataset[0]

    def test_face_cache_preflight_lists_missing_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {"data": {"frame_mode": "face",
                                "face_box_cache": directory}}
            rows = [{"dataset": "DFD", "path": "a.mp4"}]
            with self.assertRaises(MissingFaceCacheError):
                preflight_face_cache(rows, config)

    def test_strict_clip_preflight_rejects_short_cached_video(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "DFD" / "short.mp4.npz"
            cache.parent.mkdir(parents=True)
            np.savez(cache, frame_count=np.asarray(1))
            config = {"data": {"frame_mode": "face", "face_box_cache": directory,
                               "clip_length": 32, "require_full_clip": True}}
            with self.assertRaisesRegex(ShortVideoError, "short.mp4"):
                preflight_face_cache([{"dataset": "DFD", "path": "short.mp4"}], config)

    def test_strict_decoder_does_not_repeat_a_failed_frame(self):
        class TruncatedCapture:
            def __init__(self):
                self.position = 0

            def set(self, _property, position):
                self.position = int(position)

            def read(self):
                self.position += 1
                return ((True, np.zeros((2, 2, 3), dtype=np.uint8))
                        if self.position <= 1 else (False, None))

            def grab(self):
                if self.position >= 1:
                    return False
                self.position += 1
                return True

            def retrieve(self):
                return True, np.zeros((2, 2, 3), dtype=np.uint8)

        indices = [[0, 1, 2]]
        self.assertIsNone(VideoClipDataset._decode_clips(
            TruncatedCapture(), "truncated.mp4", indices,
            require_full_clip=True))
        with self.assertRaises(ShortVideoError):
            VideoClipDataset._decode_clips_sequential(
                TruncatedCapture(), "truncated.mp4", indices,
                require_full_clip=True)

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

    def test_feature_cache_keeps_all_sixteen_clips_per_video(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.npz"
            np.savez(path, features=np.arange(32, dtype=np.float32).reshape(16, 2),
                     datasets=np.asarray(["DFD"] * 16),
                     paths=np.asarray(["a.mp4"] * 16),
                     clip_indices=np.arange(16))
            cache = FeatureCacheDataset(
                path, [{"dataset": "DFD", "path": "a.mp4"}])
            self.assertEqual(len(cache), 16)
            self.assertEqual(cache[15]["clip_index"].item(), 15)


class FinalReviewTests(unittest.TestCase):
    def test_json_safe_handles_numpy_bool_and_arrays(self):
        import json
        import numpy as np
        from video_bcnn.reporting import json_safe
        value = json_safe({"wins": [np.float64(0.7) > np.float64(0.6)],
                           "matrix": np.arange(3), "nan": float("nan")})
        self.assertEqual(json.loads(json.dumps(value, allow_nan=False)),
                         {"wins": [True], "matrix": [0, 1, 2], "nan": None})

    def test_pool_screen_never_defaults_an_incomputable_probe_to_avg(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from pool_screen import recommend_final_pool
        self.assertEqual(recommend_final_pool(float("nan"), [0.01]), "undetermined")
        self.assertEqual(recommend_final_pool(0.05, [float("nan")]), "undetermined")
        self.assertEqual(recommend_final_pool(0.05, [0.01, 0.02]), "max")
        self.assertEqual(recommend_final_pool(0.01, [0.01, 0.02]), "avg")


if __name__ == "__main__":
    unittest.main()
