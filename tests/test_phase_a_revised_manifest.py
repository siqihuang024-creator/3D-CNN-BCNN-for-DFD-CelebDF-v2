"""Checks for the separate revised Phase-A data protocol."""

import unittest

from scripts.build_phase_a_revised_manifest import build


class RevisedManifestTests(unittest.TestCase):
    def test_training_pool_and_merged_validation_remain_video_disjoint(self):
        rows = []
        for number in range(4):
            source = "train_{}".format(number)
            rows.append(self.row(source, "train", "real"))
            for method in ("A", "B", "C"):
                rows.append(self.row(source, "train", "fake", method))
        for split in ("val", "test"):
            rows.append(self.row(split, split, "real"))
            rows.append(self.row(split, split, "fake", "A"))

        result, summary = build(rows, "CelebDFv3", fake_per_real=2, seed=42)
        self.assertEqual(summary["counts"], {
            "train:fake": 8, "train:real": 4, "val:fake": 2, "val:real": 2})
        self.assertEqual({row["split"] for row in result}, {"train", "val"})
        self.assertEqual(len({row["path"] for row in result}), len(result))
        self.assertEqual(summary["source_family_overlap"], 0)
        self.assertEqual(len([row for row in result if row["original_split"] == "test"]), 2)
        self.assertEqual({row["split"] for row in rows}, {"train", "val", "test"})
        self.assertEqual(result, build(rows, "CelebDFv3", fake_per_real=2, seed=42)[0])

    def test_excluded_short_real_is_recorded_and_fake_quota_recomputed(self):
        rows = []
        for number in range(4):
            source = "train_{}".format(number)
            rows.append(self.row(source, "train", "real"))
            for method in ("A", "B", "C"):
                rows.append(self.row(source, "train", "fake", method))
        for split in ("val", "test"):
            rows.append(self.row(split, split, "real"))
            rows.append(self.row(split, split, "fake", "A"))

        excluded = "train_0/real.mp4"
        result, summary = build(rows, "CelebDFv3", fake_per_real=2, seed=42,
                                excluded_train_real_paths=[excluded])
        self.assertEqual(summary["counts"], {
            "train:fake": 6, "train:real": 3, "val:fake": 2, "val:real": 2})
        self.assertEqual(summary["excluded_train_real_paths"], [excluded])
        self.assertNotIn(excluded, {row["path"] for row in result})
        self.assertEqual(summary["source_family_overlap"], 0)
        with self.assertRaisesRegex(ValueError, "not training real videos"):
            build(rows, "CelebDFv3", fake_per_real=2, seed=42,
                  excluded_train_real_paths=["val/real.mp4"])

    @staticmethod
    def row(source, split, class_name, method="real"):
        return {"dataset": "CelebDFv3", "path": "{}/{}.mp4".format(source, method),
                "split": split, "class_name": class_name, "source_clip": source,
                "method": method}


if __name__ == "__main__":
    unittest.main()
