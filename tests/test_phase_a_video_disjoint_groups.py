"""Regression checks for the relaxed, video-only split protocol."""

import unittest

from scripts.build_phase_a_video_disjoint_groups import build


class VideoDisjointGroupTests(unittest.TestCase):
    @staticmethod
    def row(number, class_name, method, source, split):
        return {"dataset": "CelebDFv3", "path": "{}/{:03d}.mp4".format(method, number),
                "class_name": class_name, "method": method, "source_clip": source,
                "split": split}

    def fixture(self):
        rows = [self.row(i, "real", "real", "source_{}".format(i % 4),
                         "train" if i < 4 else "val" if i < 6 else "test")
                for i in range(8)]
        # Every source family contains several fakes. A video-only split must be
        # allowed to separate the family without repeating an actual video.
        for method in ("A", "B"):
            for i in range(16):
                rows.append(self.row(i, "fake", method, "source_{}".format(i % 8),
                                     "train" if i < 4 else "val" if i < 10 else "test"))
        return rows

    def test_fixed_holdouts_disjoint_fake_groups_and_random_real_draws(self):
        rows = self.fixture()
        options = dict(groups=2, train_real_count=4, val_real_count=2,
                       test_real_count=2, train_fake_count=8, fake_per_group=4)
        manifests, summary = build(rows, seed=42, **options)
        self.assertEqual(summary["counts"]["unused_fake"], 20)
        self.assertGreater(summary["source_family_overlap_counts"]["train/val"]["min"], 0)
        self.assertEqual(len(manifests), 2)
        self.assertEqual(build(rows, seed=42, **options)[0], manifests)
        holdouts = []
        training_fakes = []
        for manifest in manifests:
            per_split = {split: [row for row in manifest if row["split"] == split]
                         for split in ("train", "val", "test")}
            self.assertEqual([len(per_split[s]) for s in ("train", "val", "test")],
                             [8, 4, 4])
            self.assertEqual(len({row["path"] for row in manifest}), 16)
            self.assertEqual({row["method"] for row in per_split["train"]
                              if row["class_name"] == "fake"}, {"A", "B"})
            training_fakes.extend(row["path"] for row in per_split["train"]
                                  if row["class_name"] == "fake")
            holdouts.append([(row["path"], row["split"]) for row in manifest
                             if row["split"] != "train"])
        self.assertEqual(holdouts[0], holdouts[1])
        self.assertEqual(len(set(training_fakes)), 8)
        original_split = {row["path"]: row["split"] for row in rows}
        for manifest in manifests:
            self.assertTrue(all(row["split"] == original_split[row["path"]]
                                for row in manifest))

    def test_duplicate_video_path_rejected(self):
        rows = self.fixture()
        rows.append(dict(rows[0], path=rows[0]["path"].upper()))
        with self.assertRaisesRegex(ValueError, "repeated video path"):
            build(rows, groups=2, train_real_count=4, val_real_count=2,
                  test_real_count=2, train_fake_count=8, fake_per_group=4)


if __name__ == "__main__":
    unittest.main()
