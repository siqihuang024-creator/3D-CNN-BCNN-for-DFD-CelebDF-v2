"""Safety checks for sequential Phase-A group training."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_phase_a_groups_05_30 as runner


class GroupRunnerTests(unittest.TestCase):
    def test_group_command_uses_its_manifest_seed_and_output(self):
        command = runner._command(5)
        self.assertEqual(command[command.index("--manifest") + 1],
                         "artifacts/manifests/celeb_video_disjoint_seed42/group_05.csv")
        self.assertEqual(command[command.index("--seed") + 1], "46")
        self.assertEqual(command[command.index("--run-dir") + 1],
                         "artifacts/v2_video_disjoint/e2_group05_formal")

    def test_complete_run_is_skipped_but_partial_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(runner, "ROOT", Path(directory)):
                self.assertFalse(runner._complete(5))
                manifest, run_dir_name = runner._paths(5)
                run_dir = Path(directory) / run_dir_name
                (run_dir / "logs").mkdir(parents=True)
                (run_dir / "checkpoints").mkdir()
                (run_dir / "logs" / "history.json").write_text(
                    json.dumps([{"epoch": n} for n in range(1, 101)]),
                    encoding="utf-8")
                (run_dir / "checkpoints" / "best.pt").write_bytes(b"checkpoint")
                (run_dir / "checkpoints" / "last.pt").write_bytes(b"checkpoint")
                (run_dir / "config.json").write_text(
                    json.dumps({"seed": 46, "train": {
                        "epochs": 100, "early_stopping_enabled": False}}),
                    encoding="utf-8")
                (run_dir / "runtime.json").write_text(
                    json.dumps({"manifest": manifest,
                                "experiment": "E2_celeb"}),
                    encoding="utf-8")
                self.assertTrue(runner._complete(5))
                (run_dir / "checkpoints" / "last.pt").unlink()
                with self.assertRaisesRegex(RuntimeError, "incomplete"):
                    runner._complete(5)

    def test_failed_group_stops_before_the_next_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for group in (5, 6):
                manifest, _ = runner._paths(group)
                path = root / manifest
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("manifest", encoding="utf-8")
            with patch.object(runner, "ROOT", root), \
                    patch.object(runner, "_check_protocol"), \
                    patch.object(sys, "argv", [
                        "runner", "--start-group", "5", "--end-group", "6"]), \
                    patch.object(runner.subprocess, "run") as run:
                run.return_value.returncode = 1
                with self.assertRaisesRegex(RuntimeError, "Group 05 failed"):
                    runner.main()
                run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
