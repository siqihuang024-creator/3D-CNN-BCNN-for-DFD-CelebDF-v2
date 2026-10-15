"""Checks that the decode audit counts frames, not only ffmpeg errors."""

import sys
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import scan_decode_health


class DecodeHealthTests(unittest.TestCase):
    def test_probe_reads_final_decoded_frame_count(self):
        completed = CompletedProcess(
            args=[], returncode=0,
            stdout=b"frame=1\nprogress=continue\nframe=32\nprogress=end\n",
            stderr=b"")
        with patch.object(scan_decode_health.subprocess, "run",
                          return_value=completed):
            verdict, messages, first, _, frames = scan_decode_health.probe(
                Path("sample.mp4"), timeout=10)
        self.assertEqual((verdict, messages, first, frames), ("ok", 0, "", 32))

    def test_nonzero_ffmpeg_exit_is_corrupt_even_without_error_text(self):
        completed = CompletedProcess(args=[], returncode=1, stdout=b"frame=1\n",
                                     stderr=b"")
        with patch.object(scan_decode_health.subprocess, "run",
                          return_value=completed):
            verdict, _, _, _, frames = scan_decode_health.probe(
                Path("sample.mp4"), timeout=10)
        self.assertEqual((verdict, frames), ("corrupt", 1))


if __name__ == "__main__":
    unittest.main()
