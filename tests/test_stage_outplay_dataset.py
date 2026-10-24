"""Unit tests for scripts/stage_outplay_dataset.py (the one-time uploader).

Run with: .venv/bin/python3 -m unittest tests.test_stage_outplay_dataset -v
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))

import stage_outplay_dataset as sod  # noqa: E402


def _write_window(split_root, split, sport, game, window_dir, frame_count=32, image_bytes=b"x", map_bytes=b"yy"):
    base = os.path.join(split_root, split, sport, game, os.path.basename(window_dir.rstrip("/")))
    for kind, payload in (("images", image_bytes), ("maps", map_bytes)):
        folder = os.path.join(base, kind)
        os.makedirs(folder, exist_ok=True)
        for index in range(1, frame_count + 1):
            with open(os.path.join(folder, "%04d.png" % index), "wb") as handle:
                handle.write(payload)


class ValidateSplitTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_manifest(self, windows):
        with open(os.path.join(self.tmp, "split_manifest.json"), "w") as handle:
            json.dump({"windows": windows}, handle)

    def test_counts_files_and_bytes(self):
        self._write_manifest([
            {"split": "train", "sport": "basketball", "game": "game1", "dir": "outputs/saliency_vinet/win0001"},
        ])
        _write_window(self.tmp, "train", "basketball", "game1", "win0001", image_bytes=b"x", map_bytes=b"yy")

        file_count, total_bytes = sod.validate_split(self.tmp)

        self.assertEqual(file_count, 1 + 32 + 32)
        manifest_bytes = os.path.getsize(os.path.join(self.tmp, "split_manifest.json"))
        self.assertEqual(total_bytes, manifest_bytes + 32 * 1 + 32 * 2)

    def test_missing_manifest_exits(self):
        with self.assertRaises(SystemExit):
            sod.validate_split(self.tmp)

    def test_missing_frame_exits(self):
        self._write_manifest([
            {"split": "train", "sport": "basketball", "game": "game1", "dir": "win0001"},
        ])
        _write_window(self.tmp, "train", "basketball", "game1", "win0001")
        os.remove(os.path.join(self.tmp, "train", "basketball", "game1", "win0001", "images", "0001.png"))
        with self.assertRaises(SystemExit):
            sod.validate_split(self.tmp)

    def test_empty_windows_exits(self):
        self._write_manifest([])
        with self.assertRaises(SystemExit):
            sod.validate_split(self.tmp)

    def test_window_dir_basename_strips_prefix(self):
        # The manifest's "dir" is an outputs/saliency_vinet path; only its
        # basename is an S3 key segment under split_root.
        self.assertEqual(sod._window_dir_name("outputs/saliency_vinet/win0001"), "win0001")
        self.assertEqual(sod._window_dir_name("win0001/"), "win0001")
        with self.assertRaises(ValueError):
            sod._window_dir_name("")


class ProfileNameFromEnvFileTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_reads_bracketed_profile_name_only(self):
        env_path = os.path.join(self.tmp, "fixture.env")
        with open(env_path, "w") as handle:
            handle.write("[914888343667_aws-streaming-user2]\n")
            handle.write("aws_access_key_id=NOT_A_REAL_SECRET\n")
            handle.write("aws_secret_access_key=NOT_A_REAL_SECRET\n")
            handle.write("aws_session_token=NOT_A_REAL_SECRET\n")
        self.assertEqual(
            sod._profile_name_from_env_file(env_path), "914888343667_aws-streaming-user2"
        )

    def test_missing_file_exits(self):
        with self.assertRaises(SystemExit):
            sod._profile_name_from_env_file(os.path.join(self.tmp, "missing.env"))

    def test_no_header_exits(self):
        env_path = os.path.join(self.tmp, "fixture.env")
        with open(env_path, "w") as handle:
            handle.write("aws_access_key_id=NOT_A_REAL_SECRET\n")
        with self.assertRaises(SystemExit):
            sod._profile_name_from_env_file(env_path)


if __name__ == "__main__":
    unittest.main()
