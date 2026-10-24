"""Unit tests for ViNet_S/dataset_version.py (the --dataset_version pin).

Run with: .venv/bin/python3 -m unittest tests.test_dataset_version -v
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "ViNet_S"))

import dataset_version as dv  # noqa: E402


class ParseLakefsUriTest(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(
            dv.parse_lakefs_uri("lakefs://sports-outplay-saliency-vinet/v1"),
            ("sports-outplay-saliency-vinet", "v1"),
        )

    def test_rejects_datasets_segment(self):
        with self.assertRaises(SystemExit):
            dv.parse_lakefs_uri("lakefs://datasets/sports-outplay-saliency-vinet/v1")

    def test_rejects_non_lakefs_scheme(self):
        with self.assertRaises(SystemExit):
            dv.parse_lakefs_uri("s3://sports-outplay-saliency-vinet/v1")

    def test_rejects_empty(self):
        with self.assertRaises(SystemExit):
            dv.parse_lakefs_uri("")


class Sha256FileTest(unittest.TestCase):
    def test_matches_hashlib(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "manifest.json")
            with open(path, "wb") as handle:
                handle.write(b'{"windows": []}')
            expected = hashlib.sha256(b'{"windows": []}').hexdigest()
            self.assertEqual(dv.sha256_file(path), expected)


class _FakeCompletedProcess:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class VerifyDatasetVersionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.split_root = os.path.join(self.tmp, "split")
        os.makedirs(self.split_root)
        self.manifest_path = os.path.join(self.split_root, "split_manifest.json")
        with open(self.manifest_path, "w") as handle:
            json.dump({"windows": []}, handle)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _fake_run_copying(self, src_path):
        """Build a subprocess.run stand-in that copies src_path to the
        --out argument of a `cohub datasets download` invocation."""

        def _run(cmd, check=False, capture_output=False, text=False):
            out_path = cmd[cmd.index("--out") + 1]
            shutil.copyfile(src_path, out_path)
            return _FakeCompletedProcess(returncode=0)

        return _run

    def test_matching_manifest_passes(self):
        original_run = subprocess.run
        dv.subprocess.run = self._fake_run_copying(self.manifest_path)
        try:
            repo, tag = dv.verify_dataset_version(
                "lakefs://sports-outplay-saliency-vinet/v1", self.split_root
            )
        finally:
            dv.subprocess.run = original_run
        self.assertEqual((repo, tag), ("sports-outplay-saliency-vinet", "v1"))

    def test_mismatched_manifest_exits(self):
        other = os.path.join(self.tmp, "other_manifest.json")
        with open(other, "w") as handle:
            json.dump({"windows": [{"split": "train"}]}, handle)
        original_run = subprocess.run
        dv.subprocess.run = self._fake_run_copying(other)
        try:
            with self.assertRaises(SystemExit):
                dv.verify_dataset_version(
                    "lakefs://sports-outplay-saliency-vinet/v1", self.split_root
                )
        finally:
            dv.subprocess.run = original_run

    def test_empty_dataset_version_skips_cohub(self):
        def _fail_if_called(cmd, check=False, capture_output=False, text=False):
            raise AssertionError("cohub should not be called when --dataset_version is empty")

        original_run = subprocess.run
        dv.subprocess.run = _fail_if_called
        try:
            self.assertIsNone(dv.verify_dataset_version("", self.split_root))
        finally:
            dv.subprocess.run = original_run

    def test_missing_local_manifest_exits(self):
        os.remove(self.manifest_path)
        with self.assertRaises(SystemExit):
            dv.verify_dataset_version(
                "lakefs://sports-outplay-saliency-vinet/v1", self.split_root
            )

    def test_cohub_not_on_path_exits(self):
        def _raise_not_found(cmd, check=False, capture_output=False, text=False):
            raise FileNotFoundError("no such file: cohub")

        original_run = subprocess.run
        dv.subprocess.run = _raise_not_found
        try:
            with self.assertRaises(SystemExit):
                dv.verify_dataset_version(
                    "lakefs://sports-outplay-saliency-vinet/v1", self.split_root
                )
        finally:
            dv.subprocess.run = original_run


class CheckResumeDatasetVersionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.run_dir = os.path.join(self.tmp, "run")
        os.makedirs(os.path.join(self.run_dir, "config"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_args_json(self, dataset_version):
        path = os.path.join(self.run_dir, "config", "args.json")
        with open(path, "w") as handle:
            json.dump({"dataset_version": dataset_version}, handle)

    def test_matching_uri_is_fine(self):
        self._write_args_json("lakefs://sports-outplay-saliency-vinet/v1")
        dv.check_resume_dataset_version("lakefs://sports-outplay-saliency-vinet/v1", self.run_dir)

    def test_disagreeing_uri_exits(self):
        self._write_args_json("lakefs://sports-outplay-saliency-vinet/v1")
        with self.assertRaises(SystemExit):
            dv.check_resume_dataset_version("lakefs://sports-outplay-saliency-vinet/v2", self.run_dir)

    def test_omitted_flag_on_pinned_run_exits(self):
        self._write_args_json("lakefs://sports-outplay-saliency-vinet/v1")
        with self.assertRaises(SystemExit):
            dv.check_resume_dataset_version("", self.run_dir)

    def test_missing_args_json_is_a_noop(self):
        # Older runs with no config/args.json at all should not block resume.
        dv.check_resume_dataset_version("lakefs://sports-outplay-saliency-vinet/v1", self.run_dir)

    def test_missing_stored_value_is_a_noop(self):
        # Older runs that never recorded a dataset_version should not block resume.
        self._write_args_json(None)
        dv.check_resume_dataset_version("lakefs://sports-outplay-saliency-vinet/v1", self.run_dir)


if __name__ == "__main__":
    unittest.main()
