"""Unit tests for the cohub-release helper scripts (scripts/export_release_weights.py
and scripts/read_run_metadata.py), exercised with a tiny torch state dict.

Run with: .venv/bin/python3 -m unittest tests.test_cohub_release -v
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))

import export_release_weights as erw  # noqa: E402
import read_run_metadata as rrm  # noqa: E402


def _tiny_bundle(val_cc=0.5, val_sim=0.6, best_loss=0.4321, epoch=7):
    model = torch.nn.Linear(3, 3)
    optimizer = torch.optim.Adam(model.parameters())
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_loss": best_loss,
        "val_cc": val_cc,
        "val_sim": val_sim,
        "run_name": "fixture_run",
    }


class ExportWeightsOnlyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_exports_model_only(self):
        bundle = _tiny_bundle()
        checkpoint_path = os.path.join(self.tmp, "best.pt")
        torch.save(bundle, checkpoint_path)

        out_path = os.path.join(self.tmp, "release", "sports-outplay-saliency-vinet.pt")
        returned_bundle = erw.export_weights_only(checkpoint_path, out_path)

        self.assertEqual(returned_bundle["best_loss"], 0.4321)
        self.assertTrue(os.path.isfile(out_path))

        exported = torch.load(out_path, map_location="cpu")
        self.assertNotIn("model", exported)
        self.assertNotIn("optimizer", exported)
        self.assertNotIn("epoch", exported)
        self.assertEqual(set(exported.keys()), set(bundle["model"].keys()))
        for key, tensor in bundle["model"].items():
            self.assertTrue(torch.equal(exported[key], tensor))

    def test_rejects_non_bundle_checkpoint(self):
        checkpoint_path = os.path.join(self.tmp, "not_a_bundle.pt")
        torch.save(torch.randn(2, 2), checkpoint_path)
        with self.assertRaises(SystemExit):
            erw.export_weights_only(checkpoint_path, os.path.join(self.tmp, "out.pt"))

    def test_rejects_missing_checkpoint(self):
        with self.assertRaises(SystemExit):
            erw.export_weights_only(os.path.join(self.tmp, "missing.pt"), os.path.join(self.tmp, "out.pt"))

    def test_rejects_bundle_without_model_key(self):
        checkpoint_path = os.path.join(self.tmp, "bad_bundle.pt")
        torch.save({"optimizer": {}, "epoch": 1}, checkpoint_path)
        with self.assertRaises(SystemExit):
            erw.export_weights_only(checkpoint_path, os.path.join(self.tmp, "out.pt"))


class ReadRunMetadataTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.run_dir = os.path.join(self.tmp, "run")
        os.makedirs(os.path.join(self.run_dir, "config"))
        os.makedirs(os.path.join(self.run_dir, "checkpoints"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_args_json(self, payload):
        path = os.path.join(self.run_dir, "config", "args.json")
        with open(path, "w") as handle:
            json.dump(payload, handle)

    def _write_checkpoint(self, **overrides):
        bundle = _tiny_bundle(**overrides)
        torch.save(bundle, os.path.join(self.run_dir, "checkpoints", "best.pt"))
        return bundle

    def test_reads_dataset_version(self):
        self._write_args_json({"dataset_version": "lakefs://sports-outplay-saliency-vinet/v1"})
        self.assertEqual(
            rrm.read_dataset_version(self.run_dir), "lakefs://sports-outplay-saliency-vinet/v1"
        )

    def test_missing_dataset_version_exits(self):
        self._write_args_json({"dataset": "outplay"})
        with self.assertRaises(SystemExit):
            rrm.read_dataset_version(self.run_dir)

    def test_missing_args_json_exits(self):
        with self.assertRaises(SystemExit):
            rrm.read_dataset_version(self.run_dir)

    def test_reads_val_metrics_from_checkpoint(self):
        self._write_checkpoint(val_cc=0.55, val_sim=0.66, best_loss=0.1234)
        self.assertEqual(rrm.read_val_metric(self.run_dir, "val_loss"), repr(0.1234))
        self.assertEqual(rrm.read_val_metric(self.run_dir, "val_cc"), repr(0.55))
        self.assertEqual(rrm.read_val_metric(self.run_dir, "val_sim"), repr(0.66))

    def test_val_metric_empty_when_absent(self):
        bundle = _tiny_bundle()
        del bundle["val_cc"]
        del bundle["val_sim"]
        torch.save(bundle, os.path.join(self.run_dir, "checkpoints", "best.pt"))
        self.assertEqual(rrm.read_val_metric(self.run_dir, "val_cc"), "")
        self.assertEqual(rrm.read_val_metric(self.run_dir, "val_sim"), "")


if __name__ == "__main__":
    unittest.main()
