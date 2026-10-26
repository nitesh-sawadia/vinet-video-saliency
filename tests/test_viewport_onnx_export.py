"""Unit tests for the 9:16 viewport head and schema contracts.

Run with: .venv/bin/python3 -m unittest tests.test_viewport_onnx_export -v
"""

from __future__ import annotations

import json
import os
import sys
import unittest

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))

import viewport_onnx as vo  # noqa: E402


class NineSixteenHeadTest(unittest.TestCase):
    def test_dirac_peak_horizontal(self):
        # Single bright column at x=100 should place viewport top-left so
        # the ones-kernel mass is maximized; with full-height viewport the
        # only free axis is x (Hout=1).
        saliency = torch.zeros(vo.FRAME_H, vo.FRAME_W)
        peak_x = 100
        saliency[:, peak_x] = 1.0

        bbox = vo.bbox_conf_from_saliency(saliency)
        self.assertEqual(tuple(bbox.shape), (5,))

        # Score map Wout = 384 - 126 + 1 = 259; argmax at the x that covers peak_x.
        # Best top-left x is clamp(peak_x - (vw-1), 0, wout-1) for a single column,
        # i.e. the window that still contains peak_x with max overlap (=1).
        # Any window containing column 100 has mass 224; leftmost such x is
        # max(0, 100-125)=0? Wait: window [x, x+126) contains 100 iff x <= 100 <= x+125
        # so x in [100-125, 100] ∩ [0, 258] = [0, 100]. All have same mass.
        # argmax returns the first maximum -> x=0.
        x1, y1, x2, y2, conf = [float(v) for v in bbox]
        self.assertAlmostEqual(y1, 0.0, places=5)
        self.assertAlmostEqual(y2, 1.0, places=5)
        self.assertAlmostEqual(x2 - x1, vo.VIEWPORT_W / vo.FRAME_W, places=5)
        self.assertGreater(conf, 0.0)
        self.assertAlmostEqual(x1, 0.0, places=5)

    def test_unique_peak_forces_x(self):
        # Put mass only in a block that only one window can cover fully.
        saliency = torch.zeros(vo.FRAME_H, vo.FRAME_W)
        # Fill columns 50..175 (126 cols) with 1s — exact viewport width.
        x0 = 50
        saliency[:, x0 : x0 + vo.VIEWPORT_W] = 1.0
        bbox = vo.bbox_conf_from_saliency(saliency)
        x1, y1, x2, y2, conf = [float(v) for v in bbox]
        self.assertAlmostEqual(x1, x0 / vo.FRAME_W, places=5)
        self.assertAlmostEqual(x2, (x0 + vo.VIEWPORT_W) / vo.FRAME_W, places=5)
        self.assertAlmostEqual(y1, 0.0, places=5)
        self.assertAlmostEqual(y2, 1.0, places=5)
        # Mean activation = 1.0 inside the window.
        self.assertAlmostEqual(conf, 1.0, places=4)

    def test_onnx_head_matches_python(self):
        saliency = torch.zeros(vo.FRAME_H, vo.FRAME_W)
        saliency[:, 80:80 + vo.VIEWPORT_W] = 0.5
        head = vo._NineSixteenHead()
        with torch.no_grad():
            a = vo.bbox_conf_from_saliency(saliency)
            b = head(saliency)
        self.assertTrue(torch.allclose(a, b, atol=1e-6))


class SchemaContractTest(unittest.TestCase):
    def test_input_schema(self):
        path = os.path.join(_REPO_ROOT, "schemas", "viewport_input_schema.json")
        with open(path) as handle:
            schema = json.load(handle)
        self.assertEqual(schema["x-name"], "frames")
        self.assertEqual(schema["x-dtype"], "uint8")
        self.assertEqual(schema["x-shape"], [32, 3, 224, 384])

    def test_output_schema(self):
        path = os.path.join(_REPO_ROOT, "schemas", "viewport_output_schema.json")
        with open(path) as handle:
            schema = json.load(handle)
        self.assertEqual(schema["x-name"], "bbox_conf")
        self.assertEqual(schema["x-dtype"], "float32")
        self.assertEqual(schema["x-shape"], [5])
        self.assertIn("x1, y1, x2, y2, conf", schema["x-layout"])

    def test_inference_contract_exists(self):
        path = os.path.join(_REPO_ROOT, "docs", "INFERENCE_CONTRACT.md")
        self.assertTrue(os.path.isfile(path))
        with open(path) as handle:
            text = handle.read()
        self.assertIn("bbox_conf", text)
        self.assertIn("frames", text)


if __name__ == "__main__":
    unittest.main()
