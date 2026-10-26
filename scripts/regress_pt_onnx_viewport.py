#!/usr/bin/env python3
"""Compare PyTorch viewport wrapper against serving + debug ONNX exports.

Runs on CUDA by default (plan). Synthetic uint8 clips; saliency atol 1e-4,
bbox_conf atol 1e-5.

Used by `make regress-viewport` / `make release-onnx`.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from viewport_onnx import (  # noqa: E402
    CLIP_LEN,
    FRAME_H,
    FRAME_W,
    ViewportDebugONNXWrapper,
    load_saliency_model,
)

SALIENCY_ATOL = 1e-4
BBOX_ATOL = 1e-5
DEFAULT_NUM_CLIPS = 4


def _require_ort():
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit(
            "onnxruntime is required for regression: pip install onnxruntime-gpu "
            "or onnxruntime"
        ) from exc
    return ort


def make_synthetic_clips(num: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    clips = []
    for i in range(num):
        # Structured pattern so peaks are stable across PT/ORT.
        base = rng.integers(0, 64, size=(CLIP_LEN, 3, FRAME_H, FRAME_W), dtype=np.uint8)
        # Bright vertical band that moves with seed/index.
        x0 = (40 + 17 * i) % (FRAME_W - 40)
        base[:, :, :, x0 : x0 + 30] = 220
        clips.append(np.ascontiguousarray(base))
    return clips


def run_pt(wrapper: ViewportDebugONNXWrapper, clip_u8: np.ndarray, device: torch.device):
    t = torch.from_numpy(clip_u8).to(device)
    with torch.no_grad():
        saliency, bbox = wrapper(t)
    return (
        saliency.detach().cpu().numpy().astype(np.float32),
        bbox.detach().cpu().numpy().astype(np.float32),
    )


def run_ort_debug(session, clip_u8: np.ndarray):
    feeds = {session.get_inputs()[0].name: clip_u8}
    outs = session.run(None, feeds)
    # Expect saliency, bbox_conf
    if len(outs) != 2:
        raise RuntimeError("debug ONNX expected 2 outputs, got %d" % len(outs))
    return outs[0].astype(np.float32), outs[1].astype(np.float32).reshape(-1)


def run_ort_serving(session, clip_u8: np.ndarray):
    feeds = {session.get_inputs()[0].name: clip_u8}
    outs = session.run(None, feeds)
    if len(outs) != 1:
        raise RuntimeError("serving ONNX expected 1 output, got %d" % len(outs))
    return outs[0].astype(np.float32).reshape(-1)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--onnx", required=True, help="serving ONNX (bbox_conf only)")
    p.add_argument("--debug-onnx", required=True, help="debug ONNX (saliency + bbox_conf)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-clips", type=int, default=DEFAULT_NUM_CLIPS)
    p.add_argument("--saliency-atol", type=float, default=SALIENCY_ATOL)
    p.add_argument("--bbox-atol", type=float, default=BBOX_ATOL)
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    ort = _require_ort()

    for path in (args.checkpoint, args.onnx, args.debug_onnx):
        if not os.path.isfile(path):
            print("Not found:", path, file=sys.stderr)
            return 1

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable; refusing (plan requires CUDA regression)", file=sys.stderr)
        return 1

    backbone, _ = load_saliency_model(args.checkpoint, device)
    wrapper = ViewportDebugONNXWrapper(backbone).to(device).eval()

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess_dbg = ort.InferenceSession(args.debug_onnx, providers=providers)
    sess_srv = ort.InferenceSession(args.onnx, providers=providers)

    clips = make_synthetic_clips(args.num_clips)
    max_sal = 0.0
    max_bbox = 0.0
    max_srv = 0.0

    for i, clip in enumerate(clips):
        sal_pt, bbox_pt = run_pt(wrapper, clip, device)
        sal_ort, bbox_ort = run_ort_debug(sess_dbg, clip)
        bbox_srv = run_ort_serving(sess_srv, clip)

        d_sal = float(np.max(np.abs(sal_pt - sal_ort)))
        d_bbox = float(np.max(np.abs(bbox_pt - bbox_ort)))
        d_srv = float(np.max(np.abs(bbox_pt - bbox_srv)))
        max_sal = max(max_sal, d_sal)
        max_bbox = max(max_bbox, d_bbox)
        max_srv = max(max_srv, d_srv)
        print(
            "clip %d: saliency_maxabs=%.3e bbox_debug_maxabs=%.3e bbox_serving_maxabs=%.3e"
            % (i, d_sal, d_bbox, d_srv)
        )

    ok = True
    if max_sal > args.saliency_atol:
        print("FAIL saliency maxabs %.3e > %.3e" % (max_sal, args.saliency_atol), file=sys.stderr)
        ok = False
    if max_bbox > args.bbox_atol or max_srv > args.bbox_atol:
        print(
            "FAIL bbox_conf maxabs debug=%.3e serving=%.3e > %.3e"
            % (max_bbox, max_srv, args.bbox_atol),
            file=sys.stderr,
        )
        ok = False

    if ok:
        print(
            "PASS: saliency<=%.1e bbox<=%.1e (n=%d)"
            % (args.saliency_atol, args.bbox_atol, args.num_clips)
        )
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
