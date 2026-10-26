#!/usr/bin/env python3
"""Export ViNet-S + 9:16 viewport head to ONNX (serving + debug).

Serving graph: uint8 frames [32,3,224,384] -> bbox_conf float32[5]
Debug graph:   same input -> (saliency[224,384], bbox_conf[5])  (regression only)

Also writes a weights-only .pt of the ViNet backbone for the paired Cohub
pytorch flavor.

Used by `make export-viewport` / `make release-onnx`.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from viewport_onnx import (  # noqa: E402
    CLIP_LEN,
    FRAME_H,
    FRAME_W,
    ViewportDebugONNXWrapper,
    ViewportONNXWrapper,
    export_weights_only_state_dict,
    load_saliency_model,
)

DEFAULT_OPSET = 17


def _dummy_uint8(device: torch.device) -> torch.Tensor:
    return torch.zeros(CLIP_LEN, 3, FRAME_H, FRAME_W, dtype=torch.uint8, device=device)


def export_serving(wrapper: ViewportONNXWrapper, out_path: str, opset: int, device: torch.device) -> None:
    wrapper.eval()
    dummy = _dummy_uint8(device)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    _onnx_export(
        wrapper,
        dummy,
        out_path,
        opset=opset,
        input_names=["frames"],
        output_names=["bbox_conf"],
    )


def export_debug(wrapper: ViewportDebugONNXWrapper, out_path: str, opset: int, device: torch.device) -> None:
    wrapper.eval()
    dummy = _dummy_uint8(device)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    _onnx_export(
        wrapper,
        dummy,
        out_path,
        opset=opset,
        input_names=["frames"],
        output_names=["saliency", "bbox_conf"],
    )


def _onnx_export(model, dummy, out_path, *, opset, input_names, output_names) -> None:
    """Prefer legacy exporter: dynamo path struggles with argmax-derived coords."""
    kwargs = dict(
        opset_version=opset,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=None,
        do_constant_folding=True,
    )
    try:
        torch.onnx.export(model, dummy, out_path, dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, out_path, **kwargs)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="vinet_s_dhf1k.pt or training-bundle .pt")
    p.add_argument("--out-onnx", required=True, help="serving ONNX path")
    p.add_argument("--out-debug-onnx", required=True, help="debug dual-output ONNX path")
    p.add_argument("--out-pt", required=True, help="weights-only backbone .pt for Cohub PT flavor")
    p.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    p.add_argument("--device", default="cpu", help="cpu or cuda (export usually cpu)")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not os.path.isfile(args.checkpoint):
        print("Checkpoint not found:", args.checkpoint, file=sys.stderr)
        return 1

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to cpu", file=sys.stderr)
        device = torch.device("cpu")

    backbone, meta = load_saliency_model(args.checkpoint, device)
    serving = ViewportONNXWrapper(backbone).to(device).eval()
    debug = ViewportDebugONNXWrapper(backbone).to(device).eval()

    with torch.no_grad():
        export_serving(serving, args.out_onnx, args.opset, device)
        export_debug(debug, args.out_debug_onnx, args.opset, device)

    export_weights_only_state_dict(args.checkpoint, args.out_pt)

    print(
        "Exported serving ONNX=%s debug ONNX=%s weights PT=%s (meta=%s)"
        % (args.out_onnx, args.out_debug_onnx, args.out_pt, meta)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
