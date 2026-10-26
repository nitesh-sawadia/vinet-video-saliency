"""Shared ViNet-S viewport wrapper: uint8 clip -> normalized [x1,y1,x2,y2,conf].

Used by export_viewport_onnx.py, regress_pt_onnx_viewport.py, and unit tests.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

FRAME_H = 224
FRAME_W = 384
CLIP_LEN = 32
VIEWPORT_H = 224
VIEWPORT_W = 126
VIEWPORT_AREA = VIEWPORT_H * VIEWPORT_W

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _repo_vinet_s() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ViNet_S"))


def ensure_vinet_path() -> None:
    root = _repo_vinet_s()
    if root not in sys.path:
        sys.path.insert(0, root)


def load_saliency_model(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, dict]:
    """Build VideoSaliencyModel and load weights (bundle or plain state dict)."""
    ensure_vinet_path()
    from ViNet_S_model import VideoSaliencyModel

    model = VideoSaliencyModel(
        use_upsample=True,
        num_hier=3,
        num_clips=CLIP_LEN,
        grouped_conv=True,
        root_grouping=True,
        depth=False,
        efficientnet=False,
        BiCubic=True,
    )
    try:
        bundle = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        bundle = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(bundle, dict) and "model" in bundle:
        state = bundle["model"]
        meta = {
            "epoch": bundle.get("epoch"),
            "best_loss": bundle.get("best_loss"),
            "val_cc": bundle.get("val_cc"),
            "val_sim": bundle.get("val_sim"),
        }
    elif isinstance(bundle, dict) and bundle and all(torch.is_tensor(v) for v in bundle.values()):
        state = bundle
        meta = {}
    else:
        raise SystemExit("Unsupported checkpoint format: %s" % checkpoint_path)

    cleaned = {}
    for key, value in state.items():
        if isinstance(key, str) and key.startswith("module."):
            cleaned[key[len("module.") :]] = value
        else:
            cleaned[key] = value
    model.load_state_dict(cleaned, strict=True)
    model.to(device)
    model.eval()
    return model, meta


def squeeze_saliency(pred: torch.Tensor) -> torch.Tensor:
    """Model output -> (H, W) for a single unbatched forward."""
    if pred.dim() == 4 and pred.shape[0] == 1 and pred.shape[1] == 1:
        return pred[0, 0]
    if pred.dim() == 3 and pred.shape[0] == 1:
        return pred[0]
    if pred.dim() == 2:
        return pred
    raise RuntimeError("Unexpected saliency shape %s" % (tuple(pred.shape),))


def bbox_conf_from_saliency(saliency: torch.Tensor) -> torch.Tensor:
    """Last-frame map (H,W) -> float32[5] normalized [x1,y1,x2,y2,conf].

    Uses 9:16 fit-inside ones-kernel box-sum + argmax (notebook convolution_scores).
    """
    if saliency.dim() != 2:
        raise ValueError("expected saliency (H, W), got %s" % (tuple(saliency.shape),))
    if saliency.shape != (FRAME_H, FRAME_W):
        raise ValueError("expected saliency (%d, %d), got %s" % (FRAME_H, FRAME_W, tuple(saliency.shape)))

    maps = F.relu(saliency.float()).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    kernel = torch.ones(
        (1, 1, VIEWPORT_H, VIEWPORT_W),
        dtype=maps.dtype,
        device=maps.device,
    )
    scores = F.conv2d(maps, kernel).squeeze(0).squeeze(0)  # [Hout, Wout]
    flat = scores.reshape(-1)
    peak_idx = int(torch.argmax(flat).item())
    peak_mass = flat[peak_idx]
    hout, wout = scores.shape
    y = peak_idx // wout
    x = peak_idx % wout

    x1 = float(x) / float(FRAME_W)
    y1 = float(y) / float(FRAME_H)
    x2 = float(x + VIEWPORT_W) / float(FRAME_W)
    y2 = float(y + VIEWPORT_H) / float(FRAME_H)
    conf = peak_mass / float(VIEWPORT_AREA)
    return torch.stack(
        [
            saliency.new_tensor(x1),
            saliency.new_tensor(y1),
            saliency.new_tensor(x2),
            saliency.new_tensor(y2),
            conf.to(dtype=saliency.dtype if saliency.is_floating_point() else torch.float32),
        ]
    ).float()


class _NineSixteenHead(nn.Module):
    """ONNX-friendly 9:16 box-sum head producing [x1,y1,x2,y2,conf]."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "kernel",
            torch.ones(1, 1, VIEWPORT_H, VIEWPORT_W),
            persistent=False,
        )
        self.register_buffer(
            "inv_frame",
            torch.tensor([1.0 / FRAME_W, 1.0 / FRAME_H, 1.0 / FRAME_W, 1.0 / FRAME_H], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "size_px",
            torch.tensor([VIEWPORT_W, VIEWPORT_H], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "inv_area",
            torch.tensor(1.0 / VIEWPORT_AREA, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, saliency_hw: torch.Tensor) -> torch.Tensor:
        # saliency_hw: [H, W]
        maps = F.relu(saliency_hw).unsqueeze(0).unsqueeze(0)
        scores = F.conv2d(maps, self.kernel.to(dtype=maps.dtype)).reshape(-1)
        # Argmax + gather stays ONNX-friendly (avoid data-dependent Python indexing).
        peak_idx = torch.argmax(scores)
        peak_mass = torch.gather(scores, 0, peak_idx.reshape(1)).reshape(())
        wout = FRAME_W - VIEWPORT_W + 1
        peak_idx_f = peak_idx.to(dtype=torch.float32)
        y_f = torch.floor(peak_idx_f / float(wout))
        x_f = peak_idx_f - y_f * float(wout)
        x1 = x_f * self.inv_frame[0]
        y1 = y_f * self.inv_frame[1]
        x2 = (x_f + self.size_px[0]) * self.inv_frame[2]
        y2 = (y_f + self.size_px[1]) * self.inv_frame[3]
        conf = peak_mass.to(dtype=torch.float32) * self.inv_area
        return torch.stack([x1, y1, x2, y2, conf])


class ViewportONNXWrapper(nn.Module):
    """Serving graph: uint8 frames [32,3,224,384] -> bbox_conf [5]."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = _NineSixteenHead()
        self.register_buffer(
            "mean",
            torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def preprocess(self, frames_uint8: torch.Tensor) -> torch.Tensor:
        # frames: [T, C, H, W] uint8 or float
        x = frames_uint8.to(dtype=torch.float32) / 255.0
        # ImageNet norm over channel dim; reshape to [T,C,H,W] then batch.
        x = (x - self.mean) / self.std
        # [T,C,H,W] -> [1,C,T,H,W]
        return x.permute(1, 0, 2, 3).unsqueeze(0)

    def forward_saliency(self, frames_uint8: torch.Tensor) -> torch.Tensor:
        clip = self.preprocess(frames_uint8)
        pred = self.backbone(clip)
        return squeeze_saliency(pred)

    def forward(self, frames_uint8: torch.Tensor) -> torch.Tensor:
        saliency = self.forward_saliency(frames_uint8)
        return self.head(saliency)


class ViewportDebugONNXWrapper(ViewportONNXWrapper):
    """Debug graph: returns (saliency[H,W], bbox_conf[5])."""

    def forward(self, frames_uint8: torch.Tensor):  # type: ignore[override]
        saliency = self.forward_saliency(frames_uint8)
        bbox = self.head(saliency)
        return saliency, bbox


def export_weights_only_state_dict(checkpoint_path: str, out_path: str) -> dict:
    """Write backbone weights-only state dict for Cohub PT flavor."""
    device = torch.device("cpu")
    model, meta = load_saliency_model(checkpoint_path, device)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    return meta
