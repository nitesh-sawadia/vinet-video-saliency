"""Panel images: frame | GT overlay | pred overlay."""

from __future__ import annotations

import os

import cv2
import numpy as np
from PIL import Image

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
_OVERLAY_ALPHA = 0.5


def denormalize_frame_chw(frame_chw) -> np.ndarray:
    frame = np.asarray(frame_chw, dtype=np.float32)
    image = frame * _IMAGENET_STD + _IMAGENET_MEAN
    return np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)


def minmax(sal: np.ndarray) -> np.ndarray:
    low = float(np.min(sal))
    high = float(np.max(sal))
    if high <= low:
        return np.zeros_like(sal, dtype=np.float32)
    return ((sal - low) / (high - low)).astype(np.float32, copy=False)


def jet_overlay_chw(frame_chw, sal_hw) -> np.ndarray:
    frame = denormalize_frame_chw(frame_chw)
    sal = np.asarray(sal_hw, dtype=np.float32)
    height, width = frame.shape[1], frame.shape[2]
    if sal.shape != (height, width):
        sal = cv2.resize(sal, (width, height), interpolation=cv2.INTER_LINEAR)
    sal = minmax(sal)
    heat_bgr = cv2.applyColorMap(np.round(sal * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    frame_hwc = np.transpose(frame, (1, 2, 0))
    blended = (1.0 - _OVERLAY_ALPHA) * frame_hwc + _OVERLAY_ALPHA * heat
    return np.ascontiguousarray(np.transpose(np.clip(blended, 0.0, 1.0), (2, 0, 1)))


def chw_to_u8(chw: np.ndarray) -> np.ndarray:
    hwc = np.transpose(np.clip(chw, 0.0, 1.0), (1, 2, 0))
    return np.round(hwc * 255.0).astype(np.uint8)


def build_panel_u8(frame_chw, gt_hw, pred_hw) -> np.ndarray:
    """Return Hx(3W)x3 uint8 panel."""
    left = chw_to_u8(denormalize_frame_chw(frame_chw))
    mid = chw_to_u8(jet_overlay_chw(frame_chw, gt_hw))
    right = chw_to_u8(jet_overlay_chw(frame_chw, pred_hw))
    return np.concatenate([left, mid, right], axis=1)


def save_panel(path: str, frame_chw, gt_hw, pred_hw) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    panel = build_panel_u8(frame_chw, gt_hw, pred_hw)
    Image.fromarray(panel).save(path)


def load_rgb_u8(path: str, size_wh=(384, 224)) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if image.size != size_wh:
        image = image.resize(size_wh, Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)
