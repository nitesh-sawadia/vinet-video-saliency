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


def save_clip_video(
    path: str,
    frames_tchw,
    gt_maps_thw,
    pred_hw,
    fps: int = 8,
) -> str:
    """Write a 32-frame comparison clip: raw | GT overlay | pred overlay.

    Saves an animated WebP (browser-friendly) at ``path`` with a ``.webp``
    suffix if needed. Also writes a side-car ``.mp4`` when OpenCV can encode.
    Returns the WebP path used for the review UI.
    """
    frames = np.asarray(frames_tchw)
    gts = np.asarray(gt_maps_thw)
    pred = np.asarray(pred_hw, dtype=np.float32)
    if frames.ndim != 4:
        raise ValueError("frames must be TCHW, got %s" % (frames.shape,))
    if gts.ndim != 3:
        raise ValueError("gt maps must be THW, got %s" % (gts.shape,))
    if pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred[0]

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    root, ext = os.path.splitext(path)
    webp_path = path if ext.lower() == ".webp" else root + ".webp"
    panels = [build_panel_u8(frames[t], gts[t], pred) for t in range(frames.shape[0])]
    images = [Image.fromarray(panel) for panel in panels]
    duration_ms = max(1, int(round(1000.0 / float(fps))))
    images[0].save(
        webp_path,
        format="WEBP",
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        quality=80,
        method=4,
    )

    mp4_path = root + ".mp4"
    height, width = panels[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(mp4_path, fourcc, float(fps), (width, height))
    if writer.isOpened():
        try:
            for panel in panels:
                writer.write(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
    return webp_path


def load_rgb_u8(path: str, size_wh=(384, 224)) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if image.size != size_wh:
        image = image.resize(size_wh, Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)
