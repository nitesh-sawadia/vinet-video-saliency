"""Panel images: frame (+ Sol boxes) | GT overlay | pred overlay."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
_OVERLAY_ALPHA = 0.5
# Cyan Sol boxes (BGR for OpenCV draws on RGB arrays after conversion).
_SOL_BOX_RGB = (0, 230, 255)
_SOL_BOX_THICKNESS = 2


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


def _pre_split_from_source(pre_split_root: str | Path, source: str) -> Path | None:
    """Map a split-manifest ``source`` path onto the Linode pre_split tree."""
    if not pre_split_root or not source:
        return None
    parts = Path(source).parts
    for marker in ("saliency_sol_annotation", "pre_split"):
        if marker in parts:
            index = parts.index(marker)
            return Path(pre_split_root).joinpath(*parts[index + 1 :])
    # Fall back to sport/game/window trailing triple when present.
    if len(parts) >= 3:
        return Path(pre_split_root).joinpath(*parts[-3:])
    return None


def load_sol_native_targets(
    window_path: str | Path,
    *,
    pre_split_root: str | Path | None = None,
    source: str | None = None,
    native_start: int | None = None,
    native_end: int | None = None,
) -> list[list[dict[str, Any]]]:
    """Load per-frame Sol targets aligned to the 32-frame window.

    Prefer the window's own ``boxes.json`` (sliced copy written at split time,
    including any later GT regen). If missing, slice ``native_frames`` from the
    Linode ``pre_split`` segment pointed at by the manifest ``source``.
    """
    window_path = Path(window_path)
    doc: dict[str, Any] | None = None
    boxes_path = window_path / "boxes.json"
    if boxes_path.is_file():
        doc = json.loads(boxes_path.read_text())
        native = list(doc.get("native_frames") or [])
        if native:
            return [list(frame.get("targets") or []) for frame in native]

    # Fall back to pre_split Sol annotation.
    if source is None and isinstance(doc, dict):
        src_meta = doc.get("source") if isinstance(doc.get("source"), dict) else {}
        source = src_meta.get("dir")
        if native_start is None:
            native_start = src_meta.get("native_start")
        if native_end is None:
            native_end = src_meta.get("native_end")
    pre_dir = _pre_split_from_source(pre_split_root, source or "") if pre_split_root else None
    if pre_dir is None or not (pre_dir / "boxes.json").is_file():
        return []
    pre_doc = json.loads((pre_dir / "boxes.json").read_text())
    native = list(pre_doc.get("native_frames") or [])
    if not native:
        return []
    start = int(native_start or 0)
    end = int(native_end) if native_end is not None else start + 32
    sliced = native[start:end]
    return [list(frame.get("targets") or []) for frame in sliced]


def draw_sol_boxes_rgb(
    image_rgb: np.ndarray,
    targets: list[dict[str, Any]] | None,
    *,
    draw_labels: bool = True,
) -> np.ndarray:
    """Draw normalized Sol xyxy boxes onto an RGB uint8 frame."""
    if not targets:
        return image_rgb
    out = np.ascontiguousarray(image_rgb.copy())
    height, width = out.shape[:2]
    for target in targets:
        box = target.get("bbox_xyxy_norm") if isinstance(target, dict) else None
        if not box or len(box) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        p1 = (int(round(x1 * width)), int(round(y1 * height)))
        p2 = (int(round(x2 * width)), int(round(y2 * height)))
        cv2.rectangle(out, p1, p2, _SOL_BOX_RGB, _SOL_BOX_THICKNESS)
        if not draw_labels:
            continue
        label = str(target.get("label") or target.get("target_type") or target.get("target_id") or "")
        if not label:
            continue
        text = label[:28]
        ty = max(14, p1[1] - 4)
        cv2.putText(
            out,
            text,
            (p1[0], ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            _SOL_BOX_RGB,
            1,
            cv2.LINE_AA,
        )
    return out


def build_panel_u8(
    frame_chw,
    gt_hw,
    pred_hw,
    sol_targets: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    """Return Hx(3W)x3 uint8 panel. Left frame includes Sol boxes when provided."""
    left = chw_to_u8(denormalize_frame_chw(frame_chw))
    if sol_targets:
        left = draw_sol_boxes_rgb(left, sol_targets)
    mid = chw_to_u8(jet_overlay_chw(frame_chw, gt_hw))
    right = chw_to_u8(jet_overlay_chw(frame_chw, pred_hw))
    return np.concatenate([left, mid, right], axis=1)


def save_panel(
    path: str,
    frame_chw,
    gt_hw,
    pred_hw,
    sol_targets: list[dict[str, Any]] | None = None,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    panel = build_panel_u8(frame_chw, gt_hw, pred_hw, sol_targets=sol_targets)
    Image.fromarray(panel).save(path)


def save_clip_video(
    path: str,
    frames_tchw,
    gt_maps_thw,
    pred_hw,
    fps: int = 8,
    sol_targets_per_frame: list[list[dict[str, Any]]] | None = None,
) -> str:
    """Write a 32-frame comparison clip: frame+Sol boxes | GT overlay | pred overlay.

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
    panels = []
    for t in range(frames.shape[0]):
        targets = None
        if sol_targets_per_frame is not None and t < len(sol_targets_per_frame):
            targets = sol_targets_per_frame[t]
        panels.append(build_panel_u8(frames[t], gts[t], pred, sol_targets=targets))
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
