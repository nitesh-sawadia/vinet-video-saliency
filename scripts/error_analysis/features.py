"""Geometric features comparing a predicted saliency map to GT."""

from __future__ import annotations

import numpy as np


def _as_prob(sal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    sal = np.asarray(sal, dtype=np.float64)
    sal = np.clip(sal, 0.0, None)
    total = float(sal.sum())
    if total <= eps:
        return np.full(sal.shape, 1.0 / sal.size, dtype=np.float64)
    return sal / total


def _peak_yx(sal: np.ndarray) -> tuple[float, float]:
    flat = int(np.argmax(sal))
    y, x = np.unravel_index(flat, sal.shape)
    return float(y), float(x)


def _radius_from_center(y: float, x: float, height: int, width: int) -> float:
    cy, cx = (height - 1) * 0.5, (width - 1) * 0.5
    dy = (y - cy) / max(height - 1, 1)
    dx = (x - cx) / max(width - 1, 1)
    return float(np.hypot(dy, dx))


def _entropy(prob: np.ndarray, eps: float = 1e-12) -> float:
    flat = prob.reshape(-1)
    flat = flat[flat > eps]
    return float(-(flat * np.log(flat)).sum())


def _local_maxima_count(sal: np.ndarray, min_rel: float = 0.35) -> int:
    """Count coarse local maxima above a fraction of the peak (3x3)."""
    if sal.size == 0:
        return 0
    peak = float(sal.max())
    if peak <= 0:
        return 0
    # Max-pool downsample for speed; keeps peaks that fall between strides.
    step = 4
    height, width = sal.shape
    nh, nw = height // step, width // step
    if nh < 3 or nw < 3:
        small = sal
    else:
        cropped = sal[: nh * step, : nw * step]
        small = cropped.reshape(nh, step, nw, step).max(axis=(1, 3))
    thr = peak * min_rel
    padded = np.pad(small, 1, mode="edge")
    center = padded[1:-1, 1:-1]
    neigh_max = np.maximum.reduce(
        [
            padded[0:-2, 0:-2],
            padded[0:-2, 1:-1],
            padded[0:-2, 2:],
            padded[1:-1, 0:-2],
            padded[1:-1, 2:],
            padded[2:, 0:-2],
            padded[2:, 1:-1],
            padded[2:, 2:],
        ]
    )
    mask = (center >= thr) & (center >= neigh_max) & (center > 0)
    return int(mask.sum())


def _best_shift(pred: np.ndarray, gt: np.ndarray, max_shift: int = 24) -> tuple[float, float, float]:
    """Return (best_corr, shift_y_norm, shift_x_norm) via coarse search."""
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)
    pred = pred - pred.mean()
    gt = gt - gt.mean()
    pred_norm = float(np.linalg.norm(pred))
    gt_norm = float(np.linalg.norm(gt))
    if pred_norm < 1e-8 or gt_norm < 1e-8:
        return 0.0, 0.0, 0.0
    pred = pred / pred_norm
    gt = gt / gt_norm
    height, width = gt.shape
    best = -1.0
    best_dy = 0
    best_dx = 0
    step = max(2, max_shift // 8)
    for dy in range(-max_shift, max_shift + 1, step):
        for dx in range(-max_shift, max_shift + 1, step):
            ys = slice(max(0, dy), min(height, height + dy))
            xs = slice(max(0, dx), min(width, width + dx))
            yt = slice(max(0, -dy), min(height, height - dy))
            xt = slice(max(0, -dx), min(width, width - dx))
            a = pred[ys, xs]
            b = gt[yt, xt]
            if a.size < 64:
                continue
            corr = float((a * b).sum())
            if corr > best:
                best = corr
                best_dy = dy
                best_dx = dx
    shift_norm = float(np.hypot(best_dy / max(height - 1, 1), best_dx / max(width - 1, 1)))
    return float(best), float(best_dy / max(height - 1, 1)), shift_norm


def frame_luma_stats(frame_rgb_u8: np.ndarray) -> tuple[float, float]:
    """frame_rgb_u8: HxWx3 uint8 or float in [0,1]/"""
    frame = np.asarray(frame_rgb_u8, dtype=np.float64)
    if frame.max() > 1.5:
        frame = frame / 255.0
    luma = 0.2126 * frame[..., 0] + 0.7152 * frame[..., 1] + 0.0722 * frame[..., 2]
    return float(luma.mean()), float(luma.std())


def clip_motion_mean(frames_rgb: list[np.ndarray]) -> float:
    """Mean absolute frame-to-frame luma diff over a subsample of the clip."""
    if len(frames_rgb) < 2:
        return 0.0
    idxs = list(range(0, len(frames_rgb), max(1, len(frames_rgb) // 8)))
    if idxs[-1] != len(frames_rgb) - 1:
        idxs.append(len(frames_rgb) - 1)
    diffs = []
    prev = None
    for index in idxs:
        frame = np.asarray(frames_rgb[index], dtype=np.float64)
        if frame.max() > 1.5:
            frame = frame / 255.0
        luma = 0.2126 * frame[..., 0] + 0.7152 * frame[..., 1] + 0.0722 * frame[..., 2]
        if prev is not None:
            diffs.append(float(np.mean(np.abs(luma - prev))))
        prev = luma
    return float(np.mean(diffs)) if diffs else 0.0


def compute_temporal_features(gt_maps_thw: np.ndarray) -> dict:
    """Temporal stats over the 32 GT maps (model pred is a single static map)."""
    maps = np.asarray(gt_maps_thw, dtype=np.float64)
    if maps.ndim != 3 or maps.shape[0] < 2:
        return {
            "gt_peak_displacement_norm": 0.0,
            "gt_peak_path_norm": 0.0,
            "gt_temporal_corr_mean": 1.0,
        }
    t_count, height, width = maps.shape
    peaks = [_peak_yx(maps[t]) for t in range(t_count)]
    path = 0.0
    for i in range(1, t_count):
        y0, x0 = peaks[i - 1]
        y1, x1 = peaks[i]
        path += float(
            np.hypot((y1 - y0) / max(height - 1, 1), (x1 - x0) / max(width - 1, 1))
        )
    y0, x0 = peaks[0]
    y1, x1 = peaks[-1]
    displacement = float(
        np.hypot((y1 - y0) / max(height - 1, 1), (x1 - x0) / max(width - 1, 1))
    )

    corrs = []
    for i in range(1, t_count):
        a = maps[i - 1].reshape(-1)
        b = maps[i].reshape(-1)
        a = a - a.mean()
        b = b - b.mean()
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom < 1e-8:
            continue
        corrs.append(float(np.dot(a, b) / denom))
    return {
        "gt_peak_displacement_norm": displacement,
        "gt_peak_path_norm": path,
        "gt_temporal_corr_mean": float(np.mean(corrs)) if corrs else 1.0,
    }


def compute_map_features(
    pred: np.ndarray,
    gt: np.ndarray,
    frame_rgb: np.ndarray | None = None,
    clip_frames: list[np.ndarray] | None = None,
    gt_maps_thw: np.ndarray | None = None,
) -> dict:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError("pred/gt shape mismatch: %s vs %s" % (pred.shape, gt.shape))
    height, width = gt.shape
    pred_p = _as_prob(pred)
    gt_p = _as_prob(gt)
    gy, gx = _peak_yx(gt)
    py, px = _peak_yx(pred)
    peak_dist = float(
        np.hypot((gy - py) / max(height - 1, 1), (gx - px) / max(width - 1, 1))
    )
    gt_r = _radius_from_center(gy, gx, height, width)
    pred_r = _radius_from_center(py, px, height, width)
    best_corr, _, shift_norm = _best_shift(pred, gt)

    features = {
        "gt_peak_yx": [gy, gx],
        "pred_peak_yx": [py, px],
        "gt_peak_radius": gt_r,
        "pred_peak_radius": pred_r,
        "center_bias_delta": float(gt_r - pred_r),
        "peak_distance_norm": peak_dist,
        "gt_entropy": _entropy(gt_p),
        "pred_entropy": _entropy(pred_p),
        "gt_local_maxima": _local_maxima_count(gt),
        "best_shift_corr": float(best_corr),
        "best_shift_norm": float(shift_norm),
        "frame_luma_mean": 0.0,
        "frame_luma_std": 0.0,
        "clip_motion_mean": 0.0,
        "gt_peak_displacement_norm": 0.0,
        "gt_peak_path_norm": 0.0,
        "gt_temporal_corr_mean": 1.0,
    }
    if frame_rgb is not None:
        mean, std = frame_luma_stats(frame_rgb)
        features["frame_luma_mean"] = mean
        features["frame_luma_std"] = std
    if clip_frames:
        features["clip_motion_mean"] = clip_motion_mean(clip_frames)
    if gt_maps_thw is not None:
        features.update(compute_temporal_features(gt_maps_thw))
    return features
