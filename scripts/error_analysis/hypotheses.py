"""Fixed failure-mode catalog and geometric proposal rules."""

from __future__ import annotations

# Stable ids used for bucketing and the review UI.
HYPOTHESES = [
    {
        "id": "center_bias",
        "title": "Center bias",
        "description": (
            "Prediction stays near frame center while ground-truth attention "
            "is peripheral (sideline action, corner play, wide shot)."
        ),
    },
    {
        "id": "wrong_focus",
        "title": "Wrong focus / competing players",
        "description": (
            "Multiple plausible targets; prediction peak is far from GT peak "
            "and lands on a different player or region."
        ),
    },
    {
        "id": "spatial_shift",
        "title": "Spatial shift",
        "description": (
            "Prediction shape roughly matches GT but is translated "
            "(tracking lag or slight mis-localization)."
        ),
    },
    {
        "id": "over_diffuse",
        "title": "Over-diffuse prediction",
        "description": "GT is sharp/peaky while the prediction spreads mass too widely.",
    },
    {
        "id": "over_peaky",
        "title": "Over-peaky prediction",
        "description": "GT is broad/soft while the prediction collapses to a tight blob.",
    },
    {
        "id": "camera_or_motion",
        "title": "Camera motion / fast action",
        "description": (
            "Heavy pan, zoom, or motion blur across the clip makes temporal "
            "aggregation fail."
        ),
    },
    {
        "id": "temporal_inconsistency",
        "title": "Temporal inconsistency",
        "description": (
            "Ground-truth attention moves across the 32-frame clip (player/ball "
            "tracking, cut, or evolving play) while the model emits one static "
            "map for the whole window, so early/mid frames disagree with the "
            "prediction even if the last frame is closer."
        ),
    },
    {
        "id": "graphics_overlay",
        "title": "Broadcast graphics",
        "description": "Scorebug, replay bug, or on-screen text distracts the model.",
    },
    {
        "id": "annotation_issue",
        "title": "Annotation / GT issue",
        "description": (
            "Ground-truth map looks noisy, empty, or inconsistent with the "
            "visible action (data quality, not model)."
        ),
    },
    {
        "id": "low_visibility",
        "title": "Low visibility",
        "description": "Dark, low-contrast, occluded, or tiny target in frame.",
    },
    {
        "id": "unexplained",
        "title": "Unexplained",
        "description": "Hard example with no confirmed geometric or VLM hypothesis.",
    },
]

HYPOTHESIS_IDS = [item["id"] for item in HYPOTHESES]
HYPOTHESIS_BY_ID = {item["id"]: item for item in HYPOTHESES}


def propose_from_features(features: dict) -> list[dict]:
    """Propose hypotheses from geometric features. Each entry is
    {id, confidence, evidence} before VLM confirmation."""
    proposals = []

    gt_r = float(features.get("gt_peak_radius", 0.0))
    pred_r = float(features.get("pred_peak_radius", 0.0))
    peak_dist = float(features.get("peak_distance_norm", 0.0))
    center_delta = float(features.get("center_bias_delta", 0.0))
    gt_entropy = float(features.get("gt_entropy", 0.0))
    pred_entropy = float(features.get("pred_entropy", 0.0))
    gt_maxima = int(features.get("gt_local_maxima", 0))
    corr_shift = float(features.get("best_shift_corr", 0.0))
    shift_mag = float(features.get("best_shift_norm", 0.0))
    frame_std = float(features.get("frame_luma_std", 0.0))
    frame_mean = float(features.get("frame_luma_mean", 0.0))
    motion = float(features.get("clip_motion_mean", 0.0))
    gt_peak_disp = float(features.get("gt_peak_displacement_norm", 0.0))
    gt_peak_path = float(features.get("gt_peak_path_norm", 0.0))
    gt_temp_corr = float(features.get("gt_temporal_corr_mean", 1.0))

    if center_delta > 0.18 and pred_r < 0.28 and gt_r > 0.35:
        proposals.append(
            {
                "id": "center_bias",
                "confidence": min(0.95, 0.45 + center_delta),
                "evidence": (
                    "pred radius=%.2f gt radius=%.2f center_bias_delta=%.2f"
                    % (pred_r, gt_r, center_delta)
                ),
            }
        )

    if peak_dist > 0.22 and gt_maxima >= 2:
        proposals.append(
            {
                "id": "wrong_focus",
                "confidence": min(0.95, 0.4 + peak_dist),
                "evidence": (
                    "peak_dist=%.2f gt_local_maxima=%d" % (peak_dist, gt_maxima)
                ),
            }
        )
    elif peak_dist > 0.28:
        proposals.append(
            {
                "id": "wrong_focus",
                "confidence": min(0.85, 0.35 + peak_dist),
                "evidence": "peak_dist=%.2f (single-mode GT)" % peak_dist,
            }
        )

    if corr_shift > 0.55 and shift_mag > 0.06 and peak_dist > 0.08:
        proposals.append(
            {
                "id": "spatial_shift",
                "confidence": min(0.9, 0.35 + corr_shift * 0.5),
                "evidence": (
                    "best_shift_corr=%.2f shift_norm=%.2f" % (corr_shift, shift_mag)
                ),
            }
        )

    if pred_entropy - gt_entropy > 0.35 and gt_entropy < 4.2:
        proposals.append(
            {
                "id": "over_diffuse",
                "confidence": min(0.9, 0.4 + (pred_entropy - gt_entropy) * 0.3),
                "evidence": "pred_entropy=%.2f gt_entropy=%.2f" % (pred_entropy, gt_entropy),
            }
        )

    if gt_entropy - pred_entropy > 0.45 and pred_entropy < 3.8:
        proposals.append(
            {
                "id": "over_peaky",
                "confidence": min(0.9, 0.4 + (gt_entropy - pred_entropy) * 0.25),
                "evidence": "pred_entropy=%.2f gt_entropy=%.2f" % (pred_entropy, gt_entropy),
            }
        )

    if motion > 0.085:
        proposals.append(
            {
                "id": "camera_or_motion",
                "confidence": min(0.85, 0.35 + motion * 4.0),
                "evidence": "clip_motion_mean=%.3f" % motion,
            }
        )

    # Model predicts one map for the window; large GT peak travel => temporal miss.
    if gt_peak_disp > 0.12 or gt_peak_path > 0.35 or gt_temp_corr < 0.55:
        proposals.append(
            {
                "id": "temporal_inconsistency",
                "confidence": min(
                    0.95,
                    0.4
                    + 0.8 * gt_peak_disp
                    + 0.25 * max(0.0, gt_peak_path - 0.2)
                    + 0.35 * max(0.0, 0.7 - gt_temp_corr),
                ),
                "evidence": (
                    "gt_peak_disp=%.2f gt_peak_path=%.2f gt_temporal_corr=%.2f"
                    % (gt_peak_disp, gt_peak_path, gt_temp_corr)
                ),
            }
        )

    if frame_mean < 0.22 or frame_std < 0.08:
        proposals.append(
            {
                "id": "low_visibility",
                "confidence": 0.55 if frame_mean < 0.22 else 0.5,
                "evidence": "luma_mean=%.2f luma_std=%.2f" % (frame_mean, frame_std),
            }
        )

    proposals.sort(key=lambda item: item["confidence"], reverse=True)
    # Keep top proposals for VLM; unexplained filled later if none confirm.
    return proposals[:4]
