"""VLM-audit Sol boxes on hard examples; request corrected boxes when inadequate."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from hypotheses import HYPOTHESIS_BY_ID
from viz import (
    crop_left_panel,
    draw_sol_boxes_rgb,
    draw_vlm_boxes_rgb,
    load_rgb_u8,
    load_sol_native_targets,
)
from vlm import DEFAULT_VLM_MODEL, audit_sol_saliency


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _default_pre_split_root(split_root: str | None) -> str | None:
    if not split_root:
        return None
    parent = os.path.dirname(os.path.abspath(split_root.rstrip(os.sep)))
    candidate = os.path.join(parent, "pre_split")
    return candidate if os.path.isdir(candidate) else None


def _manifest_by_rel(split_root: str) -> dict[str, dict]:
    path = Path(split_root) / "split_manifest.json"
    if not path.is_file():
        return {}
    manifest = json.loads(path.read_text())
    by_rel: dict[str, dict] = {}
    for window in manifest.get("windows") or []:
        if not isinstance(window, dict):
            continue
        name = Path(str(window.get("dir") or "")).name
        rel = f'{window.get("split")}/{window.get("sport")}/{window.get("game")}/{name}'
        by_rel[rel] = window
    return by_rel


def _keyframe_indices(n_frames: int, max_frames: int) -> list[int]:
    if n_frames <= 0:
        return []
    if n_frames <= max_frames:
        return list(range(n_frames))
    # Evenly spaced, always include first and last.
    raw = [int(round(i * (n_frames - 1) / (max_frames - 1))) for i in range(max_frames)]
    out: list[int] = []
    for index in raw:
        if not out or index != out[-1]:
            out.append(index)
    if out[-1] != n_frames - 1:
        out.append(n_frames - 1)
    return out


def _summarize_targets(targets: list[dict]) -> list[dict]:
    summary = []
    for target in targets or []:
        box = target.get("bbox_xyxy_norm") or []
        area = 0.0
        if len(box) == 4:
            area = max(0.0, float(box[2]) - float(box[0])) * max(
                0.0, float(box[3]) - float(box[1])
            )
        summary.append(
            {
                "label": target.get("label") or target.get("target_type"),
                "bbox_xyxy_norm": box,
                "area_frac": round(area, 4),
            }
        )
    return summary


def _densify_corrected_frames(
    corrected_frames: list[dict],
    *,
    n_frames: int = 32,
    scene_change_at: list[int] | None = None,
) -> list[dict]:
    """Interpolate keyframe boxes onto every frame, without blending across cuts.

    Within a continuous scene segment, match targets by stable label and lerp.
    At ``scene_change_at`` boundaries, hold the previous keyframe (no cross-fade)
    so a legitimate scene change can switch targets cleanly.
    """
    if not corrected_frames:
        return []
    keyed = {
        int(row["frame_index"]): list(row.get("targets") or [])
        for row in corrected_frames
        if "frame_index" in row
    }
    keys = sorted(keyed)
    if not keys:
        return []
    cuts = {int(v) for v in (scene_change_at or [])}

    def _lerp_box(a: list[float], b: list[float], t: float) -> list[float]:
        return [round((1.0 - t) * float(a[i]) + t * float(b[i]), 5) for i in range(4)]

    def _crosses_cut(left: int, right: int) -> bool:
        if left >= right:
            return False
        return any(left < cut <= right for cut in cuts)

    dense: list[dict] = []
    for frame_index in range(n_frames):
        if frame_index in keyed:
            dense.append({"frame_index": frame_index, "targets": keyed[frame_index]})
            continue
        left = max((k for k in keys if k <= frame_index), default=keys[0])
        right = min((k for k in keys if k >= frame_index), default=keys[-1])
        if left == right or _crosses_cut(left, right):
            # Hold last keyframe inside the segment (or before a scene cut).
            dense.append({"frame_index": frame_index, "targets": keyed[left]})
            continue
        t = (frame_index - left) / float(right - left)
        left_targets = keyed[left]
        right_by_label = {
            str(target.get("label") or ""): target for target in keyed[right]
        }
        merged = []
        for index, src in enumerate(left_targets):
            label = str(src.get("label") or "")
            dst = right_by_label.get(label)
            if dst is None and index < len(keyed[right]):
                dst = keyed[right][index]
            if dst is None or not src.get("bbox_xyxy_norm") or not dst.get("bbox_xyxy_norm"):
                merged.append(dict(src))
                continue
            merged.append(
                {
                    "label": label or dst.get("label") or "salient",
                    "bbox_xyxy_norm": _lerp_box(
                        src["bbox_xyxy_norm"], dst["bbox_xyxy_norm"], t
                    ),
                    "weight": round(
                        (1.0 - t) * float(src.get("weight", 1.0))
                        + t * float(dst.get("weight", 1.0)),
                        4,
                    ),
                }
            )
        dense.append({"frame_index": frame_index, "targets": merged})
    return dense


def _load_audit_frames(
    sample: dict,
    out_dir: str,
    *,
    split_root: str | None,
    pre_split_root: str | None,
    man: dict,
    max_frames: int,
) -> tuple[list[Image.Image], list[int], list[dict], list[list[dict]]]:
    """Return (pil_frames, indices, sol_summary, sol_targets_per_keyframe)."""
    window_path = Path(sample.get("path") or "")
    if (not window_path.is_dir()) and split_root and sample.get("relpath"):
        window_path = Path(split_root) / sample["relpath"]

    sol_all = load_sol_native_targets(
        window_path,
        pre_split_root=pre_split_root,
        source=man.get("source"),
        native_start=man.get("native_start"),
        native_end=man.get("native_end"),
    )

    clip_rel = sample.get("clip") or ("clips/%s.webp" % sample["id"])
    clip_path = Path(out_dir) / clip_rel
    images: list[Image.Image] = []
    indices: list[int] = []
    sol_for_keys: list[list[dict]] = []

    if clip_path.is_file():
        webp = Image.open(clip_path)
        n_frames = int(getattr(webp, "n_frames", 1) or 1)
        indices = _keyframe_indices(n_frames, max_frames)
        for index in indices:
            webp.seek(index)
            panel = np.asarray(webp.convert("RGB"))
            left = crop_left_panel(panel)
            # Prefer clean frame + Sol boxes from disk when available.
            frame_file = window_path / "images" / ("%04d.png" % (index + 1))
            if frame_file.is_file():
                left = load_rgb_u8(str(frame_file))
                targets = sol_all[index] if index < len(sol_all) else []
                left = draw_sol_boxes_rgb(left, targets)
            images.append(Image.fromarray(left))
            sol_for_keys.append(sol_all[index] if index < len(sol_all) else [])
    elif window_path.is_dir() and (window_path / "images").is_dir():
        n_frames = len(list((window_path / "images").glob("*.png"))) or 32
        indices = _keyframe_indices(n_frames, max_frames)
        for index in indices:
            frame_file = window_path / "images" / ("%04d.png" % (index + 1))
            if not frame_file.is_file():
                continue
            left = load_rgb_u8(str(frame_file))
            targets = sol_all[index] if index < len(sol_all) else []
            left = draw_sol_boxes_rgb(left, targets)
            images.append(Image.fromarray(left))
            sol_for_keys.append(targets)
        indices = indices[: len(images)]
    else:
        raise FileNotFoundError(
            "Need clip webp or window images for %s" % sample.get("id")
        )

    summary = [
        {"frame_index": index, "targets": _summarize_targets(targets)}
        for index, targets in zip(indices, sol_for_keys)
    ]
    return images, indices, summary, sol_for_keys


def _clean_left_from_panel(panel_rgb: np.ndarray) -> np.ndarray:
    """Left third of the 1x3 review panel (RGB; may still include baked Sol cyan)."""
    return crop_left_panel(panel_rgb)


def _save_corrected_track_clip(
    out_dir: str,
    sample: dict,
    *,
    dense_frames: list[dict],
    fps: int = 8,
) -> str | None:
    """Write a 32-frame WebP: clean-ish RGB + green VLM-corrected boxes only.

    Uses the left third of the existing review clip as the RGB source (same
    coordinate frame the VLM used). Prefer densified 32-frame tracks so the UI
    shows temporal consistency, not a single still.
    """
    if not dense_frames:
        return None
    clip_rel = sample.get("clip") or ("clips/%s.webp" % sample["id"])
    clip_path = Path(out_dir) / clip_rel
    if not clip_path.is_file():
        return None

    by_index = {
        int(row["frame_index"]): list(row.get("targets") or [])
        for row in dense_frames
        if "frame_index" in row
    }
    webp = Image.open(clip_path)
    n_frames = int(getattr(webp, "n_frames", 1) or 1)
    panels: list[Image.Image] = []
    for t in range(n_frames):
        webp.seek(t)
        left = _clean_left_from_panel(np.asarray(webp.convert("RGB")))
        # Paint a dark translucent veil over any baked cyan Sol boxes, then
        # redraw RGB by blending toward a box-free look is hard; instead draw
        # thick green corrected boxes so the VLM track dominates.
        left = draw_vlm_boxes_rgb(left, by_index.get(t) or [], draw_labels=True)
        # Caption frame index for temporal review.
        captioned = left.copy()
        cv2.putText(
            captioned,
            "f%02d VLM track" % t,
            (6, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (80, 220, 120),
            1,
            cv2.LINE_AA,
        )
        panels.append(Image.fromarray(captioned))

    rel = os.path.join("sol_audit", "%s_corrected.webp" % sample["id"])
    abs_path = os.path.join(out_dir, rel)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    duration_ms = max(1, int(round(1000.0 / float(fps))))
    panels[0].save(
        abs_path,
        format="WEBP",
        save_all=True,
        append_images=panels[1:],
        duration=duration_ms,
        loop=0,
        quality=80,
        method=4,
    )
    # Also keep a mid-frame still for quick glance.
    still_rel = os.path.join("sol_audit", "%s_corrected.png" % sample["id"])
    panels[n_frames // 2].save(os.path.join(out_dir, still_rel))
    return rel.replace("\\", "/")


def rebuild_corrected_media(out_dir: str, *, fps: int = 8) -> dict:
    """Rebuild 32-frame corrected WebP tracks from sol_audit.jsonl (no VLM calls)."""
    audit_path = os.path.join(out_dir, "sol_audit.jsonl")
    if not os.path.isfile(audit_path):
        raise SystemExit("Missing sol_audit.jsonl in %s" % out_dir)
    rows = _read_jsonl(audit_path)
    scores = {
        str(r["id"]): r
        for r in _read_jsonl(os.path.join(out_dir, "scores.jsonl"))
        if r.get("hard")
    }
    updated = 0
    for row in rows:
        dense = row.get("corrected_frames_dense") or []
        if not dense:
            dense = _densify_corrected_frames(row.get("corrected_frames") or [], n_frames=32)
            row["corrected_frames_dense"] = dense
        if not dense:
            continue
        sample = dict(scores.get(str(row["id"])) or {})
        sample["id"] = row["id"]
        sample["clip"] = sample.get("clip") or row.get("clip") or ("clips/%s.webp" % row["id"])
        rel = _save_corrected_track_clip(out_dir, sample, dense_frames=dense, fps=fps)
        if rel:
            row["corrected_preview"] = rel
            row["corrected_clip"] = rel
            updated += 1
            print("rebuilt", row["id"], rel, flush=True)
    _write_jsonl(audit_path, rows)

    audit_by_id = {str(r["id"]): r for r in rows}
    for name in ("review_index.json", "analysis.json"):
        path = os.path.join(out_dir, name)
        if not os.path.isfile(path):
            continue
        with open(path) as handle:
            doc = json.load(handle)
        for sample in doc.get("samples") or []:
            audit = audit_by_id.get(str(sample.get("id")))
            if not audit:
                continue
            sample["sol_audit"] = {
                "sol_adequate": audit.get("sol_adequate"),
                "issue": audit.get("issue"),
                "confidence": audit.get("confidence"),
                "primary_track": audit.get("primary_track"),
                "scene_change_at": audit.get("scene_change_at") or [],
                "rationale": audit.get("rationale"),
                "corrected_preview": audit.get("corrected_preview"),
                "corrected_clip": audit.get("corrected_clip") or audit.get("corrected_preview"),
                "n_corrected_frames": len(audit.get("corrected_frames") or []),
                "n_dense_frames": len(audit.get("corrected_frames_dense") or []),
            }
        with open(path, "w") as handle:
            json.dump(doc, handle, indent=2)
            handle.write("\n")
        print("Updated", path, flush=True)
    return {"n_rebuilt": updated, "n_rows": len(rows)}


def audit_hard_sol_boxes(
    out_dir: str,
    *,
    split_root: str | None = None,
    pre_split_root: str | None = None,
    vlm_model: str = DEFAULT_VLM_MODEL,
    max_frames: int = 8,
    limit: int | None = None,
    ids: list[str] | None = None,
    resume: bool = True,
) -> dict:
    scores_path = os.path.join(out_dir, "scores.jsonl")
    if not os.path.isfile(scores_path):
        raise SystemExit("Missing scores.jsonl in %s" % out_dir)

    summary_path = os.path.join(out_dir, "summary.json")
    if split_root is None and os.path.isfile(summary_path):
        with open(summary_path) as handle:
            split_root = json.load(handle).get("split_root")
    if pre_split_root is None:
        pre_split_root = _default_pre_split_root(split_root)
    man_by_rel = _manifest_by_rel(split_root) if split_root else {}

    rows = _read_jsonl(scores_path)
    hard = [row for row in rows if row.get("hard")]
    hard.sort(key=lambda r: r["loss"], reverse=True)
    if ids:
        want = set(ids)
        hard = [row for row in hard if str(row.get("id")) in want]
    if limit is not None:
        hard = hard[:limit]

    audit_path = os.path.join(out_dir, "sol_audit.jsonl")
    done_by_id: dict[str, dict] = {}
    if resume and os.path.isfile(audit_path):
        for row in _read_jsonl(audit_path):
            done_by_id[str(row.get("id"))] = row

    results: list[dict] = []
    tic = time.time()
    inadequate = 0

    for i, sample in enumerate(hard):
        sample_id = str(sample["id"])
        if sample_id in done_by_id and resume:
            prior = done_by_id[sample_id]
            # Retry parse/API failures; keep successful audits.
            if prior.get("issue") != "error" and prior.get("sol_adequate") is not None:
                results.append(prior)
                if not prior.get("sol_adequate"):
                    inadequate += 1
                print(
                    "[%d/%d] %s resume issue=%s"
                    % (i + 1, len(hard), sample_id, prior.get("issue")),
                    flush=True,
                )
                continue
            print(
                "[%d/%d] %s retry prior error" % (i + 1, len(hard), sample_id),
                flush=True,
            )

        man = man_by_rel.get(str(sample.get("relpath") or "")) or {}
        try:
            frames, indices, summary, _sol = _load_audit_frames(
                sample,
                out_dir,
                split_root=split_root,
                pre_split_root=pre_split_root,
                man=man,
                max_frames=max_frames,
            )
            verdict = audit_sol_saliency(
                sample,
                frame_images=frames,
                frame_indices=indices,
                sol_targets_summary=summary,
                model=vlm_model,
            )
        except Exception as exc:
            print("Sol audit failed for %s: %s" % (sample_id, exc), flush=True)
            verdict = {
                "sol_adequate": None,
                "issue": "error",
                "confidence": 0.0,
                "rationale": str(exc),
                "corrected_frames": [],
                "frame_indices": [],
                "model": vlm_model,
                "error": str(exc),
            }
            frames, indices = [], []

        corrected = verdict.get("corrected_frames") or []
        scene_change_at = list(verdict.get("scene_change_at") or [])
        densified = (
            _densify_corrected_frames(
                corrected, n_frames=32, scene_change_at=scene_change_at
            )
            if corrected
            else []
        )
        preview = None
        if densified:
            preview = _save_corrected_track_clip(
                out_dir,
                sample,
                dense_frames=densified,
            )
        entry = {
            "id": sample_id,
            "sport": sample.get("sport"),
            "game": sample.get("game"),
            "window": sample.get("window"),
            "relpath": sample.get("relpath"),
            "loss": sample.get("loss"),
            "cc": sample.get("cc"),
            "sol_adequate": verdict.get("sol_adequate"),
            "issue": verdict.get("issue"),
            "confidence": verdict.get("confidence"),
            "primary_track": verdict.get("primary_track"),
            "scene_change_at": scene_change_at,
            "rationale": verdict.get("rationale"),
            "corrected_frames": corrected,
            "corrected_frames_dense": densified,
            "frame_indices": verdict.get("frame_indices") or indices,
            "corrected_preview": preview,
            "corrected_clip": preview,
            "model": verdict.get("model"),
            "error": verdict.get("error"),
        }
        results.append(entry)
        done_by_id[sample_id] = entry
        _write_jsonl(audit_path, list(done_by_id.values()))

        if entry.get("sol_adequate") is False:
            inadequate += 1
        print(
            "[%d/%d] %s -> adequate=%s issue=%s conf=%.2f"
            % (
                i + 1,
                len(hard),
                sample_id,
                entry.get("sol_adequate"),
                entry.get("issue"),
                float(entry.get("confidence") or 0.0),
            ),
            flush=True,
        )

    # Merge into review_index / analysis without wiping prior fields.
    audit_by_id = {str(r["id"]): r for r in results}
    for name in ("review_index.json", "analysis.json"):
        path = os.path.join(out_dir, name)
        if not os.path.isfile(path):
            continue
        with open(path) as handle:
            doc = json.load(handle)
        samples = doc.get("samples") or []
        for sample in samples:
            audit = audit_by_id.get(str(sample.get("id")))
            if not audit:
                continue
            sample["sol_audit"] = {
                "sol_adequate": audit.get("sol_adequate"),
                "issue": audit.get("issue"),
                "confidence": audit.get("confidence"),
                "primary_track": audit.get("primary_track"),
                "scene_change_at": audit.get("scene_change_at") or [],
                "rationale": audit.get("rationale"),
                "corrected_preview": audit.get("corrected_preview"),
                "corrected_clip": audit.get("corrected_clip") or audit.get("corrected_preview"),
                "n_corrected_frames": len(audit.get("corrected_frames") or []),
                "n_dense_frames": len(audit.get("corrected_frames_dense") or []),
            }
            # Surface as a hypothesis tag when Sol is inadequate.
            if audit.get("sol_adequate") is False:
                confirmed = list(sample.get("confirmed") or [])
                if not any(c.get("id") == "sol_conservative" for c in confirmed):
                    confirmed.append(
                        {
                            "id": "sol_conservative",
                            "confidence": float(audit.get("confidence") or 0.6),
                            "rationale": audit.get("rationale") or "Sol boxes miss scene salience",
                        }
                    )
                    sample["confirmed"] = confirmed
                if sample.get("primary_hypothesis") in (None, "unexplained", "annotation_issue"):
                    sample["primary_hypothesis"] = "sol_conservative"
                    sample["title"] = HYPOTHESIS_BY_ID["sol_conservative"]["title"]
                secs = list(sample.get("secondary_hypotheses") or [])
                if "sol_conservative" not in secs and sample.get("primary_hypothesis") != "sol_conservative":
                    secs.append("sol_conservative")
                sample["secondary_hypotheses"] = secs
        # Refresh bucket counts for UI.
        if name == "review_index.json":
            counts: dict[str, int] = {}
            for sample in samples:
                hid = sample.get("primary_hypothesis") or "unexplained"
                counts[hid] = counts.get(hid, 0) + 1
            doc["bucket_counts"] = counts
            # Ensure new hypothesis appears in catalog.
            hyps = doc.get("hypotheses") or []
            if not any(h.get("id") == "sol_conservative" for h in hyps):
                hyps.append(HYPOTHESIS_BY_ID["sol_conservative"])
                doc["hypotheses"] = hyps
        with open(path, "w") as handle:
            json.dump(doc, handle, indent=2)
            handle.write("\n")
        print("Updated", path, flush=True)

    report = {
        "n_audited": len(results),
        "n_inadequate": inadequate,
        "n_adequate": sum(1 for r in results if r.get("sol_adequate") is True),
        "n_errors": sum(1 for r in results if r.get("issue") == "error"),
        "elapsed_sec": time.time() - tic,
        "audit_path": audit_path,
        "model": vlm_model,
    }
    with open(os.path.join(out_dir, "sol_audit_summary.json"), "w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2), flush=True)
    return report
