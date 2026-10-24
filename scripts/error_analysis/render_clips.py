"""Render 32-frame 1x3 comparison videos for hard examples."""

from __future__ import annotations

import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


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


def _load_manifest_by_relpath(split_root: str) -> dict[str, dict]:
    path = os.path.join(split_root, "split_manifest.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as handle:
        manifest = json.load(handle)
    by_rel: dict[str, dict] = {}
    for window in manifest.get("windows") or []:
        if not isinstance(window, dict):
            continue
        name = os.path.basename(str(window.get("dir") or ""))
        rel = "%s/%s/%s/%s" % (
            window.get("split"),
            window.get("sport"),
            window.get("game"),
            name,
        )
        by_rel[rel] = window
    return by_rel


def render_hard_clips(
    out_dir: str,
    checkpoint_path: str,
    split_root: str | None = None,
    pre_split_root: str | None = None,
    batch_size: int = 8,
    num_workers: int = 2,
    device_name: str | None = None,
    fps: int = 8,
    ids: list[str] | None = None,
    refresh_panels: bool = False,
) -> dict:
    """Re-infer hard examples and write clips/<id>.webp (full 32-frame panels)."""
    from features import compute_map_features, compute_temporal_features
    from score import load_model, _squeeze_pred, _resize_pred_to_gt, _release_cuda_after_inference, _ensure_vinet_path
    from viz import denormalize_frame_chw, load_sol_native_targets, save_clip_video, save_panel

    _ensure_vinet_path()
    from outplay_dataset import OutplayWindowDataset

    scores_path = os.path.join(out_dir, "scores.jsonl")
    if not os.path.isfile(scores_path):
        raise SystemExit("Missing scores.jsonl in %s" % out_dir)
    rows = _read_jsonl(scores_path)
    hard = [row for row in rows if row.get("hard")]
    if ids:
        want = {str(item) for item in ids}
        hard = [row for row in hard if str(row.get("id")) in want]
        missing = want - {str(row.get("id")) for row in hard}
        if missing:
            print("warning: ids not in hard set:", ", ".join(sorted(missing)), flush=True)
    if not hard:
        raise SystemExit("No hard examples flagged in %s" % scores_path)

    summary_path = os.path.join(out_dir, "summary.json")
    if split_root is None and os.path.isfile(summary_path):
        with open(summary_path) as handle:
            split_root = json.load(handle).get("split_root")
    if not split_root:
        raise SystemExit("--split-root is required (not found in summary.json)")
    if pre_split_root is None:
        pre_split_root = _default_pre_split_root(split_root)
    manifest_by_rel = _load_manifest_by_relpath(split_root)

    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(
        "Rendering %d hard clips on %s (pre_split=%s) ..."
        % (len(hard), device, pre_split_root or "none"),
        flush=True,
    )
    model, _meta = load_model(checkpoint_path, device)
    dataset = OutplayWindowDataset(split_root, "val", clip_size=32, validate=False)

    # Resolve by on-disk path. Score indexes go stale when split_manifest is filtered.
    path_to_index = {path: index for index, path in enumerate(dataset.paths)}
    by_id = {str(row["id"]): row for row in rows}
    hard_indexes: list[int] = []
    hard_rows: list[dict] = []
    skipped = []
    for row in hard:
        path = row.get("path") or os.path.join(split_root, row.get("relpath") or "")
        index = path_to_index.get(path)
        if index is None:
            skipped.append(str(row.get("id")))
            continue
        hard_indexes.append(index)
        hard_rows.append(by_id[str(row["id"])])
    if skipped:
        print("warning: missing from current split:", ", ".join(skipped), flush=True)
    if not hard_indexes:
        raise SystemExit("No requested hard examples remain in the current split")

    subset = Subset(dataset, hard_indexes)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    clips_dir = os.path.join(out_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    tic = time.time()
    done = 0
    with torch.no_grad():
        cursor = 0
        for batch in loader:
            clips, _gts, all_maps = batch
            bsz = int(clips.shape[0])
            batch_rows = hard_rows[cursor : cursor + bsz]
            cursor += bsz
            clips_dev = clips.to(device, non_blocking=True)
            preds = _squeeze_pred(model(clips_dev.permute(0, 2, 1, 3, 4)))
            # Match GT spatial size from maps.
            gt0 = all_maps[:, -1].to(device, non_blocking=True)
            preds = _resize_pred_to_gt(preds, gt0)
            preds_cpu = preds.detach().cpu().numpy()
            clips_cpu = clips.detach().cpu().numpy()
            maps_cpu = all_maps.detach().cpu().numpy()

            for b, row in enumerate(batch_rows):
                window_path = row.get("path") or os.path.join(
                    split_root, row.get("relpath") or ""
                )
                man = manifest_by_rel.get(str(row.get("relpath") or "")) or {}
                sol_targets = load_sol_native_targets(
                    window_path,
                    pre_split_root=pre_split_root,
                    source=man.get("source"),
                    native_start=man.get("native_start"),
                    native_end=man.get("native_end"),
                )
                row["sol_boxes"] = bool(sol_targets)
                clip_abs = save_clip_video(
                    os.path.join(clips_dir, "%s.webp" % row["id"]),
                    clips_cpu[b],
                    maps_cpu[b],
                    preds_cpu[b],
                    fps=fps,
                    sol_targets_per_frame=sol_targets or None,
                )
                row["clip"] = os.path.relpath(clip_abs, out_dir).replace("\\", "/")
                # Refresh temporal features now that we have all 32 GT maps.
                last_rgb = (
                    np.transpose(
                        denormalize_frame_chw(clips_cpu[b, -1]), (1, 2, 0)
                    )
                    * 255.0
                ).astype(np.uint8)
                feats = dict(row.get("features") or {})
                feats.update(
                    compute_map_features(
                        preds_cpu[b],
                        maps_cpu[b, -1],
                        frame_rgb=last_rgb,
                        gt_maps_thw=maps_cpu[b],
                    )
                )
                feats.update(compute_temporal_features(maps_cpu[b]))
                row["features"] = feats
                panel_rel = row.get("panel") or os.path.join("panels", "%s.png" % row["id"])
                panel_abs = os.path.join(out_dir, panel_rel)
                if refresh_panels or not os.path.isfile(panel_abs):
                    last_targets = sol_targets[-1] if sol_targets else None
                    save_panel(
                        panel_abs,
                        clips_cpu[b, -1],
                        maps_cpu[b, -1],
                        preds_cpu[b],
                        sol_targets=last_targets,
                    )
                    row["panel"] = panel_rel.replace("\\", "/")
                done += 1
            print(
                "clips %d/%d  elapsed=%.1fs" % (done, len(hard_rows), time.time() - tic),
                flush=True,
            )

    _release_cuda_after_inference(model, loader)
    _write_jsonl(scores_path, rows)

    refreshed_ids = {str(row["id"]) for row in hard_rows}
    sol_by_id = {row["id"]: bool(row.get("sol_boxes")) for row in rows if row.get("hard")}
    # Refresh review_index.json if classify already ran.
    review_path = os.path.join(out_dir, "review_index.json")
    if os.path.isfile(review_path):
        with open(review_path) as handle:
            review = json.load(handle)
        clip_by_id = {row["id"]: row.get("clip") for row in rows if row.get("hard")}
        panel_by_id = {row["id"]: row.get("panel") for row in rows if row.get("hard")}
        for sample in review.get("samples") or []:
            sample_id = sample.get("id")
            if sample_id in clip_by_id:
                sample["clip"] = clip_by_id[sample_id]
            if sample_id in panel_by_id and panel_by_id[sample_id]:
                sample["panel"] = panel_by_id[sample_id]
            if sample_id in refreshed_ids:
                # Preserve prior gt_regenerated flags; only bump media cache key.
                sample["media_version"] = "sol-boxes"
            if sample_id in sol_by_id:
                sample["sol_boxes"] = sol_by_id[sample_id]
        with open(review_path, "w") as handle:
            json.dump(review, handle, indent=2)
            handle.write("\n")
        print("Updated", review_path, flush=True)

    analysis_path = os.path.join(out_dir, "analysis.json")
    if os.path.isfile(analysis_path):
        with open(analysis_path) as handle:
            analysis = json.load(handle)
        clip_by_id = {row["id"]: row.get("clip") for row in rows if row.get("hard")}
        panel_by_id = {row["id"]: row.get("panel") for row in rows if row.get("hard")}
        for sample in analysis.get("samples") or []:
            sample_id = sample.get("id")
            if sample_id in clip_by_id:
                sample["clip"] = clip_by_id[sample_id]
            if sample_id in panel_by_id and panel_by_id[sample_id]:
                sample["panel"] = panel_by_id[sample_id]
            if sample_id in refreshed_ids:
                sample["media_version"] = "sol-boxes"
            if sample_id in sol_by_id:
                sample["sol_boxes"] = sol_by_id[sample_id]
        with open(analysis_path, "w") as handle:
            json.dump(analysis, handle, indent=2)
            handle.write("\n")

    with_boxes = sum(1 for row in hard_rows if row.get("sol_boxes"))
    print(
        "Wrote %d clips -> %s (%d with Sol boxes)" % (done, clips_dir, with_boxes),
        flush=True,
    )
    return {"n_clips": done, "clips_dir": clips_dir, "n_with_sol_boxes": with_boxes}
