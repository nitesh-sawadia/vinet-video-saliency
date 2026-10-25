"""Score every outplay val window with a training-bundle checkpoint.

Inference is batched: DataLoader batches -> one model forward per batch.
Per-sample metrics are derived from that batched forward without a second
pass. Hard-example panels reuse cached last-frame / pred maps (no re-infer).
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def _repo_vinet_s() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ViNet_S"))


def _ensure_vinet_path() -> None:
    root = _repo_vinet_s()
    if root not in sys.path:
        sys.path.insert(0, root)


def load_model(checkpoint_path: str, device: torch.device):
    _ensure_vinet_path()
    from ViNet_S_model import VideoSaliencyModel

    model = VideoSaliencyModel(
        use_upsample=True,
        num_hier=3,
        num_clips=32,
        grouped_conv=True,
        root_grouping=True,
        depth=False,
        efficientnet=False,
        BiCubic=True,
    )
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


def _path_meta(path: str, split_root: str) -> dict:
    rel = os.path.relpath(path, split_root)
    parts = rel.replace("\\", "/").split("/")
    sport = parts[1] if len(parts) > 1 else ""
    game = parts[2] if len(parts) > 2 else ""
    window = parts[3] if len(parts) > 3 else os.path.basename(path)
    return {
        "path": path,
        "relpath": rel.replace("\\", "/"),
        "sport": sport,
        "game": game,
        "window": window,
    }


def _squeeze_pred(pred: torch.Tensor) -> torch.Tensor:
    """Model output -> (B, H, W)."""
    if pred.dim() == 4 and pred.shape[1] == 1:
        return pred[:, 0]
    if pred.dim() == 3:
        return pred
    raise RuntimeError("Unexpected pred shape %s" % (tuple(pred.shape),))


def _resize_pred_to_gt(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """pred/gt are (B, H, W)."""
    if pred.shape[-2:] == gt.shape[-2:]:
        return pred
    return F.interpolate(
        pred.unsqueeze(1),
        size=gt.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )[:, 0]


def _kldiv_per_sample(s_map: torch.Tensor, gt: torch.Tensor, eps: float = 2.2204e-16) -> torch.Tensor:
    batch, height, width = s_map.shape
    s_flat = s_map.reshape(batch, -1)
    g_flat = gt.reshape(batch, -1)
    s_prob = s_flat / s_flat.sum(dim=1, keepdim=True).clamp_min(eps)
    g_prob = g_flat / g_flat.sum(dim=1, keepdim=True).clamp_min(eps)
    s_prob = s_prob.view(batch, height, width)
    g_prob = g_prob.view(batch, height, width)
    return (g_prob * torch.log(eps + g_prob / (s_prob + eps))).reshape(batch, -1).sum(dim=1)


def _cc_per_sample(s_map: torch.Tensor, gt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    batch = s_map.shape[0]
    s_flat = s_map.reshape(batch, -1)
    g_flat = gt.reshape(batch, -1)
    s_flat = (s_flat - s_flat.mean(dim=1, keepdim=True)) / s_flat.std(dim=1, keepdim=True).clamp_min(eps)
    g_flat = (g_flat - g_flat.mean(dim=1, keepdim=True)) / g_flat.std(dim=1, keepdim=True).clamp_min(eps)
    ab = (s_flat * g_flat).sum(dim=1)
    aa = (s_flat * s_flat).sum(dim=1)
    bb = (g_flat * g_flat).sum(dim=1)
    return ab / torch.sqrt(aa * bb).clamp_min(eps)


def _release_cuda_after_inference(model, loader=None) -> None:
    """Drop the model / dataloader and return VRAM to the driver."""
    try:
        if model is not None:
            model.cpu()
    except Exception:
        pass
    del model
    if loader is not None:
        # Break persistent worker processes before dropping the loader.
        try:
            loader._iterator = None  # type: ignore[attr-defined]
        except Exception:
            pass
        del loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        try:
            free_b, total_b = torch.cuda.mem_get_info()
            print(
                "Released CUDA after inference (free=%.1f / total=%.1f GiB)"
                % (free_b / (1024 ** 3), total_b / (1024 ** 3)),
                flush=True,
            )
        except Exception:
            print("Released CUDA cache after inference", flush=True)


def _sim_per_sample(s_map: torch.Tensor, gt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    batch = s_map.shape[0]

    def _norm(x):
        flat = x.reshape(batch, -1)
        lo = flat.min(dim=1, keepdim=True).values
        hi = flat.max(dim=1, keepdim=True).values
        scaled = (flat - lo) / (hi - lo).clamp_min(eps)
        return (scaled / scaled.sum(dim=1, keepdim=True).clamp_min(eps)).view_as(x)

    s_n = _norm(s_map)
    g_n = _norm(gt)
    return torch.min(s_n, g_n).reshape(batch, -1).sum(dim=1)


def _last_frame_dataset_cls():
    """Dataset class built lazily so ViNet_S is only imported when scoring."""
    _ensure_vinet_path()
    from torch.utils.data import Dataset
    import outplay_dataset as od

    class LastFrameGtWindows(Dataset):
        """Val windows -> (index, 32-frame clip, maps/0032.png only).

        The other 31 maps are never opened: the audit compares one prediction
        with the last-frame GT the training loss reads.
        """

        def __init__(self, split_root: str, split: str = "val"):
            manifest_path = os.path.join(split_root, "split_manifest.json")
            if not os.path.isfile(manifest_path):
                raise FileNotFoundError("Outplay manifest not found: %s" % manifest_path)
            with open(manifest_path) as handle:
                manifest = json.load(handle)
            windows = manifest.get("windows") if isinstance(manifest, dict) else None
            if not isinstance(windows, list):
                raise ValueError("Outplay manifest has no windows list: %s" % manifest_path)
            self.paths = [
                od.window_path(split_root, window)
                for window in windows
                if isinstance(window, dict) and window.get("split") == split
            ]
            self.last_map_name = od.frame_names()[-1]

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, index):
            path = self.paths[index]
            frames = []
            for name in od.frame_names():
                frame_path = os.path.join(path, "images", name)
                if not os.path.isfile(frame_path):
                    raise FileNotFoundError("Outplay window is missing %s" % frame_path)
                frames.append(od._load_frame(frame_path))
            map_path = os.path.join(path, "maps", self.last_map_name)
            if not os.path.isfile(map_path):
                raise FileNotFoundError("Outplay window is missing %s" % map_path)
            clip = torch.stack(frames, dim=0).float()
            gt = od._load_map(map_path).float()
            return index, clip, gt

    return LastFrameGtWindows


def _sha256_file(path: str) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str | None:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def _save_pred_png(path: str, pred: np.ndarray) -> None:
    low, high = float(pred.min()), float(pred.max())
    scaled = (pred - low) / max(high - low, 1e-8)
    cv2.imwrite(path, np.round(scaled * 255.0).astype(np.uint8))


def score_dhf1k_baseline(
    checkpoint_path: str,
    split_root: str,
    out_dir: str,
    *,
    batch_size: int = 16,
    num_workers: int = 4,
    device_name: str | None = None,
    resume: bool = True,
    model=None,
) -> dict:
    """Score DHF1K on every val window against maps/0032.png only.

    Writes ``dhf1k_scores.jsonl`` (KL, CC, SIM, peak distance, shape mismatch
    per window) and ``preds/<id>.png`` (min-max uint8 prediction). Rows are
    appended so an interrupted run resumes at the first unscored window.
    ``model`` lets tests inject a stand-in network.
    """
    from torch.utils.data import Subset

    from features import last_frame_disagreement

    if batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(out_dir, exist_ok=True)
    preds_dir = os.path.join(out_dir, "preds")
    os.makedirs(preds_dir, exist_ok=True)
    scores_path = os.path.join(out_dir, "dhf1k_scores.jsonl")
    provenance_path = os.path.join(out_dir, "audit_manifest.json")

    manifest_path = os.path.join(split_root, "split_manifest.json")
    provenance = {
        "kind": "dhf1k_disagreement_audit",
        "checkpoint": os.path.abspath(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "split_root": os.path.abspath(split_root),
        "split": "val",
        "split_manifest_sha256": _sha256_file(manifest_path),
        "gt_map": "maps/0032.png",
        "git_head": _git_head(),
    }
    if os.path.isfile(provenance_path):
        with open(provenance_path) as handle:
            prior = json.load(handle)
        for key in ("checkpoint_sha256", "split_root", "split_manifest_sha256"):
            if prior.get(key) != provenance[key]:
                raise SystemExit(
                    "%s in %s does not match this run (%s); use a new --out"
                    % (key, provenance_path, provenance[key])
                )
        provenance = prior
    elif os.path.isfile(scores_path):
        raise SystemExit("%s exists without audit_manifest.json; use a new --out" % scores_path)

    done: set[str] = set()
    if resume and os.path.isfile(scores_path):
        with open(scores_path) as handle:
            for line in handle:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["relpath"])
    elif os.path.isfile(scores_path):
        os.remove(scores_path)
        done = set()

    dataset = _last_frame_dataset_cls()(split_root, "val")
    if not len(dataset):
        raise SystemExit("No val windows in %s" % manifest_path)
    todo = [
        i
        for i, path in enumerate(dataset.paths)
        if _path_meta(path, split_root)["relpath"] not in done
    ]
    print(
        "DHF1K audit: val windows=%d already scored=%d remaining=%d"
        % (len(dataset), len(dataset) - len(todo), len(todo)),
        flush=True,
    )

    ckpt_meta = provenance.get("checkpoint_meta") or {}
    if todo:
        if model is None:
            print("Loading checkpoint on %s ..." % device, flush=True)
            model, ckpt_meta = load_model(checkpoint_path, device)
        loader = DataLoader(
            Subset(dataset, todo),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
        tic = time.time()
        scored = 0
        with open(scores_path, "a") as handle, torch.no_grad():
            for batch_i, (indices, clips, gts) in enumerate(loader):
                clips = clips.to(device, non_blocking=True)
                gts = gts.to(device, non_blocking=True)
                preds = _squeeze_pred(model(clips.permute(0, 2, 1, 3, 4)))
                preds = _resize_pred_to_gt(preds, gts)
                preds_cpu = preds.detach().float().cpu().numpy()
                gts_cpu = gts.detach().float().cpu().numpy()
                for b in range(int(clips.shape[0])):
                    index = int(indices[b])
                    meta = _path_meta(dataset.paths[index], split_root)
                    sample_id = "%05d_%s" % (index, meta["window"])
                    pred_rel = "preds/%s.png" % sample_id
                    _save_pred_png(os.path.join(out_dir, pred_rel), preds_cpu[b])
                    row = {
                        "id": sample_id,
                        "index": index,
                        **meta,
                        "components": last_frame_disagreement(preds_cpu[b], gts_cpu[b]),
                        "pred_png": pred_rel,
                    }
                    handle.write(json.dumps(row) + "\n")
                    scored += 1
                handle.flush()
                if (batch_i + 1) % 5 == 0 or scored == len(todo):
                    print(
                        "scored %d/%d  elapsed=%.1fs" % (scored, len(todo), time.time() - tic),
                        flush=True,
                    )
        _release_cuda_after_inference(model, loader)

    provenance["checkpoint_meta"] = ckpt_meta
    provenance["n_val"] = len(dataset)
    with open(provenance_path, "w") as handle:
        json.dump(provenance, handle, indent=2)
        handle.write("\n")
    return {"scores_path": scores_path, "n_val": len(dataset), "n_scored_now": len(todo)}


def score_validation(
    checkpoint_path: str,
    split_root: str,
    out_dir: str,
    batch_size: int = 16,
    num_workers: int = 4,
    device_name: str | None = None,
    hard_fraction: float = 0.12,
    hard_min: int = 40,
    hard_max: int = 80,
) -> dict:
    """Write scores.jsonl + summary.json. Returns summary dict."""
    from features import compute_map_features
    from viz import denormalize_frame_chw, save_panel, save_clip_video, load_rgb_u8

    _ensure_vinet_path()
    from outplay_dataset import OutplayWindowDataset, frame_names

    if batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")

    device = torch.device(
        device_name
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    os.makedirs(out_dir, exist_ok=True)
    panels_dir = os.path.join(out_dir, "panels")
    os.makedirs(panels_dir, exist_ok=True)

    print("Loading checkpoint on %s ..." % device, flush=True)
    model, ckpt_meta = load_model(checkpoint_path, device)
    print("Checkpoint meta:", ckpt_meta, flush=True)
    print("Building val dataset (validate=False for mount speed) ...", flush=True)
    dataset = OutplayWindowDataset(split_root, "val", clip_size=32, validate=False)
    print(
        "Val windows: %d  batch_size=%d  num_workers=%d"
        % (len(dataset), batch_size, num_workers),
        flush=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    scores_path = os.path.join(out_dir, "scores.jsonl")
    rows = []
    # Cache pred maps + last RGB frame so hard panels need no second forward.
    pred_cache: dict[int, np.ndarray] = {}
    frame_cache: dict[int, np.ndarray] = {}
    tic = time.time()
    with open(scores_path, "w") as handle, torch.no_grad():
        index = 0
        for batch_i, batch in enumerate(loader):
            clips, gts, _all_maps = batch
            bsz = int(clips.shape[0])
            clips = clips.to(device, non_blocking=True)
            gts = gts.to(device, non_blocking=True)
            preds = _squeeze_pred(model(clips.permute(0, 2, 1, 3, 4)))
            preds = _resize_pred_to_gt(preds, gts)

            kl = _kldiv_per_sample(preds, gts)
            cc_v = _cc_per_sample(preds, gts)
            sim_v = _sim_per_sample(preds, gts)
            loss_v = kl - cc_v

            preds_cpu = preds.detach().cpu().numpy()
            gts_cpu = gts.detach().cpu().numpy()
            last_frames = clips[:, -1].detach().cpu().numpy()

            for b in range(bsz):
                path = dataset.paths[index]
                meta = _path_meta(path, split_root)
                pred_np = preds_cpu[b]
                gt_np = gts_cpu[b]
                last_frame = last_frames[b]
                last_rgb = (
                    np.transpose(denormalize_frame_chw(last_frame), (1, 2, 0)) * 255.0
                ).astype(np.uint8)
                feats = compute_map_features(
                    pred_np, gt_np, frame_rgb=last_rgb, clip_frames=None
                )
                sample_id = "%05d_%s" % (index, meta["window"])
                row = {
                    "id": sample_id,
                    "index": index,
                    **meta,
                    "loss": float(loss_v[b].item()),
                    "kldiv": float(kl[b].item()),
                    "cc": float(cc_v[b].item()),
                    "sim": float(sim_v[b].item()),
                    "features": feats,
                    "panel": None,
                }
                rows.append(row)
                handle.write(json.dumps(row) + "\n")
                pred_cache[index] = pred_np
                frame_cache[index] = last_frame
                index += 1

            if (batch_i + 1) % 5 == 0 or index == len(dataset):
                print(
                    "scored %d/%d  batches=%d  elapsed=%.1fs"
                    % (index, len(dataset), batch_i + 1, time.time() - tic),
                    flush=True,
                )

    # Panels use CPU caches only — free the GPU before VLM classify / review.
    _release_cuda_after_inference(model, loader)
    model = None
    loader = None

    ranked = sorted(rows, key=lambda r: r["loss"], reverse=True)
    n_hard = int(round(len(rows) * hard_fraction))
    n_hard = max(hard_min, min(hard_max, n_hard, len(rows)))
    hard_ids = {row["id"] for row in ranked[:n_hard]}

    print(
        "Writing %d hard-example panels + 32-frame clips from cached preds..."
        % n_hard,
        flush=True,
    )
    clips_dir = os.path.join(out_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    for row in ranked[:n_hard]:
        index = row["index"]
        pred = pred_cache[index]
        # Full clip reload for the 32-frame video (pred stays cached).
        clip_t, gt_t, all_maps = dataset[index]
        clip_np = clip_t.numpy()
        gt_np = gt_t.numpy()
        maps_np = all_maps.numpy()
        last_frame = frame_cache[index]

        panel_rel = os.path.join("panels", "%s.png" % row["id"])
        panel_abs = os.path.join(out_dir, panel_rel)
        save_panel(panel_abs, last_frame, gt_np, pred)
        clip_abs = save_clip_video(
            os.path.join(clips_dir, "%s.webp" % row["id"]),
            clip_np,
            maps_np,
            pred,
            fps=8,
        )
        clip_rel = os.path.relpath(clip_abs, out_dir).replace("\\", "/")
        clip_rgbs = [
            load_rgb_u8(os.path.join(row["path"], "images", name))
            for name in frame_names()[::4]
        ]
        feats = dict(row.get("features") or {})
        feats.update(
            compute_map_features(
                pred,
                gt_np,
                frame_rgb=(
                    np.transpose(denormalize_frame_chw(last_frame), (1, 2, 0)) * 255.0
                ).astype(np.uint8),
                clip_frames=clip_rgbs,
                gt_maps_thw=maps_np,
            )
        )
        row["features"] = feats
        row["panel"] = panel_rel
        row["clip"] = clip_rel
        row["hard"] = True

    for row in rows:
        row["hard"] = row["id"] in hard_ids

    with open(scores_path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    summary = {
        "checkpoint": checkpoint_path,
        "checkpoint_meta": ckpt_meta,
        "split_root": split_root,
        "n_val": len(rows),
        "n_hard": n_hard,
        "batch_size": batch_size,
        "device": str(device),
        "mean_loss": float(np.mean([r["loss"] for r in rows])),
        "mean_cc": float(np.mean([r["cc"] for r in rows])),
        "mean_sim": float(np.mean([r["sim"] for r in rows])),
        "mean_kldiv": float(np.mean([r["kldiv"] for r in rows])),
        "hard_ids": sorted(hard_ids),
        "scores_path": scores_path,
        "elapsed_sec": time.time() - tic,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(
        "Done. n=%d mean_loss=%.4f mean_cc=%.4f hard=%d batch_size=%d -> %s"
        % (
            summary["n_val"],
            summary["mean_loss"],
            summary["mean_cc"],
            n_hard,
            batch_size,
            out_dir,
        ),
        flush=True,
    )
    return summary
