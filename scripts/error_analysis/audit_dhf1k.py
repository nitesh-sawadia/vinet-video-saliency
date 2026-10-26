"""DHF1K-baseline disagreement audit on split_stride8 validation (last frame only).

Stages (each resumable, outputs under a separate analysis directory):

  score       DHF1K forward pass; KL / CC / SIM / peak distance / shape mismatch
              of the one prediction vs ``maps/0032.png``  -> dhf1k_scores.jsonl
  select      rank every window; keep all eligible ones for review
              -> ranking.jsonl, selection.json, candidates.jsonl, media/
  adjudicate  VLM verdict, then Sol box correction for gt_likely_wrong only
              -> vlm_results.jsonl, media/<id>_review.png
  index       review_index.json for ``python -m error_analysis review``

The split, pre_split and manifest are read-only. Human decisions live in
``human_reviews.json`` (written by the review server); re-export reads only
records whose decision is ``accept``.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from audit_sol import _default_pre_split_root, _manifest_by_rel, _read_jsonl, _write_jsonl
from features import frame_luma_stats
from review_server import REVIEW_DECISIONS
from viz import (
    build_last_frame_review_u8,
    draw_indexed_boxes_rgb,
    jet_overlay_u8,
    load_sol_native_targets,
)
from vlm import (
    DEFAULT_VLM_MODEL,
    adjudicate_dhf1k_disagreement,
    correct_sol_boxes,
    upright_person_boxes,
)

EXPECTED_SPLIT_NAME = "split_stride8"
RANK_COMPONENTS = (
    # (component, +1 when larger means more disagreement)
    ("kldiv", 1.0),
    ("cc", -1.0),
    ("sim", -1.0),
    ("peak_distance_norm", 1.0),
    ("shape_mismatch", 1.0),
)
LAST_FRAME_INDEX = 31


def check_audit_paths(split_root: str, out_dir: str, *, allow_other_split: bool = False) -> None:
    """Refuse non-stride8 splits and output dirs inside the dataset tree."""
    split_abs = os.path.abspath(split_root.rstrip(os.sep))
    out_abs = os.path.abspath(out_dir)
    if os.path.basename(split_abs) != EXPECTED_SPLIT_NAME and not allow_other_split:
        raise SystemExit(
            "audit-dhf1k targets %s validation only; got %s" % (EXPECTED_SPLIT_NAME, split_abs)
        )
    dataset_root = os.path.dirname(split_abs)
    for root in (split_abs, dataset_root):
        if out_abs == root or out_abs.startswith(root + os.sep):
            raise SystemExit(
                "--out %s is inside the dataset tree %s; write the audit elsewhere"
                % (out_abs, root)
            )


def _rank_normalize(values: list[float]) -> list[float]:
    """Average-rank percentile in [0, 1]; ties share a value. Deterministic."""
    n = len(values)
    if n == 1:
        return [0.5]
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg / (n - 1)
        i = j + 1
    return ranks


def _exclusion_reason(components: dict) -> str | None:
    for name, _sign in RANK_COMPONENTS:
        value = components.get(name)
        if value is None or not math.isfinite(float(value)):
            return "nonfinite_%s" % name
    if float(components.get("gt_mass", 1.0)) <= 1e-8:
        return "empty_gt"
    if float(components.get("pred_range", 1.0)) <= 1e-8:
        return "flat_prediction"
    return None


def rank_disagreements(
    rows: list[dict],
    *,
    top_percent: float = 100.0,
    min_candidates: int = 0,
    max_candidates: int | None = None,
) -> tuple[list[dict], dict]:
    """Rank windows on last-frame components only; mark candidates.

    Each component is rank-normalized across eligible windows (direction so
    1 = strongest disagreement) and averaged. By default every eligible window
    is a candidate. ``top_percent`` keeps only that share, ``min_candidates``
    raises the count, and ``max_candidates`` caps it. The threshold is the
    score of the last candidate.
    """
    if not (0.0 < top_percent <= 100.0):
        raise SystemExit("--top-percent must be in (0, 100]")
    if min_candidates < 0 or (max_candidates is not None and max_candidates < 1):
        raise SystemExit("--min-candidates must be >= 0 and --max-candidates >= 1")
    out = []
    eligible = []
    for row in rows:
        entry = {
            key: row.get(key)
            for key in ("id", "index", "path", "relpath", "sport", "game", "window", "pred_png")
        }
        entry["components"] = dict(row.get("components") or {})
        reason = _exclusion_reason(entry["components"])
        entry["eligible"] = reason is None
        entry["exclusion_reason"] = reason
        out.append(entry)
        if reason is None:
            eligible.append(entry)

    normalized = {}
    for name, sign in RANK_COMPONENTS:
        values = [sign * float(e["components"][name]) for e in eligible]
        normalized[name] = _rank_normalize(values) if values else []
    for i, entry in enumerate(eligible):
        entry["normalized"] = {
            name: round(normalized[name][i], 6) for name, _sign in RANK_COMPONENTS
        }
        entry["disagreement_score"] = round(
            sum(entry["normalized"].values()) / len(RANK_COMPONENTS), 6
        )
    eligible.sort(key=lambda e: (-e["disagreement_score"], str(e["id"])))

    n_eligible = len(eligible)
    n_percent = int(math.ceil(n_eligible * top_percent / 100.0)) if n_eligible else 0
    n_select = min(max(n_percent, min_candidates), n_eligible)
    if max_candidates is not None:
        n_select = min(n_select, max_candidates)
    for rank, entry in enumerate(eligible, start=1):
        entry["rank"] = rank
        entry["candidate"] = rank <= n_select
        if rank <= n_select:
            entry["selection_reason"] = "top_percent" if rank <= n_percent else "min_candidates"
        elif rank <= n_percent:
            entry["selection_reason"] = "max_candidates_cap"
        else:
            entry["selection_reason"] = "below_threshold"
    for entry in out:
        if not entry["eligible"]:
            entry.update(
                {
                    "rank": None,
                    "candidate": False,
                    "selection_reason": "excluded",
                    "disagreement_score": None,
                    "normalized": None,
                }
            )
    threshold = eligible[n_select - 1]["disagreement_score"] if n_select else None
    selection = {
        "rank_components": [name for name, _sign in RANK_COMPONENTS],
        "top_percent": top_percent,
        "min_candidates": min_candidates,
        "max_candidates": max_candidates,
        "n_windows": len(out),
        "n_eligible": n_eligible,
        "n_excluded": len(out) - n_eligible,
        "n_top_percent": n_percent,
        "n_candidates": n_select,
        "candidate_threshold": threshold,
    }
    ranked = eligible + sorted(
        (e for e in out if not e["eligible"]), key=lambda e: str(e["id"])
    )
    return ranked, selection


def _resolve_window(entry: dict, split_root: str | None) -> Path:
    if split_root and entry.get("relpath"):
        return Path(split_root) / entry["relpath"]
    return Path(entry.get("path") or "")


def _load_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(str(path))
    return image.astype(np.float32) / 255.0


def _build_candidate_media(
    entry: dict,
    out_dir: str,
    *,
    split_root: str | None,
    pre_split_root: str | None,
    man: dict,
) -> dict:
    """Last-frame media only: frame, Sol boxes, GT overlay, DHF1K overlay."""
    window = _resolve_window(entry, split_root)
    frame_path = window / "images" / "0032.png"
    map_path = window / "maps" / "0032.png"
    frame = np.asarray(Image.open(frame_path).convert("RGB"), dtype=np.uint8)
    gt = _load_gray(map_path)
    pred = _load_gray(Path(out_dir) / entry["pred_png"])
    sol_all = load_sol_native_targets(
        window,
        pre_split_root=pre_split_root,
        source=man.get("source"),
        native_start=man.get("native_start"),
        native_end=man.get("native_end"),
    )
    sol_last = list(sol_all[LAST_FRAME_INDEX]) if len(sol_all) > LAST_FRAME_INDEX else []

    media_dir = os.path.join(out_dir, "media")
    os.makedirs(media_dir, exist_ok=True)
    sample_id = entry["id"]
    rels = {
        "frame": "media/%s_frame.png" % sample_id,
        "sol": "media/%s_sol.png" % sample_id,
        "gt_map": "media/%s_gtmap.png" % sample_id,
        "gt": "media/%s_gt.png" % sample_id,
        "pred": "media/%s_pred.png" % sample_id,
        "review": "media/%s_review.png" % sample_id,
    }
    Image.fromarray(frame).save(os.path.join(out_dir, rels["frame"]))
    cv2.imwrite(
        os.path.join(out_dir, rels["gt_map"]), np.round(np.clip(gt, 0, 1) * 255).astype(np.uint8)
    )
    Image.fromarray(draw_indexed_boxes_rgb(frame, sol_last)).save(
        os.path.join(out_dir, rels["sol"])
    )
    Image.fromarray(jet_overlay_u8(frame, gt)).save(os.path.join(out_dir, rels["gt"]))
    Image.fromarray(jet_overlay_u8(frame, pred)).save(os.path.join(out_dir, rels["pred"]))
    Image.fromarray(
        build_last_frame_review_u8(frame, gt, pred, sol_targets=sol_last)
    ).save(os.path.join(out_dir, rels["review"]))

    luma_mean, luma_std = frame_luma_stats(frame)
    return {
        "media": rels,
        "frame_size": [int(frame.shape[1]), int(frame.shape[0])],
        "sol_targets_last": sol_last,
        "sol_boxes_present": bool(sol_last),
        "context": {
            "frame_luma_mean": round(luma_mean, 4),
            "frame_luma_std": round(luma_std, 4),
            "low_visibility": bool(luma_mean < 0.22 or luma_std < 0.08),
        },
    }


def select_candidates(
    out_dir: str,
    *,
    split_root: str | None = None,
    pre_split_root: str | None = None,
    top_percent: float = 100.0,
    min_candidates: int = 0,
    max_candidates: int | None = None,
    resume: bool = True,
) -> dict:
    scores_path = os.path.join(out_dir, "dhf1k_scores.jsonl")
    if not os.path.isfile(scores_path):
        raise SystemExit("Missing dhf1k_scores.jsonl in %s; run --stages score first" % out_dir)
    if split_root is None:
        with open(os.path.join(out_dir, "audit_manifest.json")) as handle:
            split_root = json.load(handle).get("split_root")
    if split_root:
        check_audit_paths(split_root, out_dir, allow_other_split=True)
    if pre_split_root is None:
        pre_split_root = _default_pre_split_root(split_root)
    man_by_rel = _manifest_by_rel(split_root) if split_root else {}

    ranked, selection = rank_disagreements(
        _read_jsonl(scores_path),
        top_percent=top_percent,
        min_candidates=min_candidates,
        max_candidates=max_candidates,
    )
    selection_path = os.path.join(out_dir, "selection.json")
    if resume and os.path.isfile(selection_path):
        with open(selection_path) as handle:
            prior = json.load(handle)
        for key in ("top_percent", "min_candidates", "max_candidates", "n_windows"):
            if prior.get(key) != selection.get(key):
                raise SystemExit(
                    "selection.json %s=%r differs from this run (%r); pass --no-resume "
                    "or use a new --out" % (key, prior.get(key), selection.get(key))
                )
    _write_jsonl(os.path.join(out_dir, "ranking.jsonl"), ranked)

    prior_candidates = {}
    cand_path = os.path.join(out_dir, "candidates.jsonl")
    if resume and os.path.isfile(cand_path):
        prior_candidates = {str(r["id"]): r for r in _read_jsonl(cand_path)}
    candidates = []
    for entry in ranked:
        if not entry.get("candidate"):
            continue
        prior = prior_candidates.get(str(entry["id"]))
        if prior and all(
            os.path.isfile(os.path.join(out_dir, rel)) for rel in (prior.get("media") or {}).values()
        ):
            candidates.append(prior)
            continue
        man = man_by_rel.get(str(entry.get("relpath") or "")) or {}
        try:
            entry = {**entry, **_build_candidate_media(
                entry, out_dir, split_root=split_root, pre_split_root=pre_split_root, man=man
            )}
        except Exception as exc:
            print("media failed for %s: %s" % (entry["id"], exc), flush=True)
            entry = {**entry, "media_error": str(exc)}
        candidates.append(entry)
    _write_jsonl(cand_path, candidates)

    selection["split_root"] = split_root
    selection["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(selection_path, "w") as handle:
        json.dump(selection, handle, indent=2)
        handle.write("\n")
    print(
        "Selected %d/%d eligible windows (threshold=%s)"
        % (selection["n_candidates"], selection["n_eligible"], selection["candidate_threshold"]),
        flush=True,
    )
    return selection


def _open_rgb(out_dir: str, rel: str) -> Image.Image:
    return Image.open(os.path.join(out_dir, rel)).convert("RGB")


def accept_allowed(verdict: dict | None, correction: dict | None) -> bool:
    return bool(
        verdict
        and verdict.get("status") == "ok"
        and verdict.get("disposition") == "gt_likely_wrong"
        and correction
        and correction.get("status") == "valid"
        and correction.get("detections")
    )


def adjudicate_candidates(
    out_dir: str,
    *,
    vlm_model: str = DEFAULT_VLM_MODEL,
    limit: int | None = None,
    ids: list[str] | None = None,
    resume: bool = True,
    transport=None,
) -> dict:
    """Verdict call per candidate; Sol correction call only for gt_likely_wrong."""
    cand_path = os.path.join(out_dir, "candidates.jsonl")
    if not os.path.isfile(cand_path):
        raise SystemExit("Missing candidates.jsonl in %s; run --stages select first" % out_dir)
    candidates = [c for c in _read_jsonl(cand_path) if c.get("media")]
    candidates.sort(key=lambda c: c.get("rank") or 0)
    if ids:
        want = set(ids)
        candidates = [c for c in candidates if str(c["id"]) in want]
    if limit is not None:
        candidates = candidates[:limit]

    results_path = os.path.join(out_dir, "vlm_results.jsonl")
    done: dict[str, dict] = {}
    if os.path.isfile(results_path):
        done = {str(r["id"]): r for r in _read_jsonl(results_path)}

    tic = time.time()
    for i, cand in enumerate(candidates):
        sample_id = str(cand["id"])
        prior = done.get(sample_id)
        if resume and prior and (prior.get("verdict") or {}).get("status") == "ok":
            print("[%d/%d] %s resume" % (i + 1, len(candidates), sample_id), flush=True)
            continue
        media = cand["media"]
        sol_targets = cand.get("sol_targets_last") or []
        sol_image = _open_rgb(out_dir, media["sol"])
        verdict = adjudicate_dhf1k_disagreement(
            cand,
            sol_image=sol_image,
            gt_overlay=_open_rgb(out_dir, media["gt"]),
            pred_overlay=_open_rgb(out_dir, media["pred"]),
            sol_targets=sol_targets,
            model=vlm_model,
            transport=transport,
        )
        if verdict.get("status") == "ok" and verdict.get("disposition") == "gt_likely_wrong":
            correction = correct_sol_boxes(
                cand,
                sol_image=sol_image,
                sol_targets=sol_targets,
                model=vlm_model,
                transport=transport,
            )
        else:
            correction = {"status": "not_requested", "detections": []}

        proposed_rel = None
        if correction.get("status") == "valid":
            frame = np.asarray(_open_rgb(out_dir, media["frame"]), dtype=np.uint8)
            proposed_rel = "media/%s_proposed.png" % sample_id
            Image.fromarray(draw_indexed_boxes_rgb(frame, correction["detections"])).save(
                os.path.join(out_dir, proposed_rel)
            )
        if verdict.get("status") == "ok":
            frame = np.asarray(_open_rgb(out_dir, media["frame"]), dtype=np.uint8)
            Image.fromarray(
                build_last_frame_review_u8(
                    frame,
                    _load_gray(Path(out_dir) / media["gt_map"]),
                    _load_gray(Path(out_dir) / cand["pred_png"]),
                    sol_targets=sol_targets,
                    proposed_targets=(
                        correction["detections"] if correction.get("status") == "valid" else None
                    ),
                    verdict="%s (%.2f)" % (verdict["disposition"], verdict["confidence"]),
                )
            ).save(os.path.join(out_dir, media["review"]))

        entry = {
            "id": sample_id,
            "rank": cand.get("rank"),
            "relpath": cand.get("relpath"),
            "model": vlm_model,
            "verdict": verdict,
            "correction": correction,
            "proposed_media": proposed_rel,
            "accept_allowed": accept_allowed(verdict, correction),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        done[sample_id] = entry
        _write_jsonl(results_path, list(done.values()))
        print(
            "[%d/%d] %s -> %s correction=%s"
            % (
                i + 1,
                len(candidates),
                sample_id,
                verdict.get("disposition") or verdict.get("status"),
                correction.get("status"),
            ),
            flush=True,
        )

    counts: dict[str, int] = {}
    for row in done.values():
        key = (row.get("verdict") or {}).get("disposition") or (row.get("verdict") or {}).get(
            "status", "unknown"
        )
        counts[key] = counts.get(key, 0) + 1
    report = {
        "n_results": len(done),
        "dispositions": counts,
        "n_accept_allowed": sum(1 for r in done.values() if r.get("accept_allowed")),
        "model": vlm_model,
        "elapsed_sec": time.time() - tic,
    }
    with open(os.path.join(out_dir, "vlm_summary.json"), "w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def build_review_index(out_dir: str) -> dict:
    cand_path = os.path.join(out_dir, "candidates.jsonl")
    if not os.path.isfile(cand_path):
        raise SystemExit("Missing candidates.jsonl in %s" % out_dir)
    candidates = _read_jsonl(cand_path)
    results_path = os.path.join(out_dir, "vlm_results.jsonl")
    results = (
        {str(r["id"]): r for r in _read_jsonl(results_path)}
        if os.path.isfile(results_path)
        else {}
    )
    selection = {}
    if os.path.isfile(os.path.join(out_dir, "selection.json")):
        with open(os.path.join(out_dir, "selection.json")) as handle:
            selection = json.load(handle)
    samples = []
    repaired = 0
    for cand in sorted(candidates, key=lambda c: c.get("rank") or 0):
        result = results.get(str(cand["id"])) or {}
        verdict = {k: v for k, v in (result.get("verdict") or {}).items() if k != "raw"}
        correction = {k: v for k, v in (result.get("correction") or {}).items() if k != "raw"}
        if correction.get("status") == "valid" and correction.get("detections"):
            fixed = upright_person_boxes(correction["detections"])
            if fixed != correction["detections"]:
                correction["detections"] = fixed
                result["correction"]["detections"] = fixed
                proposed_rel = result.get("proposed_media")
                if proposed_rel:
                    frame = np.asarray(_open_rgb(out_dir, cand["media"]["frame"]), dtype=np.uint8)
                    Image.fromarray(draw_indexed_boxes_rgb(frame, fixed)).save(
                        os.path.join(out_dir, proposed_rel)
                    )
                repaired += 1
        samples.append(
            {
                "id": cand["id"],
                "rank": cand.get("rank"),
                "sport": cand.get("sport"),
                "game": cand.get("game"),
                "window": cand.get("window"),
                "relpath": cand.get("relpath"),
                "disagreement_score": cand.get("disagreement_score"),
                "components": cand.get("components"),
                "normalized": cand.get("normalized"),
                "selection_reason": cand.get("selection_reason"),
                "context": cand.get("context"),
                "media": cand.get("media"),
                "media_error": cand.get("media_error"),
                "proposed_media": result.get("proposed_media"),
                "sol_targets_last": cand.get("sol_targets_last") or [],
                "verdict": verdict or None,
                "correction": correction or None,
                "accept_allowed": accept_allowed(verdict, correction),
            }
        )
    if repaired:
        _write_jsonl(results_path, list(results.values()))
        print("Corrected swapped axes in %d saved VLM corrections" % repaired, flush=True)
    index = {
        "kind": "dhf1k_audit",
        "selection": selection,
        "decisions": list(REVIEW_DECISIONS),
        "samples": samples,
    }
    with open(os.path.join(out_dir, "review_index.json"), "w") as handle:
        json.dump(index, handle, indent=2)
        handle.write("\n")
    print("Wrote review_index.json with %d candidates" % len(samples), flush=True)
    return index


def load_human_reviews(out_dir: str) -> dict:
    path = os.path.join(out_dir, "human_reviews.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as handle:
        return json.load(handle)


def accepted_corrections(out_dir: str) -> list[dict]:
    """Records whose human decision is ``accept`` — the only re-export input."""
    return [
        record
        for _sid, record in sorted(load_human_reviews(out_dir).items())
        if record.get("decision") == "accept" and record.get("accepted_boxes")
    ]


def export_accepted(out_dir: str) -> str:
    """Write accepted_corrections.jsonl in the audit dir (dataset untouched)."""
    rows = accepted_corrections(out_dir)
    path = os.path.join(out_dir, "accepted_corrections.jsonl")
    _write_jsonl(path, rows)
    print("Wrote %d accepted corrections -> %s" % (len(rows), path), flush=True)
    return path


def run_audit_dhf1k(
    out_dir: str,
    *,
    stages: list[str],
    checkpoint_path: str | None = None,
    split_root: str | None = None,
    pre_split_root: str | None = None,
    batch_size: int = 16,
    num_workers: int = 4,
    device_name: str | None = None,
    top_percent: float = 100.0,
    min_candidates: int = 0,
    max_candidates: int | None = None,
    vlm_model: str = DEFAULT_VLM_MODEL,
    limit: int | None = None,
    ids: list[str] | None = None,
    resume: bool = True,
    allow_other_split: bool = False,
) -> None:
    if split_root:
        check_audit_paths(split_root, out_dir, allow_other_split=allow_other_split)
    if "score" in stages:
        if not checkpoint_path or not split_root:
            raise SystemExit("--stages score needs --checkpoint and --split-root")
        from score import score_dhf1k_baseline

        score_dhf1k_baseline(
            checkpoint_path,
            split_root,
            out_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            device_name=device_name,
            resume=resume,
        )
    if "select" in stages:
        select_candidates(
            out_dir,
            split_root=split_root,
            pre_split_root=pre_split_root,
            top_percent=top_percent,
            min_candidates=min_candidates,
            max_candidates=max_candidates,
            resume=resume,
        )
    if "adjudicate" in stages:
        adjudicate_candidates(out_dir, vlm_model=vlm_model, limit=limit, ids=ids, resume=resume)
    if "index" in stages:
        build_review_index(out_dir)
