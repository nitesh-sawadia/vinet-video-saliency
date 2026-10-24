"""Propose + VLM-confirm hypotheses for hard examples; write buckets."""

from __future__ import annotations

import json
import os
import time

from hypotheses import HYPOTHESES, HYPOTHESIS_BY_ID, propose_from_features
from vlm import confirm_hypotheses


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def classify_hard_examples(
    out_dir: str,
    *,
    vlm_model: str = "vertex_ai/gemini-2.5-flash",
    skip_vlm: bool = False,
    limit: int | None = None,
) -> dict:
    scores_path = os.path.join(out_dir, "scores.jsonl")
    if not os.path.isfile(scores_path):
        raise SystemExit("Missing scores.jsonl in %s; run score first" % out_dir)

    rows = _read_jsonl(scores_path)
    hard = [row for row in rows if row.get("hard")]
    hard.sort(key=lambda r: r["loss"], reverse=True)
    if limit is not None:
        hard = hard[:limit]

    analyses = []
    buckets: dict[str, list[str]] = {h["id"]: [] for h in HYPOTHESES}
    tic = time.time()

    for i, sample in enumerate(hard):
        proposals = propose_from_features(sample.get("features") or {})
        panel_rel = sample.get("panel")
        panel_abs = os.path.join(out_dir, panel_rel) if panel_rel else None
        vlm_result = None
        confirmed = []

        if skip_vlm:
            confirmed = [
                {
                    "id": p["id"],
                    "confidence": float(p["confidence"]),
                    "rationale": "geometric only (VLM skipped): %s" % p.get("evidence", ""),
                }
                for p in proposals
                if float(p["confidence"]) >= 0.55
            ][:2]
        elif panel_abs and os.path.isfile(panel_abs):
            try:
                vlm_result = confirm_hypotheses(
                    panel_abs,
                    sample,
                    proposals,
                    model=vlm_model,
                )
                confirmed = vlm_result["confirmed"]
            except Exception as exc:
                print("VLM failed for %s: %s" % (sample["id"], exc), flush=True)
                confirmed = [
                    {
                        "id": p["id"],
                        "confidence": float(p["confidence"]) * 0.7,
                        "rationale": "VLM failed; geometric fallback: %s" % p.get("evidence", ""),
                    }
                    for p in proposals[:1]
                ]
        else:
            confirmed = [
                {
                    "id": p["id"],
                    "confidence": float(p["confidence"]) * 0.7,
                    "rationale": "no panel; geometric fallback: %s" % p.get("evidence", ""),
                }
                for p in proposals[:1]
            ]

        if not confirmed:
            confirmed = [
                {
                    "id": "unexplained",
                    "confidence": 0.5,
                    "rationale": "No geometric or VLM hypothesis confirmed.",
                }
            ]

        primary = confirmed[0]["id"]
        if primary not in buckets:
            primary = "unexplained"
            confirmed = [
                {
                    "id": "unexplained",
                    "confidence": 0.5,
                    "rationale": "Unknown id from VLM; marked unexplained.",
                }
            ]

        entry = {
            "id": sample["id"],
            "index": sample["index"],
            "sport": sample.get("sport"),
            "game": sample.get("game"),
            "window": sample.get("window"),
            "relpath": sample.get("relpath"),
            "loss": sample["loss"],
            "cc": sample["cc"],
            "kldiv": sample["kldiv"],
            "sim": sample["sim"],
            "panel": sample.get("panel"),
            "features": sample.get("features"),
            "proposals": proposals,
            "confirmed": confirmed,
            "primary_hypothesis": primary,
            "secondary_hypotheses": [c["id"] for c in confirmed[1:]],
            "vlm": None
            if vlm_result is None
            else {
                "model": vlm_result.get("model"),
                "notes": vlm_result.get("notes"),
                "confirmed": vlm_result.get("confirmed"),
                "rejected_proposals": vlm_result.get("rejected_proposals"),
            },
        }
        analyses.append(entry)
        buckets[primary].append(sample["id"])

        # Also index secondary for UI filtering, but primary drives bucket counts.
        print(
            "[%d/%d] %s -> %s (cc=%.3f loss=%.3f)"
            % (i + 1, len(hard), sample["id"], primary, sample["cc"], sample["loss"]),
            flush=True,
        )

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_hard": len(analyses),
        "vlm_model": None if skip_vlm else vlm_model,
        "skip_vlm": skip_vlm,
        "hypotheses": HYPOTHESES,
        "bucket_counts": {hid: len(ids) for hid, ids in buckets.items() if ids},
        "buckets": {hid: ids for hid, ids in buckets.items() if ids},
        "samples": analyses,
        "elapsed_sec": time.time() - tic,
    }
    out_path = os.path.join(out_dir, "analysis.json")
    with open(out_path, "w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    # Compact index for the review UI.
    index = {
        "hypotheses": HYPOTHESES,
        "bucket_counts": report["bucket_counts"],
        "samples": [
            {
                "id": s["id"],
                "sport": s["sport"],
                "game": s["game"],
                "window": s["window"],
                "loss": s["loss"],
                "cc": s["cc"],
                "kldiv": s["kldiv"],
                "sim": s["sim"],
                "panel": s["panel"],
                "primary_hypothesis": s["primary_hypothesis"],
                "secondary_hypotheses": s["secondary_hypotheses"],
                "confirmed": s["confirmed"],
                "proposals": s["proposals"],
                "vlm_notes": (s.get("vlm") or {}).get("notes"),
                "title": HYPOTHESIS_BY_ID.get(s["primary_hypothesis"], {}).get("title"),
            }
            for s in analyses
        ],
    }
    with open(os.path.join(out_dir, "review_index.json"), "w") as handle:
        json.dump(index, handle, indent=2)
        handle.write("\n")

    print(
        "Classified %d hard examples -> %s (buckets=%s)"
        % (len(analyses), out_path, report["bucket_counts"]),
        flush=True,
    )
    return report
