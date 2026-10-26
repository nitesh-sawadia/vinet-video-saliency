# Error-analysis runbook

Run from `scripts/` (`cd scripts && python -m error_analysis ...`).

## DHF1K disagreement audit (`audit-dhf1k`)

Finds `split_stride8` validation windows where the DHF1K baseline prediction
strongly disagrees with the last-frame GT (`maps/0032.png`), asks
`gemini-3.6-flash` which side is plausible, and proposes Sol box corrections
only for `gt_likely_wrong`. Everything is last-frame only. The split,
`pre_split`, and the manifest are read-only; outputs go to `--out`, which must
be outside the dataset tree.

### 1. Score and select (inference VM `172.104.154.104`, GPU)

Confirm the Linode mount first (`ls /mnt/ml-fanpulse/sports_saliency/vinet_outplay/split_stride8/split_manifest.json`).
Sync the inference VM checkout with git, not by copying the tree (the `vm`
remote is the training VM `173.255.234.238`; do not run this audit there).

```bash
python -m error_analysis audit-dhf1k \
  --checkpoint /path/to/vinet_s_dhf1k.pt \
  --split-root /mnt/ml-fanpulse/sports_saliency/vinet_outplay/split_stride8 \
  --out ~/analysis/dhf1k_audit_stride8 \
  --stages score,select
```

`select` keeps every eligible window. Pass `--top-percent`, `--min-candidates`, or `--max-candidates` only to narrow that list.

- `score` writes `dhf1k_scores.jsonl` (KL, CC, SIM, peak distance, shape
  mismatch per window), `preds/<id>.png`, and `audit_manifest.json`
  (checkpoint and manifest sha256, split root, git HEAD). Re-running resumes
  at the first unscored window; a different checkpoint or manifest is refused.
- `select` writes `ranking.jsonl` (component values, rank-normalized
  components, score, rank, eligibility/exclusion reason, selection reason),
  `selection.json` (caps and the resulting `candidate_threshold`),
  `candidates.jsonl`, and last-frame `media/` for candidates.

### 2. Adjudicate and index (any machine with the audit dir)

Copy the audit dir off the VM if needed (generated artifacts, not source).
Needs `VERTEX_AI_SERVICE_ACCOUNT` in the repo `.env`; the gateway
(`GEMINI_API_KEY` + `BASE_URL`) is used only when it is absent.

```bash
python -m error_analysis audit-dhf1k --out ~/analysis/dhf1k_audit_stride8 \
  --stages adjudicate,index [--limit 20]
```

- Call 1 (verdict): last frame with Sol boxes, GT overlay, DHF1K overlay ->
  `gt_correct_model_wrong | gt_likely_wrong | both_plausible_or_ambiguous |
  insufficient_evidence`, confidence, evidence. No boxes.
- Call 2 (only for `gt_likely_wrong`): check-and-fix on the frame with labeled
  Sol boxes -> `issues_found`, `notes`, complete `detections`. Invalid output is
  retried once with the validator error; if it still fails the verdict is kept
  and the boxes stay uncorrected (`correction.status = invalid`).
- Results: `vlm_results.jsonl`, `vlm_summary.json`, `media/<id>_review.png`,
  `media/<id>_proposed.png`. Resume skips samples whose verdict succeeded.

### 3. Human review

```bash
python -m error_analysis review --out ~/analysis/dhf1k_audit_stride8
```

Serves `http://127.0.0.1:8765/`. If the audit dir is on the VM, tunnel instead:
`ssh -L 8765:127.0.0.1:8765 <vm>` and run `review --no-browser` there.

Per card: **Accept correction** (a valid audit correction, or boxes from a
directed prompt), **Reject** (keep current GT), **Skip**. The prompt box asks
Gemini to box only what you describe as salient on that frame; Accept then
stores those boxes. Each click POSTs to the server,
which writes `human_reviews.json` in the audit dir keyed by sample id
(decision, timestamp, and for accept the boxes from the review index). A new
choice overwrites that sample. Accept does not touch the dataset.

### 4. Re-export path

```bash
python -m error_analysis export-accepted --out ~/analysis/dhf1k_audit_stride8
```

Writes `accepted_corrections.jsonl` from records whose decision is `accept`
(nothing else is read). Then apply those boxes (frame 32 of each window) to the
Sol source annotations, regenerate maps and windows into a **new** split
prefix, and point training at that prefix. Never edit `split_stride8` in place.

### Tests

```bash
.venv/bin/python3 -m unittest tests.test_audit_dhf1k -v
```
