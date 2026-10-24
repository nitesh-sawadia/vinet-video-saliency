#!/bin/bash
set -euo pipefail

# Score the outplay val set with a run's best.pt, VLM-confirm hard-example
# hypotheses (OPENAI_API_KEY + BASE_URL from repo .env), and optionally serve
# the review UI.
#
# Usage on the training VM:
#   ./bash_scripts/run_error_analysis_outplay.sh
#   ./bash_scripts/run_error_analysis_outplay.sh --review
#   OUT=/tmp/ea_run REVIEW=1 ./bash_scripts/run_error_analysis_outplay.sh

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

RUN_DIR="${RUN_DIR:-/home/nitesh_sawadia/vinet_s/runs/20261023T103625Z_bd366b8b14164cd0b2d6e0173ba4eea9}"
CHECKPOINT="${CHECKPOINT:-$RUN_DIR/checkpoints/best.pt}"
SPLIT_ROOT="${SPLIT_ROOT:-/home/nitesh_sawadia/mnt/ml-fanpulse/sports_saliency/vinet_outplay/split}"
OUT="${OUT:-/home/nitesh_sawadia/vinet_s/error_analysis/$(basename "$RUN_DIR")}"
PYTHON="${PYTHON:-/home/nitesh_sawadia/vinet_s/venv/bin/python}"
VLM_MODEL="${VLM_MODEL:-vertex_ai/gemini-2.5-flash}"

EXTRA=()
if [[ "${SKIP_VLM:-0}" == "1" ]]; then
  EXTRA+=(--skip-vlm)
fi
if [[ "${REVIEW:-0}" == "1" ]] || [[ "${1:-}" == "--review" ]]; then
  EXTRA+=(--review --host 0.0.0.0 --port "${PORT:-8765}" --no-browser)
fi
if [[ -n "${LIMIT:-}" ]]; then
  EXTRA+=(--limit "$LIMIT")
fi

export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" -m error_analysis run \
  --checkpoint "$CHECKPOINT" \
  --split-root "$SPLIT_ROOT" \
  --out "$OUT" \
  --batch-size "${BATCH_SIZE:-16}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --vlm-model "$VLM_MODEL" \
  "${EXTRA[@]}"
