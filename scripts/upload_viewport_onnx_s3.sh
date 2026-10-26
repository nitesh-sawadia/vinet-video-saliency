#!/usr/bin/env bash
# Upload sports-salient-viewport ONNX to S3.
#
# Usage:
#   scripts/upload_viewport_onnx_s3.sh [ONNX_FILE]
#   DRY_RUN=1 scripts/upload_viewport_onnx_s3.sh [ONNX_FILE]
#
# Objects land under:
#   s3://aiml-sports-data/onnx_repo/outplay/salient_viewport/co_<git7>_<ts>.onnx
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Load AWS creds from .env if present (KEY=val or credentials-file style).
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env" 2>/dev/null || true
  set +a
  export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-${aws_access_key_id:-}}"
  export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-${aws_secret_access_key:-}}"
  export AWS_SESSION_TOKEN="${AWS_SESSION_TOKEN:-${aws_session_token:-}}"
fi

ONNX_FILE="${1:-.cohub_release/sports-salient-viewport.onnx}"
MODULE="salient_viewport"
BUCKET="s3://aiml-sports-data/onnx_repo/outplay"
ENDPOINT_URL="https://s3.us-east-1.amazonaws.com"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"

if [[ "$ONNX_FILE" != /* ]]; then
  ONNX_FILE="$REPO_ROOT/$ONNX_FILE"
fi
[ -f "$ONNX_FILE" ] || { echo "Not found: $ONNX_FILE" >&2; exit 1; }

HASH=$(git rev-parse --short=7 HEAD)
TS=$(date '+%d%m%y_%H%M%S')
NAME="co_${HASH}_${TS}.onnx"
S3_PATH="${BUCKET}/${MODULE}/${NAME}"

if [[ "${DRY_RUN:-}" == "1" ]]; then
  echo "=== DRY RUN: upload-viewport-s3 ==="
  echo "aws s3 cp $ONNX_FILE $S3_PATH --endpoint-url $ENDPOINT_URL"
  echo "=== END DRY RUN ==="
  exit 0
fi

aws s3 cp "$ONNX_FILE" "$S3_PATH" --endpoint-url "$ENDPOINT_URL"
echo "uploaded: $S3_PATH"
