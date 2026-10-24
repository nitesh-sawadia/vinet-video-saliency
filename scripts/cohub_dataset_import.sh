#!/bin/bash
set -euo pipefail

# Cohub dataset import for sports-outplay-saliency-vinet.
#
# NOT called by training or `make cohub-release`. Run this manually, once
# per dataset version, after:
#   1. scripts/stage_outplay_dataset.py --execute has uploaded the split
#      into the S3 prefix below (or it is already there), and
#   2. `cohub auth login` (on the VM, over SSH: `cohub auth login --paste`).
#
# `cohub datasets import` registers an existing S3 prefix in LakeFS without
# copying bytes, then commits and tags it. It does not fix a bad upload --
# fix the S3 prefix with the uploader first. Re-running this script against
# an already-imported tag is safe to inspect but will try to create the
# tag again; LakeFS/Cohub reject retagging an existing tag name, so bump
# --version for a new dataset version instead of re-running with the same
# one.
#
# Usage: scripts/cohub_dataset_import.sh [VERSION]
#   VERSION defaults to v1. See docs/cohub.md for the full workflow.

VERSION="${1:-v1}"
REPO="sports-outplay-saliency-vinet"
BUCKET="aiml-sports-data"
S3_URI="s3://${BUCKET}/${REPO}/${VERSION}/"
OWNER="video-intelligence"
LABEL_SCHEMA="outplay-saliency-32x224x384"
COHUB="${COHUB:-cohub}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RELEASE_DIR="${REPO_ROOT}/.cohub_release"
DESCRIPTION_FILE="${REPO_ROOT}/docs/outplay_dataset_description.txt"

command -v "$COHUB" >/dev/null 2>&1 || {
  echo "$COHUB not found on PATH. See docs/cohub.md for the v1.0.4 install." >&2
  exit 1
}

mkdir -p "$RELEASE_DIR"

echo "Dataset repo:    $REPO"
echo "Tag:             $VERSION"
echo "Source prefix:   $S3_URI"
echo "Owner:           $OWNER"
echo "Label schema:    $LABEL_SCHEMA"
echo

"$COHUB" datasets create "$REPO" || echo "datasets create: $REPO already exists, continuing" >&2

"$COHUB" datasets import "$REPO" \
  --s3-uri "$S3_URI" \
  --owner "$OWNER" \
  --label-schema "$LABEL_SCHEMA" \
  --split train-val \
  --tag "$VERSION"

"$COHUB" datasets card generate "$REPO" "$VERSION" --prefill \
  --set data_description="$(cat "$DESCRIPTION_FILE")" \
  --set owner="$OWNER" \
  --set label_schema="$LABEL_SCHEMA" \
  --set split=train-val \
  --set caveats="The loss only reads maps/0032.png; images/maps 0001-0031 are context, logged for visualization only." \
  --out "${RELEASE_DIR}/dataset_card_${REPO}_${VERSION}.pdf"

echo
echo "Imported. Verify with: $COHUB datasets get $REPO"
echo "Lineage URI for training: lakefs://${REPO}/${VERSION}"
