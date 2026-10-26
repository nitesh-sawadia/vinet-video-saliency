#!/usr/bin/env bash
# Print the exact Cohub register / card / variants-publish argv for
# sports-salient-viewport (+ -pt) without executing anything.
#
# Usage (normally via Makefile DRY_RUN=1):
#   scripts/print_viewport_cohub_commands.sh
#
# Environment (set by Makefile):
#   COHUB, MODEL_VIEWPORT, MODEL_VIEWPORT_PT, MODEL_OWNER,
#   VIEWPORT_ONNX, VIEWPORT_PT, INPUT_SCHEMA, OUTPUT_SCHEMA,
#   DATASET_VERSION, INTENDED_USE_VIEWPORT, MODEL_DESCRIPTION_VIEWPORT,
#   CONTRACT, RELEASE_DIR, PYTHON
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

COHUB="${COHUB:-cohub}"
MODEL_VIEWPORT="${MODEL_VIEWPORT:-sports-salient-viewport}"
MODEL_VIEWPORT_PT="${MODEL_VIEWPORT_PT:-sports-salient-viewport-pt}"
MODEL_OWNER="${MODEL_OWNER:-video-intelligence}"
RELEASE_DIR="${RELEASE_DIR:-.cohub_release}"
VIEWPORT_ONNX="${VIEWPORT_ONNX:-$RELEASE_DIR/$MODEL_VIEWPORT.onnx}"
VIEWPORT_PT="${VIEWPORT_PT:-$RELEASE_DIR/$MODEL_VIEWPORT_PT.pt}"
INPUT_SCHEMA="${INPUT_SCHEMA:-schemas/viewport_input_schema.json}"
OUTPUT_SCHEMA="${OUTPUT_SCHEMA:-schemas/viewport_output_schema.json}"
DATASET_VERSION="${DATASET_VERSION:-NA}"
CONTRACT="${CONTRACT:-docs/INFERENCE_CONTRACT.md}"
INTENDED_USE_VIEWPORT="${INTENDED_USE_VIEWPORT:-Predict a fit-inside 9:16 normalized viewport (+ confidence) for the last frame of a 32-frame sports outplay segment (highlight / key-moment clips) for vertical reframing.}"
MODEL_DESCRIPTION_VIEWPORT="${MODEL_DESCRIPTION_VIEWPORT:-ViNet-S (Pretrained) with in-graph ImageNet normalize and 9:16 saliency-mass viewport head; sports-agnostic outplay clips at 32x224x384.}"

GIT_SHA="$(git rev-parse HEAD)"
PREPROCESSING="$(cat "$CONTRACT")"

print_block() {
  local title="$1"
  shift
  echo "=== DRY RUN: $title ==="
  printf '%q ' "$@"
  echo
  echo "=== END DRY RUN ==="
  echo
}

# cohub CLI v1.0.4 has no --input-schema/--output-schema/--param with --file;
# I/O contract is documented in INFERENCE_CONTRACT.md and the model card.
print_block "cohub models register (onnx)" \
  "$COHUB" -o json models register \
  --name "$MODEL_VIEWPORT" \
  --owner "$MODEL_OWNER" \
  --dataset-version "$DATASET_VERSION" \
  --git-sha "$GIT_SHA" \
  --intended-use "$INTENDED_USE_VIEWPORT" \
  --description "$MODEL_DESCRIPTION_VIEWPORT" \
  --file "$VIEWPORT_ONNX" \
  --flavor onnx

print_block "cohub models register (pytorch)" \
  "$COHUB" -o json models register \
  --name "$MODEL_VIEWPORT_PT" \
  --owner "$MODEL_OWNER" \
  --dataset-version "$DATASET_VERSION" \
  --git-sha "$GIT_SHA" \
  --intended-use "$INTENDED_USE_VIEWPORT" \
  --description "$MODEL_DESCRIPTION_VIEWPORT" \
  --file "$VIEWPORT_PT" \
  --flavor pytorch

echo "=== DRY RUN: cohub models card generate (onnx) ==="
echo "$COHUB models card generate $MODEL_VIEWPORT <VERSION_FROM_REGISTER> --prefill \\"
echo "  --set training_dataset=$DATASET_VERSION \\"
echo "  --set preprocessing=\"\$(cat $CONTRACT)\" \\"
echo "  --set framework=onnx \\"
echo "  --set model_type=ViNet-S-viewport \\"
echo "  --out $RELEASE_DIR/model_card_${MODEL_VIEWPORT}_v<VERSION>.pdf"
echo "=== END DRY RUN ==="
echo

echo "=== DRY RUN: cohub models card generate (pytorch) ==="
echo "$COHUB models card generate $MODEL_VIEWPORT_PT <VERSION_FROM_REGISTER> --prefill \\"
echo "  --set training_dataset=$DATASET_VERSION \\"
echo "  --set preprocessing=\"\$(cat $CONTRACT)\" \\"
echo "  --set framework=pytorch \\"
echo "  --set model_type=ViNet-S-viewport \\"
echo "  --out $RELEASE_DIR/model_card_${MODEL_VIEWPORT_PT}_v<VERSION>.pdf"
echo "=== END DRY RUN ==="
echo

echo "=== DRY RUN: cohub variants publish (onnx) ==="
echo "$COHUB variants publish $MODEL_VIEWPORT <VERSION> base --wait"
echo "=== END DRY RUN ==="
echo

echo "=== DRY RUN: cohub variants publish (pytorch) ==="
echo "$COHUB variants publish $MODEL_VIEWPORT_PT <VERSION> base --wait"
echo "=== END DRY RUN ==="
echo

# Silence unused-var lint for preprocessing when only printing.
: "${PREPROCESSING}"
