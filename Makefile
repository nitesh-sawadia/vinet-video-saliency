# Cohub model release targets. See docs/cohub.md and docs/INFERENCE_CONTRACT.md.
#
# Existing saliency release (outplay finetune):
#   make cohub-release RUN_DIR=/path/to/run
#
# Viewport ONNX + paired PT (DHF1K / pretrained weights):
#   make release-onnx CHECKPOINT=/path/to/vinet_s_dhf1k.pt
#   make release-onnx CHECKPOINT=... DRY_RUN=1   # export+regress; print S3/Cohub only
#
# Registration is not idempotent -- a timed-out retry can create an extra,
# undeletable version -- so these never retry on their own.

PYTHON ?= $(shell test -x .venv/bin/python3 && echo .venv/bin/python3 || echo python3)
COHUB ?= cohub

MODEL_NAME := sports-outplay-saliency-vinet
MODEL_OWNER := video-intelligence
MODEL_FLAVOR := pytorch
# First registration of this name freezes this description forever.
MODEL_DESCRIPTION := ViNet-S finetuned on sports outplay saliency, 32-frame 224x384 clips.
INTENDED_USE := Predict a 224x384 saliency map for a 32-frame outplay clip

RELEASE_DIR := .cohub_release
RELEASE_WEIGHTS := $(RELEASE_DIR)/$(MODEL_NAME).pt
INPUT_SCHEMA := schemas/outplay_input_schema.json
OUTPUT_SCHEMA := schemas/outplay_output_schema.json
PREPROCESSING_DOC := docs/outplay_preprocessing.txt

# ── sports-salient-viewport (ONNX + paired PT) ─────────────────────────────
MODEL_VIEWPORT := sports-salient-viewport
MODEL_VIEWPORT_PT := sports-salient-viewport-pt
CHECKPOINT ?= /home/nitesh_sawadia/vinet_s/weights/vinet_s_dhf1k.pt
DATASET_VERSION ?= NA
CONTRACT ?= docs/INFERENCE_CONTRACT.md
VIEWPORT_INPUT_SCHEMA := schemas/viewport_input_schema.json
VIEWPORT_OUTPUT_SCHEMA := schemas/viewport_output_schema.json
VIEWPORT_ONNX := $(RELEASE_DIR)/$(MODEL_VIEWPORT).onnx
VIEWPORT_DEBUG_ONNX := $(RELEASE_DIR)/$(MODEL_VIEWPORT)-debug.onnx
VIEWPORT_PT := $(RELEASE_DIR)/$(MODEL_VIEWPORT_PT).pt
INTENDED_USE_VIEWPORT := Predict a fit-inside 9:16 normalized viewport (+ confidence) for the last frame of a 32-frame sports outplay segment (highlight / key-moment clips) for vertical reframing.
MODEL_DESCRIPTION_VIEWPORT := ViNet-S (Pretrained) with in-graph ImageNet normalize and 9:16 saliency-mass viewport head; sports-agnostic outplay clips at 32x224x384.
DRY_RUN ?=

.PHONY: cohub-release \
	export-viewport regress-viewport upload-viewport-s3 \
	register-viewport-onnx register-viewport-pt register-viewport-card \
	release-onnx

cohub-release:
	@test -n "$(RUN_DIR)" || { echo "RUN_DIR is required: make cohub-release RUN_DIR=/path/to/run" >&2; exit 1; }
	@command -v $(COHUB) >/dev/null 2>&1 || { echo "$(COHUB) not found on PATH. See docs/cohub.md for the v1.0.4 install." >&2; exit 1; }
	@set -eu; \
	mkdir -p $(RELEASE_DIR); \
	$(PYTHON) scripts/export_release_weights.py --run-dir "$(RUN_DIR)" --out $(RELEASE_WEIGHTS); \
	DATASET_VERSION=$$($(PYTHON) scripts/read_run_metadata.py --run-dir "$(RUN_DIR)" --field dataset_version); \
	VAL_LOSS=$$($(PYTHON) scripts/read_run_metadata.py --run-dir "$(RUN_DIR)" --field val_loss); \
	VAL_CC=$$($(PYTHON) scripts/read_run_metadata.py --run-dir "$(RUN_DIR)" --field val_cc); \
	VAL_SIM=$$($(PYTHON) scripts/read_run_metadata.py --run-dir "$(RUN_DIR)" --field val_sim); \
	GIT_SHA=$$(git rev-parse HEAD); \
	SET_METRICS=""; \
	[ -n "$$VAL_LOSS" ] && SET_METRICS="$$SET_METRICS --set metrics.val_loss=$$VAL_LOSS"; \
	[ -n "$$VAL_CC" ] && SET_METRICS="$$SET_METRICS --set metrics.val_cc=$$VAL_CC"; \
	[ -n "$$VAL_SIM" ] && SET_METRICS="$$SET_METRICS --set metrics.val_sim=$$VAL_SIM"; \
	echo "Registering $(MODEL_NAME) from $(RUN_DIR) (dataset_version=$$DATASET_VERSION)"; \
	if ! REGISTER_JSON=$$($(COHUB) -o json models register \
		--name $(MODEL_NAME) \
		--owner $(MODEL_OWNER) \
		--dataset-version "$$DATASET_VERSION" \
		--git-sha "$$GIT_SHA" \
		--intended-use "$(INTENDED_USE)" \
		--description "$(MODEL_DESCRIPTION)" \
		--file $(RELEASE_WEIGHTS) \
		--flavor $(MODEL_FLAVOR) \
		--input-schema $(INPUT_SCHEMA) \
		--output-schema $(OUTPUT_SCHEMA)); then \
		echo "Registration failed. Checking current state before any retry (never retried automatically):" >&2; \
		$(COHUB) models get $(MODEL_NAME) >&2 || true; \
		exit 1; \
	fi; \
	echo "$$REGISTER_JSON"; \
	VERSION=$$(echo "$$REGISTER_JSON" | $(PYTHON) -c "import json,sys; print(json.load(sys.stdin)['version'])"); \
	echo "Registered $(MODEL_NAME) version $$VERSION"; \
	$(COHUB) models card generate $(MODEL_NAME) "$$VERSION" --prefill \
		--set training_dataset="$$DATASET_VERSION" \
		--set preprocessing="$$(cat $(PREPROCESSING_DOC))" \
		--set framework=pytorch \
		--set model_type=ViNet-S \
		$$SET_METRICS \
		--out $(RELEASE_DIR)/model_card_$(MODEL_NAME)_v$$VERSION.pdf; \
	echo "Publishing base variant for $(MODEL_NAME) v$$VERSION"; \
	$(COHUB) variants publish $(MODEL_NAME) "$$VERSION" base --wait; \
	echo "Released $(MODEL_NAME) v$$VERSION (dataset_version=$$DATASET_VERSION)"

# ── Viewport export / regress / S3 / Cohub ─────────────────────────────────

export-viewport:
	@test -n "$(CHECKPOINT)" || { echo "CHECKPOINT is required" >&2; exit 1; }
	@test -f "$(CHECKPOINT)" || { echo "CHECKPOINT not found: $(CHECKPOINT)" >&2; exit 1; }
	@mkdir -p $(RELEASE_DIR)
	$(PYTHON) scripts/export_viewport_onnx.py \
		--checkpoint "$(CHECKPOINT)" \
		--out-onnx $(VIEWPORT_ONNX) \
		--out-debug-onnx $(VIEWPORT_DEBUG_ONNX) \
		--out-pt $(VIEWPORT_PT)

regress-viewport:
	@test -f "$(VIEWPORT_ONNX)" || { echo "missing $(VIEWPORT_ONNX); run make export-viewport first" >&2; exit 1; }
	@test -f "$(VIEWPORT_DEBUG_ONNX)" || { echo "missing $(VIEWPORT_DEBUG_ONNX); run make export-viewport first" >&2; exit 1; }
	@test -n "$(CHECKPOINT)" || { echo "CHECKPOINT is required" >&2; exit 1; }
	$(PYTHON) scripts/regress_pt_onnx_viewport.py \
		--checkpoint "$(CHECKPOINT)" \
		--onnx $(VIEWPORT_ONNX) \
		--debug-onnx $(VIEWPORT_DEBUG_ONNX) \
		--device cuda

upload-viewport-s3:
	@test -f "$(VIEWPORT_ONNX)" || { echo "missing $(VIEWPORT_ONNX); run make export-viewport first" >&2; exit 1; }
	DRY_RUN=$(DRY_RUN) scripts/upload_viewport_onnx_s3.sh $(VIEWPORT_ONNX)

register-viewport-onnx:
	@test -n "$(DATASET_VERSION)" || { echo "DATASET_VERSION is required (use NA if none)" >&2; exit 1; }
	@test -f "$(VIEWPORT_ONNX)" || { echo "missing $(VIEWPORT_ONNX)" >&2; exit 1; }
	@test -f "$(VIEWPORT_INPUT_SCHEMA)" || { echo "missing $(VIEWPORT_INPUT_SCHEMA)" >&2; exit 1; }
	@test -f "$(VIEWPORT_OUTPUT_SCHEMA)" || { echo "missing $(VIEWPORT_OUTPUT_SCHEMA)" >&2; exit 1; }
	@if [ "$(DRY_RUN)" = "1" ]; then \
		COHUB="$(COHUB)" MODEL_VIEWPORT="$(MODEL_VIEWPORT)" MODEL_VIEWPORT_PT="$(MODEL_VIEWPORT_PT)" \
		MODEL_OWNER="$(MODEL_OWNER)" VIEWPORT_ONNX="$(VIEWPORT_ONNX)" VIEWPORT_PT="$(VIEWPORT_PT)" \
		INPUT_SCHEMA="$(VIEWPORT_INPUT_SCHEMA)" OUTPUT_SCHEMA="$(VIEWPORT_OUTPUT_SCHEMA)" \
		DATASET_VERSION="$(DATASET_VERSION)" CONTRACT="$(CONTRACT)" RELEASE_DIR="$(RELEASE_DIR)" \
		INTENDED_USE_VIEWPORT="$(INTENDED_USE_VIEWPORT)" MODEL_DESCRIPTION_VIEWPORT="$(MODEL_DESCRIPTION_VIEWPORT)" \
		scripts/print_viewport_cohub_commands.sh; \
	else \
		command -v $(COHUB) >/dev/null 2>&1 || { echo "$(COHUB) not found on PATH. See docs/cohub.md." >&2; exit 1; }; \
		set -eu; \
		GIT_SHA=$$(git rev-parse HEAD); \
		echo "Registering $(MODEL_VIEWPORT) (onnx, dataset_version=$(DATASET_VERSION))"; \
		if ! REGISTER_JSON=$$($(COHUB) -o json models register \
			--name $(MODEL_VIEWPORT) \
			--owner $(MODEL_OWNER) \
			--dataset-version "$(DATASET_VERSION)" \
			--git-sha "$$GIT_SHA" \
			--intended-use "$(INTENDED_USE_VIEWPORT)" \
			--description "$(MODEL_DESCRIPTION_VIEWPORT)" \
			--file $(VIEWPORT_ONNX) \
			--flavor onnx); then \
			echo "Registration failed. Checking current state (never retried automatically):" >&2; \
			$(COHUB) models get $(MODEL_VIEWPORT) >&2 || true; \
			exit 1; \
		fi; \
		echo "$$REGISTER_JSON"; \
		VERSION=$$(echo "$$REGISTER_JSON" | $(PYTHON) -c "import json,sys; print(json.load(sys.stdin)['version'])"); \
		echo "$$VERSION" > $(RELEASE_DIR)/$(MODEL_VIEWPORT).version; \
		echo "Registered $(MODEL_VIEWPORT) version $$VERSION"; \
		$(COHUB) models card generate $(MODEL_VIEWPORT) "$$VERSION" --prefill \
			--set training_dataset="$(DATASET_VERSION)" \
			--set preprocessing="$$(cat $(CONTRACT))" \
			--set framework=onnx \
			--set model_type=ViNet-S-viewport \
			--out $(RELEASE_DIR)/model_card_$(MODEL_VIEWPORT)_v$$VERSION.pdf; \
		echo "Publishing base variant for $(MODEL_VIEWPORT) v$$VERSION"; \
		$(COHUB) variants publish $(MODEL_VIEWPORT) "$$VERSION" base --wait; \
	fi

register-viewport-pt:
	@test -n "$(DATASET_VERSION)" || { echo "DATASET_VERSION is required (use NA if none)" >&2; exit 1; }
	@test -f "$(VIEWPORT_PT)" || { echo "missing $(VIEWPORT_PT)" >&2; exit 1; }
	@if [ "$(DRY_RUN)" = "1" ]; then \
		echo "(pytorch register commands included in register-viewport-onnx DRY_RUN print)"; \
	else \
		command -v $(COHUB) >/dev/null 2>&1 || { echo "$(COHUB) not found on PATH. See docs/cohub.md." >&2; exit 1; }; \
		set -eu; \
		GIT_SHA=$$(git rev-parse HEAD); \
		echo "Registering $(MODEL_VIEWPORT_PT) (pytorch, dataset_version=$(DATASET_VERSION))"; \
		if ! REGISTER_JSON=$$($(COHUB) -o json models register \
			--name $(MODEL_VIEWPORT_PT) \
			--owner $(MODEL_OWNER) \
			--dataset-version "$(DATASET_VERSION)" \
			--git-sha "$$GIT_SHA" \
			--intended-use "$(INTENDED_USE_VIEWPORT)" \
			--description "$(MODEL_DESCRIPTION_VIEWPORT)" \
			--file $(VIEWPORT_PT) \
			--flavor pytorch); then \
			echo "Registration failed. Checking current state (never retried automatically):" >&2; \
			$(COHUB) models get $(MODEL_VIEWPORT_PT) >&2 || true; \
			exit 1; \
		fi; \
		echo "$$REGISTER_JSON"; \
		VERSION=$$(echo "$$REGISTER_JSON" | $(PYTHON) -c "import json,sys; print(json.load(sys.stdin)['version'])"); \
		echo "$$VERSION" > $(RELEASE_DIR)/$(MODEL_VIEWPORT_PT).version; \
		echo "Registered $(MODEL_VIEWPORT_PT) version $$VERSION"; \
		$(COHUB) models card generate $(MODEL_VIEWPORT_PT) "$$VERSION" --prefill \
			--set training_dataset="$(DATASET_VERSION)" \
			--set preprocessing="$$(cat $(CONTRACT))" \
			--set framework=pytorch \
			--set model_type=ViNet-S-viewport \
			--out $(RELEASE_DIR)/model_card_$(MODEL_VIEWPORT_PT)_v$$VERSION.pdf; \
		echo "Publishing base variant for $(MODEL_VIEWPORT_PT) v$$VERSION"; \
		$(COHUB) variants publish $(MODEL_VIEWPORT_PT) "$$VERSION" base --wait; \
	fi

register-viewport-card:
	@if [ "$(DRY_RUN)" = "1" ]; then \
		echo "(card/publish commands included in register-viewport-onnx DRY_RUN print)"; \
	else \
		echo "Cards and variants publish run inside register-viewport-onnx / register-viewport-pt"; \
	fi

# export → regress → S3 → Cohub (fail closed: regress before any remote write)
release-onnx: export-viewport regress-viewport
	$(MAKE) upload-viewport-s3 CHECKPOINT="$(CHECKPOINT)" DRY_RUN="$(DRY_RUN)"
	$(MAKE) register-viewport-onnx CHECKPOINT="$(CHECKPOINT)" DRY_RUN="$(DRY_RUN)" \
		DATASET_VERSION="$(DATASET_VERSION)"
	$(MAKE) register-viewport-pt CHECKPOINT="$(CHECKPOINT)" DRY_RUN="$(DRY_RUN)" \
		DATASET_VERSION="$(DATASET_VERSION)"
	@if [ "$(DRY_RUN)" = "1" ]; then \
		echo "DRY_RUN complete: reviewed S3/Cohub commands; no remote writes."; \
	else \
		echo "Released $(MODEL_VIEWPORT) + $(MODEL_VIEWPORT_PT) (dataset_version=$(DATASET_VERSION))"; \
	fi
