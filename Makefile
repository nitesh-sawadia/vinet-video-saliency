# Cohub model release for sports-outplay-saliency-vinet. See docs/cohub.md.
#
# `make cohub-release RUN_DIR=...` is explicit: it is never called from the
# training loop. Registration is not idempotent -- a timed-out retry can
# create an extra, undeletable version -- so this never retries on its own.

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

.PHONY: cohub-release

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
