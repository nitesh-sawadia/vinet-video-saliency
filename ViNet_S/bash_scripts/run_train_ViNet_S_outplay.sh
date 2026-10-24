#!/bin/bash
set -euo pipefail

# Resume the same run from its last checkpoint:
#   --resume <run-id | run-directory-name | checkpoints/last.pt>
# Drop logged images and videos, keeping scalar curves:
#   --clear_tensorboard_media
# --checkpoint_path is ignored when --resume is set.
#
# --dataset_version pins this run to a Cohub-versioned dataset tag. At
# startup train.py downloads that tag's split_manifest.json with `cohub`
# and checks it against --split_root; it exits if they disagree or if
# `cohub` is not on PATH (see docs/cohub.md). --resume must pass the same
# --dataset_version the run started with.

cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES=0 /home/nitesh_sawadia/vinet_s/venv/bin/python train.py \
  --dataset outplay \
  --split_root /home/nitesh_sawadia/mnt/aiml-sports-data/sports-outplay-saliency-vinet/v1 \
  --dataset_version lakefs://sports-outplay-saliency-vinet/v1 \
  --checkpoint_path /home/nitesh_sawadia/vinet_s/weights/vinet_s_dhf1k.pt \
  --clip_size 32 \
  --batch_size 32 \
  --no_workers 8 \
  --prefetch_factor 4 \
  --pin_memory True \
  --no_epochs 40 \
  --save_every 1 \
  --lr 1e-5 \
  --lr_sched True \
  --lr_factor 0.5 \
  --lr_patience 3 \
  --lr_threshold 1e-3 \
  --lr_min 1e-7 \
  --num_hier 3 \
  --grouped_conv True \
  --root_grouping True
