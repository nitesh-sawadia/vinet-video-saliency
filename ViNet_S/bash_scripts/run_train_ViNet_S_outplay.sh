#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES=0 /home/nitesh_sawadia/vinet_s/venv/bin/python train.py \
  --dataset outplay \
  --split_root /home/nitesh_sawadia/mnt/ml-fanpulse/sports_saliency/vinet_outplay/split \
  --checkpoint_path /home/nitesh_sawadia/vinet_s/weights/vinet_s_dhf1k.pt \
  --clip_size 32 \
  --batch_size 16 \
  --no_workers 8 \
  --pin_memory True \
  --no_epochs 40 \
  --save_every 1 \
  --lr 1e-4 \
  --num_hier 3 \
  --grouped_conv True \
  --root_grouping True
