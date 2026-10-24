#!/usr/bin/env python3
"""Export the weights-only `model` state dict from a training checkpoint.

best.pt (written by ViNet_S/train.py's _save_training_state) also carries
the Adam state, the completed epoch, and the best val loss/CC/SIM, so
--resume keeps working. Cohub only wants the weights, so this writes just
the `model` entry to --out (normally the gitignored .cohub_release/ dir).

Used by `make cohub-release`; not called by training.
"""

import argparse
import os
import sys

import torch


def load_training_bundle(checkpoint_path):
    """Load a train.py training-bundle checkpoint and check it has a
    `model` state dict. Exits with a message on anything else."""
    if not os.path.isfile(checkpoint_path):
        print("Checkpoint not found:", checkpoint_path)
        sys.exit(1)
    bundle = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(bundle, dict) or "model" not in bundle:
        print("%s is not a training bundle with a 'model' key" % checkpoint_path)
        sys.exit(1)
    return bundle


def weights_only_state_dict(bundle, source):
    state_dict = bundle.get("model")
    if not (isinstance(state_dict, dict) and state_dict and all(torch.is_tensor(v) for v in state_dict.values())):
        print("%s 'model' entry is not a plain state dict" % source)
        sys.exit(1)
    return state_dict


def export_weights_only(checkpoint_path, out_path):
    """Load checkpoint_path and write only its `model` state dict to
    out_path. Returns the full bundle so the caller can also read
    epoch/best_loss/val_cc/val_sim without a second disk read."""
    bundle = load_training_bundle(checkpoint_path)
    state_dict = weights_only_state_dict(bundle, checkpoint_path)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(state_dict, out_path)
    return bundle


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="training run directory (contains checkpoints/best.pt)")
    parser.add_argument("--checkpoint", default=None, help="override checkpoint path (default: <run-dir>/checkpoints/best.pt)")
    parser.add_argument("--out", required=True, help="output path for the weights-only state dict")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    checkpoint_path = args.checkpoint or os.path.join(args.run_dir, "checkpoints", "best.pt")
    bundle = export_weights_only(checkpoint_path, args.out)
    print(
        "Exported weights-only model from %s (epoch %s, val_loss %s) to %s"
        % (checkpoint_path, bundle.get("epoch"), bundle.get("best_loss"), args.out)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
