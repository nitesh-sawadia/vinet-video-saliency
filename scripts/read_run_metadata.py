#!/usr/bin/env python3
"""Print one field of a training run to stdout, for `make cohub-release`
to read into a shell variable.

  dataset_version   from <run-dir>/config/args.json (required: exits 1 if missing)
  val_loss          from <run-dir>/checkpoints/best.pt's best_loss
  val_cc            from best.pt's val_cc (empty line if the checkpoint predates it)
  val_sim           from best.pt's val_sim (empty line if the checkpoint predates it)

Used by `make cohub-release`; not called by training.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_release_weights import load_training_bundle  # noqa: E402

FIELDS = ("dataset_version", "val_loss", "val_cc", "val_sim")


def read_dataset_version(run_dir):
    config_path = os.path.join(run_dir, "config", "args.json")
    if not os.path.isfile(config_path):
        print("Run config not found: %s" % config_path, file=sys.stderr)
        sys.exit(1)
    with open(config_path) as handle:
        payload = json.load(handle)
    dataset_version = payload.get("dataset_version")
    if not dataset_version:
        print("Run has no dataset_version recorded: %s" % config_path, file=sys.stderr)
        sys.exit(1)
    return dataset_version


def read_val_metric(run_dir, field, checkpoint=None):
    checkpoint_path = checkpoint or os.path.join(run_dir, "checkpoints", "best.pt")
    bundle = load_training_bundle(checkpoint_path)
    key = "best_loss" if field == "val_loss" else field
    value = bundle.get(key)
    return "" if value is None else repr(float(value))


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--field", required=True, choices=FIELDS)
    parser.add_argument("--checkpoint", default=None, help="override checkpoint path (default: <run-dir>/checkpoints/best.pt)")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    run_dir = os.path.abspath(args.run_dir)
    if args.field == "dataset_version":
        print(read_dataset_version(run_dir))
    else:
        print(read_val_metric(run_dir, args.field, args.checkpoint))
    return 0


if __name__ == "__main__":
    sys.exit(main())
