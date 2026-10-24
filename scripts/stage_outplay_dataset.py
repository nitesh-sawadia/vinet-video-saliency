#!/usr/bin/env python3
"""One-time upload of the outplay split into
s3://aiml-sports-data/sports-outplay-saliency-vinet/<version>/.

NOT called by training or `make cohub-release`. Run this manually, once per
dataset version, before `scripts/cohub_dataset_import.sh`. See docs/cohub.md.

The split lives on Linode and is normally read through the goofys mount at
--split-root (default: the VM path from ViNet_S/README.md). This script
copies it into aiml-sports-data so `cohub datasets import` can register that
prefix with LakeFS -- import is a metadata pointer, not a copy, so the bytes
have to land in the bucket first.

Dry run by default: prints the object count and byte total and does
nothing else. Pass --execute to actually upload. Either way, this refuses
to write into a version prefix that already has objects, because
overwriting keys under an imported tag breaks that tag. A later dataset
version uploads to a sibling `vN/` prefix instead (pass --version v2).

Credentials come from the profile-style block in .env (NOT shell exports):

    [<profile>]
    aws_access_key_id=...
    aws_secret_access_key=...
    aws_session_token=...

This script only reads the bracketed profile name out of that file; it
never reads the credential values into Python, prints them, or logs them.
The AWS CLI is pointed at the file itself via AWS_SHARED_CREDENTIALS_FILE.
"""

import argparse
import json
import os
import re
import subprocess
import sys

DEFAULT_SPLIT_ROOT = "/home/nitesh_sawadia/mnt/ml-fanpulse/sports_saliency/vinet_outplay/split"
DEFAULT_BUCKET = "aiml-sports-data"
DEFAULT_DATASET = "sports-outplay-saliency-vinet"
DEFAULT_REGION = "us-east-1"
FRAME_COUNT = 32

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROFILE_HEADER_RE = re.compile(r"^\[(?P<name>[^\]]+)\]\s*$")


def _profile_name_from_env_file(env_path):
    """Read only the bracketed profile name out of an AWS-credentials-style
    file; the key/secret/token values on the following lines are never read
    into this process."""
    if not os.path.isfile(env_path):
        print("Credentials file not found:", env_path)
        sys.exit(1)
    with open(env_path) as handle:
        for line in handle:
            match = _PROFILE_HEADER_RE.match(line.strip())
            if match:
                return match.group("name")
    print("No [profile] section header found in", env_path)
    sys.exit(1)


def _frame_names():
    return ["%04d.png" % index for index in range(1, FRAME_COUNT + 1)]


def _iter_windows(manifest):
    windows = manifest.get("windows") if isinstance(manifest, dict) else None
    if not isinstance(windows, list):
        raise ValueError("Manifest has no windows list")
    for window in windows:
        missing = [key for key in ("split", "sport", "game", "dir") if key not in window]
        if missing:
            raise KeyError("Window is missing %s: %s" % (missing, window))
        yield window


def _window_dir_name(window_dir):
    text = str(window_dir).replace("\\", "/").rstrip("/")
    name = text.split("/")[-1]
    if name in ("", ".", ".."):
        raise ValueError("Window dir has no basename: %s" % window_dir)
    return name


def validate_split(split_root):
    """Check split_manifest.json and every window's 32 images + 32 maps are
    present under split_root. Returns (file_count, total_bytes); exits with
    a message on the first problem found."""
    manifest_path = os.path.join(split_root, "split_manifest.json")
    if not os.path.isfile(manifest_path):
        print("Split manifest not found:", manifest_path)
        sys.exit(1)
    with open(manifest_path) as handle:
        manifest = json.load(handle)

    file_count = 1
    total_bytes = os.path.getsize(manifest_path)
    frame_names = _frame_names()
    try:
        windows = list(_iter_windows(manifest))
    except (ValueError, KeyError) as exc:
        print("Bad manifest %s: %s" % (manifest_path, exc))
        sys.exit(1)
    if not windows:
        print("Manifest has no windows:", manifest_path)
        sys.exit(1)

    for window in windows:
        window_path = os.path.join(
            split_root, window["split"], window["sport"], window["game"], _window_dir_name(window["dir"])
        )
        for kind in ("images", "maps"):
            folder = os.path.join(window_path, kind)
            if not os.path.isdir(folder):
                print("Missing %s folder: %s" % (kind, folder))
                sys.exit(1)
            for name in frame_names:
                path = os.path.join(folder, name)
                if not os.path.isfile(path):
                    print("Missing frame:", path)
                    sys.exit(1)
                total_bytes += os.path.getsize(path)
                file_count += 1
    return file_count, total_bytes


def _destination_has_objects(bucket, prefix, env, aws_bin="aws"):
    cmd = [
        aws_bin, "s3api", "list-objects-v2",
        "--bucket", bucket, "--prefix", prefix, "--max-items", "1",
    ]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print("Could not list s3://%s/%s:" % (bucket, prefix))
        print(result.stderr.strip())
        sys.exit(1)
    payload = json.loads(result.stdout) if result.stdout.strip() else {}
    return bool(payload.get("Contents"))


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split-root", default=DEFAULT_SPLIT_ROOT, help="local goofys mount of the Linode split")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--version", default="v1", help="destination prefix segment, e.g. v1 or v2")
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--env-file", default=os.path.join(_REPO_ROOT, ".env"), help="AWS-credentials-style file (default: repo .env)")
    parser.add_argument("--profile", default=None, help="override the AWS profile name (default: read from --env-file)")
    parser.add_argument("--aws-bin", default="aws")
    parser.add_argument("--execute", action="store_true", help="actually upload; default is a dry run")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    prefix = "%s/%s/" % (args.dataset, args.version)
    destination = "s3://%s/%s" % (args.bucket, prefix)

    file_count, total_bytes = validate_split(args.split_root)
    print("Split root:       %s" % args.split_root)
    print("Destination:      %s" % destination)
    print("Files to upload:  %d" % file_count)
    print("Bytes to upload:  %d (%.2f GiB)" % (total_bytes, total_bytes / (1024.0 ** 3)))

    if not args.execute:
        print()
        print("Dry run only (default). Re-run with --execute to upload.")
        print("cohub datasets import is a metadata pointer, not a copy; this script")
        print("refuses to overwrite an existing version prefix.")
        return 0

    profile = args.profile or _profile_name_from_env_file(args.env_file)
    env = dict(os.environ)
    env["AWS_SHARED_CREDENTIALS_FILE"] = os.path.abspath(args.env_file)
    env["AWS_PROFILE"] = profile
    env["AWS_DEFAULT_REGION"] = args.region
    # Belt-and-suspenders: never let ambient shell exports override the
    # credentials file above.
    for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        env.pop(key, None)

    if _destination_has_objects(args.bucket, prefix, env, aws_bin=args.aws_bin):
        print("Refusing to upload: s3://%s/%s already has objects. Use a new --version." % (args.bucket, prefix))
        return 1

    cmd = [args.aws_bin, "s3", "sync", args.split_root, destination, "--no-progress"]
    print("+ %s" % " ".join(cmd))
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        return result.returncode
    print("Uploaded %d files (%d bytes) to %s" % (file_count, total_bytes, destination))
    return 0


if __name__ == "__main__":
    sys.exit(main())
