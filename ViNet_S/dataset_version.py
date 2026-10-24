"""Optional Cohub dataset-version pin for outplay training.

--dataset_version is off unless set. With it omitted, training reads
whatever --split_root points at (the Linode goofys mount, by default) and
never calls the Cohub CLI. Pass lakefs://<repo>/<tag> (repo, then tag --
not lakefs://datasets/...) to download that tag's split_manifest.json and
check it against the mounted split, so a pinned run cannot silently train
on the wrong data version. Frame reads stay on the mounted split; Cohub is
only the version check, not the file server. --resume of a run that stored
a URI must pass the same flag and exits if it is missing or disagrees.
Runs that never recorded one resume without it.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile

_LAKEFS_URI_RE = re.compile(r"lakefs://([^/]+)/([^/]+)")


def parse_lakefs_uri(uri):
    """Split lakefs://<repo>/<tag> into (repo, tag). Exits on a bad URI."""
    match = _LAKEFS_URI_RE.fullmatch(uri)
    if not match:
        print("--dataset_version must look like lakefs://<repo>/<tag>, got %r" % uri)
        sys.exit(1)
    return match.group(1), match.group(2)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_manifest_sha256(repo, tag, dest_dir, cohub_bin="cohub"):
    """Download <repo> <tag> split_manifest.json with the Cohub CLI into
    dest_dir and return its sha256. Exits with a clear message if `cohub`
    is not on PATH or the download fails."""
    remote_manifest = os.path.join(dest_dir, "split_manifest.json")
    cmd = [cohub_bin, "datasets", "download", repo, tag, "split_manifest.json", "--out", remote_manifest]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        print(
            "%r is not on PATH. Install cohub v1.0.4 (see docs/cohub.md) before "
            "training with --dataset_version." % cohub_bin
        )
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print("cohub datasets download failed for %s %s:" % (repo, tag))
        print((exc.stderr or exc.stdout or str(exc)).strip())
        sys.exit(1)
    return sha256_file(remote_manifest)


def verify_dataset_version(dataset_version, split_root, cohub_bin="cohub"):
    """Check --dataset_version against the mounted split when it is set.

    An empty value skips the check and returns None, so training can keep
    reading a goofys mount with no Cohub CLI. Returns (repo, tag) when the
    mounted split_manifest.json matches the tag; exits otherwise."""
    if not dataset_version:
        print(
            "No --dataset_version; skipping the Cohub manifest check and "
            "training against the mounted split."
        )
        return None
    repo, tag = parse_lakefs_uri(dataset_version)
    local_manifest = os.path.join(split_root, "split_manifest.json")
    if not os.path.isfile(local_manifest):
        print("Mounted split manifest not found:", local_manifest)
        sys.exit(1)
    local_sha = sha256_file(local_manifest)
    with tempfile.TemporaryDirectory() as tmp_dir:
        remote_sha = download_manifest_sha256(repo, tag, tmp_dir, cohub_bin=cohub_bin)
    if remote_sha != local_sha:
        print(
            "Mounted split_manifest.json (sha256 %s) does not match %s tag %s "
            "(sha256 %s)." % (local_sha, repo, tag, remote_sha)
        )
        sys.exit(1)
    print("Dataset version verified:", dataset_version, "sha256", local_sha)
    return repo, tag


def check_resume_dataset_version(dataset_version, run_dir):
    """On --resume, the run's args.json pins the dataset_version that run
    started with. Exit if the new flag disagrees; do nothing for older runs
    that never recorded one."""
    config_path = os.path.join(run_dir, "config", "args.json")
    if not os.path.isfile(config_path):
        return
    with open(config_path) as handle:
        payload = json.load(handle)
    stored = payload.get("dataset_version")
    if stored and stored != dataset_version:
        if not dataset_version:
            print(
                "This run is pinned to %s (%s). Pass the same --dataset_version to resume."
                % (stored, config_path)
            )
        else:
            print(
                "--dataset_version %s disagrees with this run's stored %s (%s)"
                % (dataset_version, stored, config_path)
            )
        sys.exit(1)
