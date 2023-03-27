#!/usr/bin/env python3

#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

"""
Creates locked environment files for Conda to install the application within the environment.

Usage: from the conda base environment, at the root of the project:
> python devtools/create_application_env_files.py

To prepare the conda base environment, see devtools/setup-conda-base.bat
"""

from __future__ import annotations

import argparse
import re
import subprocess
import warnings
from pathlib import Path

from add_url_tag_sha256 import compute_sha256
from run_conda_lock import LockFilePatcher, per_platform_env

_FORCE_NO_PIP_HASH = True
_ARCHIVE_EXT = ".tar.gz"

APP_NAME = "geoapps"


def create_distrib_noapps_lock():
    create_distrib_lock("", [], suffix="-noapps")


def create_distrib_core_lock(git_url: str):
    create_distrib_lock(git_url, ["core"], suffix="-geoapps-core")


def create_distrib_full_lock(git_url: str):
    create_distrib_lock(git_url, ["apps"], suffix="-geoapps-ui")


def create_distrib_lock(git_url: str, extras=[], suffix=""):
    print(
        f"# Creating lock file for distributing a stand-alone environment (extras={','.join(extras)})..."
    )
    py_ver = "3.10"
    platform = "win-64"
    base_filename = f"conda-py-{py_ver}-{platform}{suffix}"
    initial_lock_file = Path(f"environments/{base_filename}-tmp.lock.yml")
    tmp_suffix = f"{suffix}-tmp"
    try:
        per_platform_env(py_ver, extras, suffix=tmp_suffix)
        assert initial_lock_file.exists()
        if git_url:
            add_application(git_url, initial_lock_file)
        LockFilePatcher(initial_lock_file).patch(force_no_pip_hash=_FORCE_NO_PIP_HASH)
        final_lock_file = Path(f"{base_filename}.lock.yml")
        final_lock_file.unlink(missing_ok=True)
        initial_lock_file.rename(final_lock_file)
    finally:
        print("# Cleaning up intermediate files ...")
        initial_lock_file.unlink(missing_ok=True)
        for f in Path("environments").glob("conda-py-*-tmp.lock.yml"):
            f.unlink()


def add_application(git_url: str, lock_file: Path):
    print(f"# Patching {lock_file} for standalone environment ...")
    application_pip = f"    - {APP_NAME} @ {git_url}\n"
    with open(lock_file, "a") as file:
        file.write(application_pip)


def git_url_with_ref(args) -> tuple[str, str]:
    assert args.repo_url
    if args.ref_type == "sha":
        ref = args.ref
    elif args.ref_type == "tag":
        ref = f"refs/tags/{args.ref}"
    elif args.ref_type == "branch":
        ref = f"refs/heads/{args.ref}"
    else:
        raise RuntimeError(f"Unhandled reference type ${args.ref_type}")
    return args.repo_url, ref


def build_git_url(repo_url: str, ref: str) -> str:
    return f"{repo_url}/archive/{ref}{_ARCHIVE_EXT}"


def get_git_url():
    process = subprocess.run(
        ["git", "config", "--get-regexp", "remote.*.url"],
        check=True,
        capture_output=True,
        text=True,
    )

    mira_remote_re = re.compile(r".*\bgithub.com(?::|/)(MiraGeoscience/\S+)\s*$")
    for line in process.stdout.splitlines():
        match = mira_remote_re.match(line)
        if match:
            segment = match[1][:-4] if match[1].endswith(".git") else match[1]
            return f"https://github.com/{segment}"
    warnings.warn(
        "Could not detect the remote MiraGeoscience github repository for this application."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Creates locked environment files for Conda to install application within the environment."
    )
    parser.add_argument("ref_type", choices=["sha", "tag", "branch"])
    parser.add_argument(
        "ref", help="the git commit reference for the application pip dependency"
    )
    parser.add_argument(
        "--url",
        dest="repo_url",
        default=get_git_url(),
        help="the URL of the git repo for the application pip dependency",
    )

    repo_url, ref_path = git_url_with_ref(parser.parse_args())
    basename_match = re.match(r".*/([^/]*)$", repo_url)
    assert basename_match
    basename = basename_match[1]
    git_download_url = build_git_url(repo_url, ref_path)
    if _FORCE_NO_PIP_HASH:
        dependency_url = git_download_url
    else:
        checksum = compute_sha256(git_download_url, basename)
        dependency_url = f"{git_download_url}#sha256={checksum}"

    create_distrib_noapps_lock()
    create_distrib_core_lock(dependency_url)
    create_distrib_full_lock(dependency_url)


if __name__ == "__main__":
    main()
