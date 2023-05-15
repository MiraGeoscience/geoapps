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
import json
import re
import subprocess
import urllib
import warnings
from pathlib import Path

from add_url_tag_sha256 import compute_sha256
from run_conda_lock import LockFilePatcher, per_platform_env

_FORCE_NO_PIP_HASH = False
_ARCHIVE_EXT = ".tar.gz"

APP_NAME = "geoapps"


def create_distrib_noapps_lock():
    create_distrib_lock("", [], suffix="-noapps")


def create_distrib_core_lock(git_url: str):
    create_distrib_lock(git_url, ["core"], suffix="-geoapps-core")
    create_distrib_lock(git_url, ["core"], suffix="-geoapps-core", platform="linux-64")


def create_distrib_full_lock(git_url: str):
    create_distrib_lock(git_url, ["core", "apps"], suffix="-geoapps-ui")


def create_distrib_lock(
    version_spec: str,
    extras: list[str] | None = None,
    suffix="",
    py_ver="3.10",
    platform="win-64",
):
    if extras is None:
        extras = []
    print(
        f"# Creating lock file for distributing a stand-alone environment (extras={','.join(extras)})..."
    )
    base_filename = f"conda-py-{py_ver}-{platform}{suffix}"
    initial_lock_file = Path(f"environments/{base_filename}-tmp.lock.yml")
    tmp_suffix = f"{suffix}-tmp"
    try:
        per_platform_env(py_ver, extras, suffix=tmp_suffix)
        assert initial_lock_file.exists()
        if version_spec:
            add_application(version_spec, initial_lock_file, extras)
        LockFilePatcher(initial_lock_file).patch(force_no_pip_hash=_FORCE_NO_PIP_HASH)
        final_lock_file = Path(f"{base_filename}.lock.yml")
        final_lock_file.unlink(missing_ok=True)
        initial_lock_file.rename(final_lock_file)
    finally:
        print("# Cleaning up intermediate files ...")
        initial_lock_file.unlink(missing_ok=True)
        for f in Path("environments").glob("conda-py-*-tmp.lock.yml"):
            f.unlink()


def add_application(
    version_spec: str, lock_file: Path, extras: list[str] | None = None
):
    if extras is None:
        extras = []
    print(f"# Patching {lock_file} for standalone environment ...")
    extras_string = f"[{','.join(extras)}]" if len(extras) else ""
    application_pip = f"    - {APP_NAME}{extras_string} {version_spec}\n"
    with open(lock_file, mode="a", encoding="utf-8") as file:
        file.write(application_pip)


def git_url_with_ref(args) -> tuple[str, str]:
    assert args.repo_url
    if args.ref_type == "pypi":
        ref = args.ref
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


def hash_from_pypi(package: str, version: str) -> str:
    pypi_api_url = f"https://pypi.org/pypi/{package}/{version}/json"
    # read the hash value from the pypi API
    with urllib.request.urlopen(pypi_api_url) as answer:
        data = json.loads(answer.read().decode())
        return data["urls"][0]["digests"]["sha256"]


def main():
    parser = argparse.ArgumentParser(
        description="Creates locked environment files for Conda to install application within the environment."
    )
    parser.add_argument("ref_type", choices=["sha", "tag", "branch", "pypi"])
    parser.add_argument(
        "ref", help="the git commit reference for the application pip dependency"
    )
    parser.add_argument(
        "--url",
        dest="repo_url",
        default=get_git_url(),
        help="the URL of the git repo for the application pip dependency",
    )

    args = parser.parse_args()
    if args.ref_type != "pypi":
        repo_url, ref_path = git_url_with_ref(args)
        basename_match = re.match(r".*/([^/]*)$", repo_url)
        assert basename_match
        basename = basename_match[1]
        git_download_url = build_git_url(repo_url, ref_path)
        dependency_version_spec = f"@ {git_download_url}"
        if not _FORCE_NO_PIP_HASH:
            checksum = compute_sha256(git_download_url, basename)
            dependency_version_spec += f"#sha256={checksum}"
    else:
        version = args.ref
        dependency_version_spec = f"=== {version}"
        if not _FORCE_NO_PIP_HASH:
            checksum = hash_from_pypi(APP_NAME, version)
            dependency_version_spec += f" --hash=sha256:{checksum}"

    create_distrib_noapps_lock()
    create_distrib_core_lock(dependency_version_spec)
    create_distrib_full_lock(dependency_version_spec)


if __name__ == "__main__":
    main()
