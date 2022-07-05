#!/usr/bin/env python3

#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

"""
Creates locked environment files for Conda to install geoapps within the environment.

Usage: from the conda base environment, at the root of the project:
> python devtools/create_geoapps_env_files.py

To prepare the conda base environment, see devtools/setup-conda-base.bat
"""

import argparse
import re
from pathlib import Path

from add_url_tag_sha256 import computeSha256
from run_conda_lock import per_platform_env

_archive_ext = ".tar.gz"


def create_standalone_geoapps_lock(git_url: str):
    create_standalone_lock(git_url, full=True, suffix="-geoapps")


def create_standalone_simpeg_lock(git_url: str):
    create_standalone_lock(git_url, full=False, suffix="-geoapps-simpeg")


def create_standalone_lock(git_url: str, full: bool, suffix=""):
    print(f"# Creating lock file for stand-alone environment (full={full})...")
    py_ver = "3.9"
    platform = "win-64"
    base_filename = f"conda-py-{py_ver}-{platform}{suffix}"
    initial_lock_file = Path(f"environments/{base_filename}-tmp.lock.yml")
    try:
        per_platform_env(py_ver, full, suffix=f"{suffix}-tmp")
        assert initial_lock_file.exists()
        final_lock_file = Path(f"{base_filename}.lock.yml")
        add_geoapps(git_url, initial_lock_file, final_lock_file)
    finally:
        print(f"# Cleaning up intermediate files ...")
        initial_lock_file.unlink()
        for f in Path("environments").glob("conda-py-*-tmp.lock.yml"):
            f.unlink()


def add_geoapps(git_url: str, lock_file: Path, output_file: Path):
    print(f"# Patching {lock_file} for standalone environment ...")
    pip_dependency_re = re.compile(r"^\s*- (geoh5py|simpeg|simpeg-archive) @")
    pip_dependency_lines = []
    with open(lock_file) as input:
        for line in input:
            if pip_dependency_re.match(line):
                pip_dependency_lines.append(line)

    pip_section_re = re.compile(r"^\s*- pip:\s*$")
    geoapps_pip = f"    - geoapps @ {git_url}\n"
    print(f"# Patched file: {output_file}")
    with open(output_file, "w") as patched:
        with open(lock_file) as input:
            for line in input:
                if not pip_dependency_re.match(line):
                    patched.write(line)
                if pip_section_re.match(line):
                    for pip_line in pip_dependency_lines:
                        patched.write(pip_line)
                    patched.write(geoapps_pip)


def build_git_url(args) -> str:
    if args.ref_type == "sha":
        ref = args.ref
    elif args.ref_type == "tag":
        ref = f"refs/tags/{args.ref}"
    elif args.ref_type == "branch":
        ref = f"refs/heads/{args.ref}"
    else:
        raise RuntimeError(f"Unhandled reference type ${args.ref_type}")
    return f"{args.repo_url}/archive/{ref}{_archive_ext}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates locked environment files for Conda to install geoapps within the environment."
    )
    parser.add_argument("ref_type", choices=["sha", "tag", "branch"])
    parser.add_argument(
        "ref", help="the git commit reference for the geoapps pip dependency"
    )
    parser.add_argument(
        "--url",
        dest="repo_url",
        default="https://github.com/MiraGeoscience/geoapps",
        help="the URL of the git repo for the geoapps pip dependency",
    )
    git_url = build_git_url(parser.parse_args())
    checksum = computeSha256(git_url, "geoapps")
    checked_git_url = f"{git_url}#sha256={checksum}"

    create_standalone_geoapps_lock(checked_git_url)
    create_standalone_simpeg_lock(checked_git_url)
