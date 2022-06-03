#!/usr/bin/env python3

#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

"""
Creates locked environment files for Conda to install geoapps within the environment.

Usage: from a the conda base environment, at the root of the project:
> python devtools/create_geoapps_env_files.py

To prepare the conda base environment, see devtools/setup-conda-base.bat
"""

import re
from pathlib import Path

from run_conda_lock import per_platform_env


def create_standalone_geoapps_lock(git_ref: str):
    create_standalone_lock(git_ref, full=True, suffix="-geoapps")


def create_standalone_simpeg_lock(git_ref: str):
    create_standalone_lock(git_ref, full=False, suffix="-geoapps-simpeg")


def create_standalone_lock(git_ref: str, full: bool, suffix=""):
    print(f"# Creating lock file for stand-alone environment (full={full})...")
    py_ver = "3.9"
    platform = "win-64"
    base_filename = f"environments/conda-py-{py_ver}-{platform}{suffix}"
    initial_lock_file = Path(f"{base_filename}-tmp.lock.yml")
    try:
        per_platform_env(py_ver, full, suffix=f"{suffix}-tmp")
        assert initial_lock_file.exists()
        final_lock_file = Path(f"{base_filename}.lock.yml")
        add_geoapps(git_ref, initial_lock_file, final_lock_file)
    finally:
        print(f"# Cleaning up intermediate files ...")
        initial_lock_file.unlink()
        for f in Path("environments").glob("conda-py-*-tmp.lock.yml"):
            f.unlink()


def add_geoapps(git_ref: str, lock_file: Path, output_file: Path):
    print(f"# Patching {lock_file} for standalone environment ...")
    exclude_re = re.compile(r"^\s*- (geoh5py|simpeg|simpeg-archive) @")
    pip_re = re.compile(r"^\s*- pip:\s*$")
    geoapps_pip = f"    - geoapps @ https://github.com/MiraGeoscience/sebhmg/archive/refs/{git_ref}.tar.gz#sha256=None\n"
    print(f"# Patched file: {output_file}")
    with open(output_file, "w") as patched:
        with open(lock_file) as input:
            for line in input:
                if not exclude_re.match(line):
                    patched.write(line)
                if pip_re.match(line):
                    patched.write(geoapps_pip)


if __name__ == "__main__":
    git_ref = "heads/GEOPY-205"  # TODO
    create_standalone_geoapps_lock(git_ref)
    create_standalone_simpeg_lock(git_ref)
