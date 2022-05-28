#!/usr/bin/env python3

#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import os
import subprocess


def create_multi_platform_lock(py_ver: str, platform: str = None):
    print(f"# Create multi-platform lock file for Python {py_ver}")
    platform_option = f"-p {platform}" if platform else ""
    subprocess.run(
        f"conda-lock lock -f pyproject.toml -f env-python-{py_ver}.yml {platform_option} --lockfile conda-py-{py_ver}-lock.yml",
        env=dict(os.environ, PYTHONUTF8="1"),
        shell=True,
        check=True,
        stderr=subprocess.STDOUT,
    )


def per_platform_env(py_ver: str, full=True, dev=False, suffix=""):
    print(
        f"# Create per platform Conda env files for Python {py_ver} ({'WITH' if dev else 'NO'} dev dependencies) "
    )
    dev_dep_option = "--dev-dependencies" if dev else "--no-dev-dependencies"
    dev_suffix = "-dev" if dev else ""
    extras_option = "--extras full" if full else ""
    subprocess.run(
        (
            f"conda-lock render {dev_dep_option} {extras_option} -k env"
            f" --filename-template conda-py-{py_ver}-{{platform}}{dev_suffix}{suffix}.lock conda-py-{py_ver}-lock.yml"
        ),
        env=dict(os.environ, PYTHONUTF8="1"),
        shell=True,
        check=True,
        stderr=subprocess.STDOUT,
    )


if __name__ == "__main__":
    for py_ver in ["3.9", "3.8", "3.7"]:
        create_multi_platform_lock(py_ver)
        per_platform_env(py_ver, dev=False)
        per_platform_env(py_ver, dev=True)

    # for simpeg with Python 3.9
    per_platform_env("3.9", full=False, dev=False, suffix="-simpeg")
