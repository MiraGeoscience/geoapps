#!/usr/bin/env python3

#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import subprocess


def create_multi_platform_lock(py_ver: str):
    print(f"# Create multi-platform lock file for Python {py_ver}")
    subprocess.run(
        f"conda-lock lock -f pyproject.toml -f env-python-{py_ver}.yml --lockfile conda-py-{py_ver}-lock.yml",
        shell=True,
        check=True,
        stderr=subprocess.STDOUT,
    )


def per_platform_env(py_ver: str, dev=False):
    print(
        f"# Create per platform Conda env files for Python {py_ver} ({'WITH' if dev else 'NO'} dev dependencies) "
    )
    dev_dep_option = "--dev-dependencies" if dev else "--no-dev-dependencies"
    dev_suffix = "-dev" if dev else ""
    subprocess.run(
        (
            f"conda-lock render {dev_dep_option} --extras full -k env"
            f" --filename-template conda-py-{py_ver}-{{platform}}{dev_suffix}.lock conda-py-{py_ver}-lock.yml"
        ),
        shell=True,
        check=True,
        stderr=subprocess.STDOUT,
    )


for py_ver in ["3.9", "3.8", "3.7"]:
    create_multi_platform_lock(py_ver)
    per_platform_env(py_ver, dev=False)
    per_platform_env(py_ver, dev=True)
