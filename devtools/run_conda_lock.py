#!/usr/bin/env python3

#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import subprocess

for py_ver in ["3.9", "3.8", "3.7"]:
    print(f"# Create multi-platform lock file for Python {py_ver}")
    subprocess.run(
        f"conda-lock lock -f pyproject.toml -f env-python-{py_ver}.yml --lockfile conda-py-{py_ver}-lock.yml",
        shell=True,
        check=True,
        stderr=subprocess.STDOUT,
    )

    print(
        f"# Create per platform Conda env files for Python {py_ver} (no dev dependencies) "
    )
    subprocess.run(
        (
            "conda-lock render --no-dev-dependencies --extras full -k env"
            f" --filename-template conda-py-{py_ver}-{{platform}}.lock conda-py-{py_ver}-lock.yml"
        ),
        shell=True,
        check=True,
        stderr=subprocess.STDOUT,
    )
