#!/usr/bin/env python3

#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import subprocess

for py_ver in ["3.7", "3.8", "3.9"]:
    print(f"# for Python version {py_ver}")
    subprocess.run(
        [
            "conda-lock",
            "lock",
            "-f",
            "environment.yml",
            "-f",
            f"env-python-{py_ver}.yml",
            "--filename-template",
            f"conda-py-{py_ver}-{{platform}}.lock",
        ]
    )
