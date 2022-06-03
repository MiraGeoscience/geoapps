#!/usr/bin/env python3

#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

"""
Creates a zipped conda environment that can be unpacked to run simpeg.

The environment is created for Windows, and uses Python 3.9

Usage: from a the conda base environment, at the root of the project:
> python devtools/pack_simpeg_env.py

To prepare the conda base environment, see devtools/setup-conda-base.bat
"""

import os
import subprocess
from datetime import datetime
from pathlib import Path

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%G%m%d-%H%M%S")
    env_name = f"geoapps-simpeg-{timestamp}"

    py_ver = "3.9"
    platform = "win-64"
    lock_file = Path(f"environments/conda-py-{py_ver}-{platform}-simpeg.lock.yml")
    assert lock_file.is_file()

    try:
        print(f"# Create environment {env_name} from {lock_file}")
        subprocess.run(
            f"""conda env create -f {lock_file} -n {env_name} ^
            && conda activate {env_name} && python -m pip install . -v --no-deps ^
            && conda deactivate
            """,
            env=dict(os.environ, PYTHONUTF8="1"),
            shell=True,
            check=True,
            stderr=subprocess.STDOUT,
        )
        print(f"# Pack environment {env_name}")
        subprocess.run(
            f"conda-pack -n {env_name} -o {env_name}.zip",
            shell=True,
            check=True,
            stderr=subprocess.STDOUT,
        )
    finally:
        subprocess.run(
            f"conda remove -n {env_name} --all -y",
            shell=True,
            check=True,
            stderr=subprocess.STDOUT,
        )
