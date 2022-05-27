#!/usr/bin/env python3

#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import os
import subprocess
from datetime import datetime
from pathlib import Path

from run_conda_lock import per_platform_env

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%G%m%d-%H%M%S")
    py_ver = "3.9"
    env_name = f"geoapps-simpeg-{timestamp}"

    suffix = f"-simpeg{timestamp}"
    per_platform_env(py_ver, full=False, dev=False, suffix=suffix)
    lock_file = Path(f"conda-py-{py_ver}-win-64{suffix}.lock.yml")
    assert lock_file.is_file()

    print(f"# Install environment {env_name} from {lock_file.name}")
    try:
        subprocess.run(
            f"""conda env create -f {lock_file.name} -n {env_name} ^
            && conda activate {env_name} && python -m pip install . -v --no-deps ^
            && conda deactivate
            """,
            env=dict(os.environ, PYTHONUTF8="1"),
            shell=True,
            check=True,
            stderr=subprocess.STDOUT,
        )
        subprocess.run(
            f"conda-pack -n {env_name} -o {env_name}.zip",
            shell=True,
            check=True,
            stderr=subprocess.STDOUT,
        )
        subprocess.run(
            f"conda remove -n {env_name} --all -y",
            shell=True,
            check=True,
            stderr=subprocess.STDOUT,
        )
        lock_file.unlink()
    finally:
        for p in ("osx-64", "linux-64"):
            f = Path(f"conda-py-{py_ver}-{p}{suffix}.lock.yml")
            if f.is_file():
                f.unlink()
