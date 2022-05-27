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

timestamp = datetime.now().strftime("%G%m%d-%H%M%S")
py_ver = "3.9"
lock_file = f"conda-py-{py_ver}-lock.yml"
env_name = f"geoapps-simpeg-{timestamp}"

print(f"# Install environment {env_name} from {lock_file}")
subprocess.run(
    (
        f"conda-lock install --no-dev -n {env_name} {lock_file} ^"
        f"&& conda activate {env_name} && python -m pip install . -v --no-deps ^"
        "&& conda deactivate ^"
        f"conda-pack -n {env_name} -o {env_name}.zip",
    ),
    env=dict(os.environ, PYTHONUTF8="1"),
    shell=True,
    check=True,
    stderr=subprocess.STDOUT,
)
