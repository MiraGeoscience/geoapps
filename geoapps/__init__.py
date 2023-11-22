#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

__version__ = "0.11.0-rc.2.post.1"

import os
import warnings
from pathlib import Path


def assets_path() -> Path:
    """Return the path to the assets folder."""

    assets_dir_env_var = "GEOAPPS_ASSETS_DIR"
    assets_dirname = os.environ.get(assets_dir_env_var, None)
    if assets_dirname:
        assets_folder = Path(assets_dirname)
        if not assets_folder.is_dir():
            warnings.warn(
                f"Custom assets folder not found: {assets_dir_env_var}={assets_dirname}"
            )
        else:
            return assets_folder

    parent = Path(__file__).parent
    folder_name = f"{parent.name}-assets"
    assets_folder = parent.parent / folder_name
    if not assets_folder.is_dir():
        raise RuntimeError(f"Assets folder not found: {assets_folder}")

    return assets_folder
