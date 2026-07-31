# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024-2026 Mira Geoscience Ltd.                                '
#                                                                              '
#  This file is part of geoapps.                                               '
#                                                                              '
#  geoapps is distributed under the terms and conditions of the MIT License    '
#  (see LICENSE file at the root of this source code package).                 '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from __future__ import annotations

from typing import ClassVar, Literal

from pathlib import Path

from geoapps_utils.base import Options
from geoh5py.data import Data
from geoh5py.objects import ObjectBase

from geoapps import assets_path

class DataInterpolationParams(Options):
    """
    Parameter class for data interpolation application.
    """
    name: ClassVar[str] = "data_interpolation"
    default_ui_json: ClassVar[Path] = (
            assets_path() / "uijson/implicit_modelling.ui.json"
    )
    title: str = "Geoapps - Data Interpolation"
    run_command: str = "geoapps.interpolation.driver"
    conda_environment: str = "geoapps"

    objects: ObjectBase
    data: Data
    out_object: ObjectBase

    skew_angle: float | None = None
    skew_factor: float | None = None
    space: Literal["linear", "log"] = "log"
    max_distance: float | None = None
    xy_extent: ObjectBase | None
    topography_objects: ObjectBase | None
    topography_data: Data | None
    max_depth: float | None = None
    no_data_value: float | None = None
