# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024-2026 Mira Geoscience Ltd.                                '
#                                                                              '
#  This file is part of geoapps.                                               '
#                                                                              '
#  geoapps is distributed under the terms and conditions of the MIT License    '
#  (see LICENSE file at the root of this source code package).                 '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal

from geoapps_utils.base import Options
from geoh5py.data import Data
from geoh5py.objects import ObjectBase
from geoh5py.ui_json import UIJson
from pydantic import ConfigDict

from geoapps import assets_path

from .constants import default_ui_json


class DataInterpolationParams(Options):
    """
    Parameter class for data interpolation application.
    """

    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)

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
    space: Literal["Linear", "Log"] = "Log"
    method: Literal["Inverse Distance", "Nearest"] = "Inverse Distance"
    max_distance: float | None = None
    xy_extent: ObjectBase | None = None
    topography_objects: ObjectBase | None
    topography_data: Data | None
    max_depth: float | None = None
    no_data_value: float | None = None
    ga_group_name: str = "_Interp"

    @classmethod
    def get_default_ui_json(cls) -> UIJson:
        """
        Load the driver's default ui.json template from disk
        with no parameters filled in.

        :return: The default ui.json configuration.
        """
        return UIJson.from_dict(default_ui_json)
