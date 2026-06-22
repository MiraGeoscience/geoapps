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
from typing import ClassVar

from geoapps_utils.base import Options
from geoh5py.data import Data
from geoh5py.objects import ObjectBase
from pydantic import Field

from geoapps import assets_path


class ScatterPlotParams(Options):
    """
    Parameter class for data interpolation application.
    """

    name: ClassVar[str] = "scatter_plot"
    default_ui_json: ClassVar[Path] = assets_path() / "uijson/scatter.ui.json"
    title: str = "Geoapps - 3D Scatter plot"
    run_command: str = "geoapps.scatter_plot.driver"
    conda_environment: str = "geoapps"

    objects: ObjectBase
    downsampling: int = Field(100, ge=1, le=100)
    x: Data
    x_log: bool = False
    x_min: float | None = None
    x_max: float | None = None
    x_thresh: float | None = None
    y: Data
    y_log: bool = False
    y_min: float | None = None
    y_max: float | None = None
    y_thresh: float | None = None
    z: Data | None = None
    z_log: bool = False
    z_min: float | None = None
    z_max: float | None = None
    z_thresh: float | None = None
    color: Data | None = None
    color_log: bool = False
    color_min: float | None = None
    color_max: float | None = None
    color_maps: str = "inferno"
    color_thresh: float | None = None
    size: Data | None = None
    size_log: bool = False
    size_min: float | None = None
    size_max: float | None = None
    size_thresh: float | None = None
    size_markers: float = Field(1.0, gt=0, le=100)
