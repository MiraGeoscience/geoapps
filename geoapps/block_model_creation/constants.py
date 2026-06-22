# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024-2026 Mira Geoscience Ltd.                                '
#                                                                              '
#  This file is part of geoapps.                                               '
#                                                                              '
#  geoapps is distributed under the terms and conditions of the MIT License    '
#  (see LICENSE file at the root of this source code package).                 '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from __future__ import annotations

from uuid import UUID

from geoapps import assets_path


app_initializer = {
    "geoh5": str(assets_path() / "FlinFlon.geoh5"),
    "monitoring_directory": str((assets_path() / "Temp").resolve()),
    "objects": UUID("{2e814779-c35f-4da0-ad6a-39a6912361f9}"),
    "cell_size_x": 50.0,
    "cell_size_y": 50.0,
    "cell_size_z": 50.0,
    "depth_core": 500.0,
    "expansion_factor": 1.05,
    "new_grid": "BlockModel",
    "horizontal_padding": 500.0,
    "bottom_padding": 500.0,
}
