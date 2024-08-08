# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024 Mira Geoscience Ltd.                                     '
#                                                                              '
#  This file is part of geoapps.                                               '
#                                                                              '
#  geoapps is distributed under the terms and conditions of the MIT License    '
#  (see LICENSE file at the root of this source code package).                 '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from __future__ import annotations

import multiprocessing
from uuid import UUID

from geoapps import assets_path


app_initializer = {
    "geoh5": str(assets_path() / "FlinFlon.geoh5"),
    "monitoring_directory": str((assets_path() / "Temp").resolve()),
    "forward_only": False,
    "data_object": UUID("{7aaf00be-adbf-4540-8333-8ac2c2a3c31a}"),
    "gxx_channel": UUID("{3d7ace18-e9c5-4cef-9ca3-8adc12fd53c4}"),
    "gxx_uncertainty": 1.0,
    "gyy_channel": UUID("{1d001501-3d84-4afb-8e24-6be267827ae0}"),
    "gyy_uncertainty": 1.0,
    "gzz_channel": UUID("{82e34b29-a6f7-4488-944c-ff5bd8580a13}"),
    "gzz_uncertainty": 1.0,
    "gxy_channel": UUID("{a960226f-e69c-4131-9855-cd59d98ca994}"),
    "gxy_uncertainty": 1.0,
    "gxz_channel": UUID("{1daec416-29b6-4e66-8a25-b366ef41bb03}"),
    "gxz_uncertainty": 1.0,
    "gyz_channel": UUID("{45a05273-3d57-45de-b435-d752077bb2f4}"),
    "gyz_uncertainty": 1.0,
    "mesh": UUID("{f6b08e3b-9a85-45ab-a487-4700e3ca1917}"),
    "resolution": 50.0,
    "window_center_x": 314565.0,
    "window_center_y": 6072334.0,
    "window_width": 1000.0,
    "window_height": 1500.0,
    "window_azimuth": 0.0,
    "s_norm": 0.0,
    "x_norm": 2.0,
    "y_norm": 2.0,
    "z_norm": 2.0,
    "starting_model": 1e-3,
    "max_global_iterations": 25,
    "topography_object": UUID("{ab3c2083-6ea8-4d31-9230-7aad3ec09525}"),
    "topography": UUID("{a603a762-f6cb-4b21-afda-3160e725bf7d}"),
    "z_from_topo": True,
    "receivers_offset_z": 60.0,
    "fix_aspect_ratio": True,
    "colorbar": False,
    "n_cpu": int(multiprocessing.cpu_count() / 2),
}
