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
    "resolution": 50.0,
    "tmi_channel": UUID("{a342e416-946a-4162-9604-6807ccb06073}"),
    "tmi_uncertainty": 10.0,
    "tmi_channel_bool": True,
    "mesh": UUID("{f6b08e3b-9a85-45ab-a487-4700e3ca1917}"),
    "inducing_field_strength": 60000.0,
    "inducing_field_inclination": 79.0,
    "inducing_field_declination": 11.0,
    "window_center_x": 314600.0,
    "window_center_y": 6072300.0,
    "window_width": 1000.0,
    "window_height": 1500.0,
    "window_azimuth": 0.0,
    "s_norm": 0.0,
    "x_norm": 2.0,
    "y_norm": 2.0,
    "z_norm": 2.0,
    "starting_model": 1e-4,
    "max_global_iterations": 25,
    "topography_object": UUID("{ab3c2083-6ea8-4d31-9230-7aad3ec09525}"),
    "topography": UUID("{a603a762-f6cb-4b21-afda-3160e725bf7d}"),
    "z_from_topo": True,
    "receivers_offset_z": 60.0,
    "fix_aspect_ratio": True,
    "colorbar": False,
    "n_cpu": int(multiprocessing.cpu_count() / 2),
}
