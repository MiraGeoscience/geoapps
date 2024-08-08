# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024 Mira Geoscience Ltd.                                     '
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
    "geoh5": str(assets_path() / "FlinFlon_dcip.geoh5"),
    "data_object": UUID("{6e14de2c-9c2f-4976-84c2-b330d869cb82}"),
    "chargeability_channel": UUID("{162320e6-2b80-4877-9ec1-a8f5b6a13673}"),
    "chargeability_uncertainty": 0.001,
    "line_object": UUID("{d400e8f1-8460-4609-b852-b3b93f945770}"),
    "line_id": 5,
    "mesh": UUID("{537cdf17-28c9-4baa-a1ac-07c37662583d}"),
    "starting_model": 1e-4,
    "conductivity_model": 0.1,
    "s_norm": 0.0,
    "x_norm": 2.0,
    "z_norm": 2.0,
    "upper_bound": 100.0,
    "lower_bound": 1e-5,
    "max_global_iterations": 25,
    "topography_object": UUID("{ab3c2083-6ea8-4d31-9230-7aad3ec09525}"),
    "topography": UUID("{a603a762-f6cb-4b21-afda-3160e725bf7d}"),
    "z_from_topo": True,
    "receivers_offset_z": 0.0,
}
