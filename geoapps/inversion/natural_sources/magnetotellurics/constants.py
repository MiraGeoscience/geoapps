# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024-2025 Mira Geoscience Ltd.                                '
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
    "geoh5": str(assets_path() / "FlinFlon_natural_sources.geoh5"),
    "topography_object": UUID("{cfabb8dd-d1ad-4c4e-a87c-7b3dd224c3f5}"),
    "data_object": UUID("{9664afc1-cbda-4955-b936-526ca771f517}"),
    "zxx_real_channel": UUID("{a73159fc-8c1b-411a-b435-12a5dac4a209}"),
    "zxx_real_uncertainty": UUID("{e752e8d8-e8e3-4575-b20c-bc2d37cbd269}"),
    "zxx_imag_channel": UUID("{46271e74-9573-4cd6-8bcb-4c45495fe539}"),
    "zxx_imag_uncertainty": UUID("{73f77c42-ab78-4972-bb69-b16c990bf7dc}"),
    "zxy_real_channel": UUID("{40bdf2a1-237f-49e4-baa8-a7c0785f369a}"),
    "zxy_real_uncertainty": UUID("{8802e943-354f-4ce4-a81f-dde9ef08b8ec}"),
    "zxy_imag_channel": UUID("{1a135542-b2be-4096-9629-a0bc4357970d}"),
    "zxy_imag_uncertainty": UUID("{fac85198-cbd2-4510-bce7-12b4b5fcae2f}"),
    "zyx_real_channel": UUID("{21e6737d-de1a-4af4-9c92-aeeeb6eecf34}"),
    "zyx_real_uncertainty": UUID("{08141050-365c-40aa-bcfb-54841c9492ce}"),
    "zyx_imag_channel": UUID("{f1d2750a-99bf-4876-833b-19b9f46124a4}"),
    "zyx_imag_uncertainty": UUID("{2664535c-295a-4e2a-b403-2a57a821fe08}"),
    "zyy_real_channel": UUID("{9b7f06e9-5bfb-4a5e-ba90-9cec9990d7d5}"),
    "zyy_real_uncertainty": UUID("{61d1a3e9-f7ff-4fd8-bc61-2d1b24b9adc6}"),
    "zyy_imag_channel": UUID("{c9133116-043b-40d9-853d-21f6357f927f}"),
    "zyy_imag_uncertainty": UUID("{11ebb4f3-eacf-4558-b240-b958526dd273}"),
    "mesh": UUID("{1200396b-bc4a-4519-85e1-558c2dcac1dd}"),
    "starting_model": 0.0003,
    "reference_model": 0.0003,
    "background_conductivity": 0.0003,
    "octree_levels_topo": [0, 0, 4, 4],
    "octree_levels_obs": [4, 4, 4, 4],
    "depth_core": 500.0,
    "horizontal_padding": 1000.0,
    "vertical_padding": 1000.0,
    "s_norm": 0.0,
    "x_norm": 2.0,
    "y_norm": 2.0,
    "z_norm": 2.0,
    "upper_bound": 100.0,
    "lower_bound": 1e-5,
    "max_global_iterations": 50,
}
