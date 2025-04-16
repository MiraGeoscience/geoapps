# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024-2025 Mira Geoscience Ltd.                                '
#                                                                              '
#  This file is part of geoapps.                                               '
#                                                                              '
#  geoapps is distributed under the terms and conditions of the MIT License    '
#  (see LICENSE file at the root of this source code package).                 '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from dash import html

import geoapps.inversion.base_inversion_layout as base_layout


component_list = [
    "gx",
    "gy",
    "gz",
    "gxx",
    "gxy",
    "gxz",
    "gyy",
    "gyz",
    "gzz",
    "guv",
]

object_selection_layout = base_layout.get_object_selection_layout(
    "gravity", component_list
)

gravity_inversion_params = {"model": {"label": "Density", "units": "g/cc"}}
inversion_params_layout = base_layout.get_inversion_params_layout(
    gravity_inversion_params
)

# Full app layout
gravity_layout = html.Div(
    [
        object_selection_layout,
        base_layout.plot_layout,
        inversion_params_layout,
        base_layout.output_layout,
    ],
    style={"width": "90%", "margin-left": "10px", "margin-right": "10px"},
)
