#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from dash import dcc, html

import geoapps.inversion.base_inversion_layout as base_layout

object_selection_layout = base_layout.get_object_selection_layout("gravity")

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
input_data_layout = base_layout.get_input_data_layout(component_list)

gravity_inversion_params = {"density": {"label": "Density", "units": "g/cc"}}
inversion_params_layout = base_layout.get_inversion_params_layout(
    gravity_inversion_params
)

# Full app layout
gravity_layout = html.Div(
    [
        object_selection_layout,
        input_data_layout,
        base_layout.topography_layout,
        inversion_params_layout,
        base_layout.output_layout,
    ],
    style={"width": "90%", "margin-left": "10px", "margin-right": "10px"},
)
