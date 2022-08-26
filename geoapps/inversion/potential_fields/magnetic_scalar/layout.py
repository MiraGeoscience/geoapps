#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from dash import html

import geoapps.inversion.base_inversion_layout as base_layout

component_list = [
    "tmi",
    "bx",
    "by",
    "bz",
    "bxx",
    "bxy",
    "bxz",
    "byy",
    "byz",
    "bzz",
]

object_selection_layout = base_layout.get_object_selection_layout(
    "magnetic_scalar", component_list
)

magnetic_scalar_inversion_params = {
    "susceptibility": {"label": "Susceptibility", "units": "SI"}
}
inversion_params_layout = base_layout.get_inversion_params_layout(
    magnetic_scalar_inversion_params
)

# Full app layout
magnetic_scalar_layout = html.Div(
    [
        object_selection_layout,
        base_layout.plot_layout,
        inversion_params_layout,
        base_layout.output_layout,
    ],
    style={"width": "90%", "margin-left": "10px", "margin-right": "10px"},
)
