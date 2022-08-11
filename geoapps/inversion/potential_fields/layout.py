#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from dash import dcc, html

import geoapps.inversion.base_layout as base_layout

object_selection_layout = base_layout.get_object_selection_layout(
    ["magnetic vector", "magnetic scalar", "gravity"]
)

inversion_params = {
    "magnetic_vector": {
        "eff_susceptibility": {"label": "Effective Susceptibility", "units": "SI"},
        "inclination": {"label": "Inclination", "units": "Degree"},
        "declination": {"label": "Declination", "units": "Degree"},
    },
    "magnetic_scalar": {
        "susceptibility": {"label": "Susceptibility", "units": "SI"},
    },
    "gravity": {
        "density": {"label": "Density", "units": "g/cc"},
    },
}
inversion_params_layout = base_layout.get_inversion_params_layout(inversion_params)

# Full app layout
potential_fields_layout = html.Div(
    [
        object_selection_layout,
        base_layout.input_data_layout,
        base_layout.topography_layout,
        inversion_params_layout,
        base_layout.output_layout,
    ],
)
