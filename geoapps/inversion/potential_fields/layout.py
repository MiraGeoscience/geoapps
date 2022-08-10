#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from dash import dcc, html

from geoapps.inversion.base_layout import (
    generate_model_component,
    ignore_values,
    input_data_layout,
    mesh,
    object_selection_layout,
    optimization,
    output_layout,
    regularization,
    topography_layout,
    upper_lower_bounds,
    workspace_layout,
)

inducing_params_layout = html.Div(
    [
        dcc.Markdown("Inducing Field Parameters"),
        dcc.Markdown("Amplitude (nT)"),
        dcc.Input(id="amplitude"),
        dcc.Markdown("Inclination (d.dd)"),
        dcc.Input(id="inclination"),
        dcc.Markdown("Declination (d.dd)"),
        dcc.Input(id="declination"),
    ]
)

starting_model_components = []
reference_model_components = []
param_names = {
    "susceptibility": {"label": "Effective Susceptibility", "units": "SI"},
    "inclination": {"label": "Inclination", "units": "Degree"},
    "declination": {"label": "Declination", "units": "Degree"},
}

for param, value in param_names.items():
    starting_model_components.append(
        generate_model_component(param, value["label"], value["units"], "starting")
    )
    reference_model_components.append(
        generate_model_component(param, value["label"], value["units"], "reference")
    )

starting_model = html.Div(starting_model_components)

reference_model = html.Div(reference_model_components)

detrend = html.Div(
    [
        dcc.Markdown("Method"),
        dcc.Dropdown(id="method", options=["all", "perimeter"]),
        dcc.Markdown("Order"),
        dcc.Input(id="order"),
    ]
)

inversion_parameters_layout = html.Div(
    [
        starting_model,
        mesh,
        reference_model,
        regularization,
        upper_lower_bounds,
        detrend,
        ignore_values,
        optimization,
    ]
)

# Full app layout
potential_fields_layout = html.Div(
    [
        workspace_layout,
        object_selection_layout,
        input_data_layout,
        topography_layout,
        inversion_parameters_layout,
        output_layout,
    ],
)
