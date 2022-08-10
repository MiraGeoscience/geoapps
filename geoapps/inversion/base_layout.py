#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from dash import dcc, html


def generate_model_component(
    param_name: str, param_label: str, units: str, model_type: str = None
) -> list:
    """
    Generate dash components for starting and reference model.
    """
    if model_type == "reference":
        radio_options = ["Constant", "Model", "None"]
        label_prefix = "Reference "
        param_prefix = "ref_"
    elif model_type == "starting":
        radio_options = ["Constant", "Model"]
        label_prefix = "Starting "
        param_prefix = "starting_"
    else:
        radio_options = ["Constant", "Model", "None"]
        label_prefix = ""
        param_prefix = ""

    component = html.Div(
        [
            dcc.Markdown(label_prefix + param_label),
            dcc.RadioItems(
                id=param_prefix + param_name + "_options",
                options=radio_options,
                value="Constant",
            ),
            html.Div(
                id=param_prefix + param_name + "_const_div",
                children=[
                    dcc.Markdown(units),
                    dcc.Input(id=param_prefix + param_name + "_const"),
                ],
            ),
            html.Div(
                id=param_prefix + param_name + "_mod_div",
                children=[
                    dcc.Markdown("Object"),
                    dcc.Input(id=param_prefix + param_name + "_obj"),
                    dcc.Markdown("Values"),
                    dcc.Input(id=param_prefix + param_name + "_data"),
                ],
            ),
        ]
    )
    return component


def get_object_selection_layout(inversion_type_list):
    if ("magnetic vector" in inversion_type_list) or (
        "magnetic scalar" in inversion_type_list
    ):
        object_selection_layout = html.Div(
            [
                dcc.Upload(
                    id="upload",
                    children=html.Button("Upload Workspace/ui.json"),
                ),
                dcc.Markdown("Object:"),
                dcc.Dropdown(id="object"),
                dcc.Markdown("Inversion Type:"),
                dcc.Dropdown(
                    id="inversion_type",
                    options=inversion_type_list,
                    value=inversion_type_list[0],
                ),
                inducing_params_div,
            ],
            style={"border": "2px black solid"},
        )
    else:
        object_selection_layout = html.Div(
            [
                dcc.Upload(
                    id="upload",
                    children=html.Button("Upload Workspace/ui.json"),
                ),
                dcc.Markdown("Object:"),
                dcc.Dropdown(id="object"),
                dcc.Markdown("Inversion Type:"),
                dcc.Dropdown(
                    id="inversion_type",
                    options=inversion_type_list,
                    value=inversion_type_list[0],
                ),
            ],
            style={"border": "2px black solid"},
        )
    return object_selection_layout


inducing_params_div = html.Div(
    id="inducing_params_div",
    children=[
        dcc.Markdown("Inducing Field Parameters"),
        dcc.Markdown("Amplitude (nT)"),
        dcc.Input(id="amplitude"),
        dcc.Markdown("Inclination (d.dd)"),
        dcc.Input(id="inclination"),
        dcc.Markdown("Declination (d.dd)"),
        dcc.Input(id="declination"),
    ],
    style={"display": "none"},
)

input_data_layout = html.Div(
    [
        html.Div(
            [
                dcc.Markdown("Input Data"),
                dcc.Markdown("Component:"),
                dcc.Dropdown(id="component"),
                dcc.Checklist(
                    id="active", options=[{"label": "Active", "value": True}]
                ),
                dcc.Markdown("Channel:"),
                dcc.Dropdown(id="channel"),
                dcc.Markdown("Uncertainties"),
                dcc.Markdown("Floor:"),
                dcc.Dropdown(id="floor"),
                dcc.Markdown("(Optional) Channel:"),
                dcc.Dropdown(id="optional_channel"),
            ]
        ),
        html.Div(
            [
                dcc.Markdown("Grid Resolution (m)"),
                dcc.Dropdown(id="resolution"),
                dcc.Markdown("Data Count:"),
                dcc.Markdown(id="data_count"),
                dcc.Graph(id="plot"),
            ]
        ),
    ],
    style={"border": "2px black solid"},
)

topography_none = html.Div(
    id="topography_none", children=[dcc.Markdown("No topography")]
)
topography_object = html.Div(
    id="topography_object",
    children=[
        dcc.Markdown("Object:"),
        dcc.Dropdown(id="topo_object"),
        dcc.Markdown("Data:"),
        dcc.Dropdown(id="topo_data"),
    ],
)
topography_sensor = html.Div(
    id="topography_sensor",
    children=[
        dcc.Markdown("Vertical offset (+ve up)"),
        dcc.Input(id="vertical_offset"),
    ],
)
topography_constant = html.Div(
    id="topography_constant",
    children=[dcc.Markdown("Elevation (m)"), dcc.Input(id="elevation")],
)

topography_layout = html.Div(
    [
        html.Div(
            [
                dcc.Markdown("Topography"),
                dcc.Markdown("Define by:"),
                dcc.RadioItems(
                    id="topography_options",
                    options=["None", "Object", "Relative to Sensor", "Constant"],
                    value="None",
                ),
                topography_none,
                topography_object,
                topography_sensor,
                topography_constant,
            ]
        ),
        html.Div(
            [
                dcc.Markdown("Sensor Location"),
                dcc.Checklist(
                    id="z_from_topo",
                    options=[{"label": "Set Z from topo + offsets", "value": True}],
                ),
                dcc.Markdown("Offsets"),
                dcc.Markdown("dx (+East)"),
                dcc.Input(id="dx"),
                dcc.Markdown("dy (+North)"),
                dcc.Input(id="dy"),
                dcc.Markdown("dz (+ve up)"),
                dcc.Input(id="dz"),
                dcc.Markdown("Radar (Optional):"),
                dcc.Dropdown(id="radar"),
            ]
        ),
    ],
    style={"border": "2px black solid"},
)

mesh = html.Div(
    id="mesh_div",
    children=[
        dcc.Markdown("Object:"),
        dcc.Dropdown(id="mesh_object"),
        dcc.Markdown("Core cell size (u, v, z)"),
        dcc.Input(id="core_cell_size_u"),
        dcc.Input(id="core_cell_size_v"),
        dcc.Input(id="core_cell_size_z"),
        dcc.Markdown("Refinement Layers"),
        dcc.Markdown("Number of cells below topography"),
        dcc.Input(id="cells_below_topo"),
        dcc.Markdown("Number of cells below sensors"),
        dcc.Input(id="cells_below_sensors"),
        dcc.Markdown("Maximum distance (m)"),
        dcc.Input(id="maximum_distance"),
        dcc.Markdown("Dimensions"),
        dcc.Markdown("Horizontal padding (m)"),
        dcc.Input(id="horizontal_padding"),
        dcc.Markdown("Vertical padding (m)"),
        dcc.Input(id="vertical_padding"),
        dcc.Markdown("Minimum depth (m)"),
        dcc.Input(id="min_depth"),
    ],
)

regularization = html.Div(
    id="regularization_div",
    children=[
        html.Div(
            [
                html.Div(
                    [
                        dcc.Markdown("Reference Model (s)"),
                        dcc.Markdown("EW-gradient (x)"),
                        dcc.Markdown("NS_gradient (y)"),
                        dcc.Markdown("Vertical-gradient (z)"),
                    ]
                ),
                html.Div(
                    [
                        dcc.Markdown("Scaling (alphas)"),
                        dcc.Input(id="alpha_s"),
                        dcc.Input(id="alpha_x"),
                        dcc.Input(id="alpha_y"),
                        dcc.Input(id="alpha_z"),
                    ]
                ),
                html.Div(
                    [
                        dcc.Markdown("Lp-norms"),
                        dcc.Input(id="s_norm"),
                        dcc.Input(id="x_norm"),
                        dcc.Input(id="y_norm"),
                        dcc.Input(id="z_norm"),
                    ]
                ),
            ]
        ),
    ],
)

upper_lower_bounds = html.Div(
    id="upper_lower_bounds_div",
    children=[
        generate_model_component("lower_bounds", "Lower Bounds", "Units"),
        generate_model_component("upper_bounds", "Upper Bounds", "Units"),
    ],
)

ignore_values = html.Div(
    id="ignore_values_div",
    children=[
        dcc.Markdown("Value (i.e. '<0' for no negatives)"),
        dcc.Input(id="ignore_values"),
    ],
)

optimization = html.Div(
    id="optimization_div",
    children=[
        html.Div(
            [
                dcc.Markdown("Max beta Iterations"),
                dcc.Markdown("Target misfit"),
                dcc.Markdown("Beta ratio (phi_d/phi_m)"),
                dcc.Markdown("Max CG Iterations"),
                dcc.Markdown("CG Tolerance"),
                dcc.Markdown("Max CPUs"),
                dcc.Markdown("Number of tiles"),
            ]
        ),
        html.Div(
            [
                dcc.Input(id="max_beta_iterations"),
                dcc.Input(id="target_misfit"),
                dcc.Input(id="beta_ratio"),
                dcc.Input(id="max_cg_iterations"),
                dcc.Input(id="cg_tol"),
                dcc.Input(id="max_cpus"),
                dcc.Input(id="num_tiles"),
            ]
        ),
    ],
)

inversion_parameters_dropdown = html.Div(
    [
        dcc.Markdown("Inversion Parameters"),
        dcc.Checklist(id="forward", options=[{"label": "Forward only", "value": True}]),
        dcc.Dropdown(
            id="param_dropdown",
            options=[
                "starting model",
                "mesh",
                "reference model",
                "regularization",
                "upper-lower bounds",
                "detrend",
                "ignore values",
                "optimization",
            ],
            value="starting model",
        ),
    ]
)

output_layout = html.Div(
    [
        dcc.Markdown("Output"),
        dcc.Markdown("Save as:"),
        dcc.Input(id="ga_group_name"),
        dcc.Markdown("Output path:"),
        dcc.Input(id="monitoring_directory"),
        html.Button("Write input", id="write_input"),
        html.Button("Compute", id="compute"),
    ],
    style={"border": "2px black solid"},
)
