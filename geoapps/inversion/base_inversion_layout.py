#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from dash import dcc, html


def get_object_selection_layout(inversion_type_list):
    standard_layout = html.Div(
        [
            dcc.Upload(
                id="upload",
                children=html.Button("Upload Workspace/ui.json"),
            ),
            dcc.Store(id="ui_json"),
            html.Div(
                [
                    dcc.Markdown(
                        "Object:", style={"display": "inline-block", "width": "30%"}
                    ),
                    dcc.Dropdown(
                        id="data_object",
                        style={"display": "inline-block", "width": "70%"},
                    ),
                ]
            ),
            html.Div(
                [
                    dcc.Markdown(
                        "Inversion Type:",
                        style={"display": "inline-block", "width": "30%"},
                    ),
                    dcc.Dropdown(
                        id="inversion_type",
                        options=inversion_type_list,
                        style={"display": "inline-block", "width": "70%"},
                    ),
                ]
            ),
        ],
        style={
            "display": "inline-block",
            "vertical-align": "top",
            "width": "40%",
            "margin-right": "5%",
        },
    )
    if ("magnetic vector" in inversion_type_list) or (
        "magnetic scalar" in inversion_type_list
    ):
        object_selection_layout = html.Div(
            [
                dcc.Markdown("##### **Object Selection**"),
                standard_layout,
                inducing_params_div,
            ],
            style={
                "border": "2px black solid",
                "padding-left": "10px",
                "padding-right": "10px",
            },
        )
    else:
        object_selection_layout = html.Div(
            [dcc.Markdown("##### **Object Selection**"), standard_layout],
            style={
                "border": "2px black solid",
                "padding-left": "10px",
                "padding-right": "10px",
            },
        )
    return object_selection_layout


def generate_model_component(
    param_name: str, param_label: str, units: str, model_type: str = None
) -> list:
    """
    Generate dash components for starting and reference model.
    """
    if model_type == "reference":
        radio_options = ["Constant", "Model", "None"]
        label_prefix = "Reference "
        param_prefix = "reference_"
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
            dcc.Markdown("**" + label_prefix + param_label + "**"),
            dcc.RadioItems(
                id=param_prefix + param_name + "_options",
                options=radio_options,
                value="Constant",
            ),
            html.Div(
                id=param_prefix + param_name + "_const_div",
                children=[
                    html.Div(
                        [
                            dcc.Markdown(units, style={"display": "inline-block"}),
                            dcc.Input(
                                id=param_prefix + param_name + "_const",
                                style={"display": "inline-block"},
                            ),
                        ]
                    ),
                ],
            ),
            html.Div(
                id=param_prefix + param_name + "_mod_div",
                children=[
                    html.Div(
                        [
                            dcc.Markdown("Object", style={"display": "inline-block"}),
                            dcc.Dropdown(
                                id=param_prefix + param_name + "_object",
                                style={"display": "inline-block"},
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            dcc.Markdown("Values", style={"display": "inline-block"}),
                            dcc.Dropdown(
                                id=param_prefix + param_name + "_data",
                                style={"display": "inline-block"},
                            ),
                        ]
                    ),
                ],
            ),
        ]
    )
    return component


def get_inversion_params_layout(inversion_params):
    starting_model_divs = []
    reference_model_divs = []
    for inversion_type, params in inversion_params.items():
        starting_model_components = []
        reference_model_components = []
        for param_name, value in params.items():
            starting_model_components.append(
                generate_model_component(
                    param_name, value["label"], value["units"], "starting"
                )
            )
            reference_model_components.append(
                generate_model_component(
                    param_name, value["label"], value["units"], "reference"
                )
            )
        starting_model_divs.append(
            html.Div(
                id="starting_" + inversion_type, children=starting_model_components
            )
        )
        reference_model_divs.append(
            html.Div(
                id="reference_" + inversion_type, children=reference_model_components
            )
        )

    starting_model = html.Div(
        id="starting_model_div",
        children=[dcc.Markdown("###### **Starting Model**")] + starting_model_divs,
        style={"display": "inline-block", "vertical-align": "top", "width": "50%"},
    )
    reference_model = html.Div(
        id="reference_model_div",
        children=[dcc.Markdown("###### **Reference Model**")] + reference_model_divs,
        style={"display": "inline-block", "vertical-align": "top", "width": "50%"},
    )

    inversion_params_layout = html.Div(
        [
            dcc.Markdown("##### **Inversion Parameters**"),
            dcc.Checklist(
                id="forward_only", options=[{"label": "Forward only", "value": True}]
            ),
            dcc.Checklist(
                id="core_params",
                options=[{"label": "Core parameters", "value": True}],
                value=[True],
            ),
            dcc.Checklist(
                id="advanced_params",
                options=[{"label": "Advanced parameters", "value": True}],
                value=[],
            ),
            html.Hr(),
            html.Div(
                id="core_params_div",
                children=[
                    starting_model,
                    html.Hr(),
                    mesh,
                    html.Hr(),
                    reference_model,
                ],
                style={
                    "border": "2px black solid",
                    "padding-left": "10px",
                    "padding-right": "10px",
                },
            ),
            html.Div(
                id="advanced_params_div",
                children=[
                    regularization,
                    html.Hr(),
                    upper_lower_bounds,
                    html.Hr(),
                    detrend,
                    html.Hr(),
                    ignore_values,
                    html.Hr(),
                    optimization,
                ],
                style={
                    "border": "2px black solid",
                    "padding-left": "10px",
                    "padding-right": "10px",
                },
            ),
        ],
        style={
            "border": "2px black solid",
            "padding-left": "10px",
            "padding-right": "10px",
        },
    )

    """
    inversion_params_layout = html.Div(
        [
            dcc.Markdown("##### **Inversion Parameters**"),
            inversion_parameters_dropdown,
            starting_model,
            mesh,
            reference_model,
            regularization,
            upper_lower_bounds,
            detrend,
            ignore_values,
            optimization,
        ],
        style={"border": "2px black solid", "padding-left": "10px", "padding-right": "10px"},
    )
    """

    return inversion_params_layout


inducing_params_div = html.Div(
    id="inducing_params_div",
    children=[
        dcc.Markdown("**Inducing Field Parameters**"),
        html.Div(
            [
                dcc.Markdown(
                    "Amplitude (nT)", style={"display": "inline-block", "width": "40%"}
                ),
                dcc.Input(
                    id="inducing_field_strength",
                    style={"display": "inline-block", "width": "60%"},
                ),
            ]
        ),
        html.Div(
            [
                dcc.Markdown(
                    "Inclination (d.dd)",
                    style={"display": "inline-block", "width": "40%"},
                ),
                dcc.Input(
                    id="inducing_field_inclination",
                    style={"display": "inline-block", "width": "60%"},
                ),
            ]
        ),
        html.Div(
            [
                dcc.Markdown(
                    "Declination (d.dd)",
                    style={"display": "inline-block", "width": "40%"},
                ),
                dcc.Input(
                    id="inducing_field_declination",
                    style={"display": "inline-block", "width": "60%"},
                ),
            ]
        ),
    ],
    style={"display": "inline-block", "vertical-align": "top", "width": "40%"},
)

plot_layout = html.Div(
    [
        html.Div(
            [
                dcc.Markdown(
                    "Grid Resolution (m)",
                    style={"display": "inline-block", "width": "30%"},
                ),
                dcc.Input(
                    id="resolution",
                    style={"display": "inline-block"},
                ),
            ]
        ),
        dcc.Markdown(
            id="data_count",
            children=["Data Count:"],
            style={"display": "inline-block", "width": "30%"},
        ),
        dcc.Checklist(
            id="colorbar",
            options=[{"label": "Colorbar", "value": True}],
            style={"display": "inline-block"},
        ),
        dcc.Checklist(
            id="fix_aspect_ratio",
            options=[{"label": "Fix Aspect Ratio", "value": True}],
            style={"display": "inline-block"},
        ),
        dcc.Graph(
            id="plot",
            style={
                # "width": "70%",
                # "height": "100%",
                "display": "inline-block",
                "vertical-align": "middle",
            },
        ),
    ],
    style={"display": "inline-block", "width": "60%"},
)

input_data_layout = html.Div(
    [
        dcc.Markdown("##### **Input Data**"),
        html.Div(
            [
                html.Div(
                    [
                        dcc.Markdown(
                            "Component:",
                            style={"display": "inline-block", "width": "30%"},
                        ),
                        dcc.Dropdown(
                            id="component",
                            style={"display": "inline-block", "width": "70%"},
                        ),
                        dcc.Store(id="full_components"),
                    ]
                ),
                dcc.Checklist(
                    id="channel_bool", options=[{"label": "Active", "value": True}]
                ),
                html.Div(
                    [
                        dcc.Markdown(
                            "Channel:",
                            style={"display": "inline-block", "width": "30%"},
                        ),
                        dcc.Dropdown(
                            id="channel",
                            style={"display": "inline-block", "width": "70%"},
                        ),
                    ]
                ),
                dcc.Markdown("**Uncertainties**"),
                dcc.RadioItems(
                    id="uncertainty_options",
                    options=[
                        "Floor",
                        "Channel",
                    ],
                    value="Floor",
                    style={
                        "display": "inline-block",
                        "vertical-align": "top",
                        "width": "30%",
                    },
                ),
                html.Div(
                    [
                        dcc.Input(
                            id="uncertainty_floor",
                        ),
                        dcc.Dropdown(
                            id="uncertainty_channel",
                        ),
                    ],
                    style={"display": "inline-block", "width": "70%"},
                ),
            ],
            style={"display": "inline-block", "vertical-align": "top", "width": "40%"},
        ),
        plot_layout,
    ],
    style={
        "border": "2px black solid",
        "padding-left": "10px",
        "padding-right": "10px",
    },
)

topography_none = html.Div(
    id="topography_none_div", children=[dcc.Markdown("No topography")]
)
topography_object = html.Div(
    id="topography_object_div",
    children=[
        html.Div(
            [
                dcc.Markdown(
                    "Object:", style={"display": "inline-block", "width": "25%"}
                ),
                dcc.Dropdown(
                    id="topography_object",
                    style={"display": "inline-block", "width": "75%"},
                ),
            ]
        ),
        html.Div(
            [
                dcc.Markdown(
                    "Data:", style={"display": "inline-block", "width": "25%"}
                ),
                dcc.Dropdown(
                    id="topography_data",
                    style={"display": "inline-block", "width": "75%"},
                ),
            ]
        ),
    ],
)
topography_constant = html.Div(
    id="topography_constant_div",
    children=[
        html.Div(
            [
                dcc.Markdown(
                    "Elevation (m)", style={"display": "inline-block", "width": "25%"}
                ),
                dcc.Input(
                    id="topography_const",
                    style={"display": "inline-block", "width": "60%"},
                ),
            ]
        ),
    ],
)

topography_layout = html.Div(
    [
        dcc.Markdown("##### **Topography**"),
        html.Div(
            [
                html.Div(
                    [
                        dcc.Markdown(
                            "Define by:",
                            style={
                                "display": "inline-block",
                                "vertical-align": "top",
                                "width": "20%",
                            },
                        ),
                        dcc.RadioItems(
                            id="topography_options",
                            options=[
                                "None",
                                "Object",
                                "Constant",
                            ],
                            value="None",
                            style={
                                "display": "inline-block",
                                "vertical-align": "top",
                                "width": "70%",
                            },
                        ),
                    ]
                ),
                topography_none,
                topography_object,
                topography_constant,
            ],
            style={"display": "inline-block", "vertical-align": "top", "width": "50%"},
        ),
        html.Div(
            [
                dcc.Markdown("**Sensor Location**"),
                dcc.Checklist(
                    id="z_from_topo",
                    options=[{"label": "Set Z from topo + offsets", "value": True}],
                ),
                dcc.Markdown("**Offsets**"),
                html.Div(
                    [
                        dcc.Markdown(
                            "dx (+East)",
                            style={"display": "inline-block", "width": "25%"},
                        ),
                        dcc.Input(
                            id="receivers_offset_x",
                            style={"display": "inline-block", "width": "50%"},
                        ),
                    ]
                ),
                html.Div(
                    [
                        dcc.Markdown(
                            "dy (+North)",
                            style={"display": "inline-block", "width": "25%"},
                        ),
                        dcc.Input(
                            id="receivers_offset_y",
                            style={"display": "inline-block", "width": "50%"},
                        ),
                    ]
                ),
                html.Div(
                    [
                        dcc.Markdown(
                            "dz (+ve up)",
                            style={"display": "inline-block", "width": "25%"},
                        ),
                        dcc.Input(
                            id="receivers_offset_z",
                            style={"display": "inline-block", "width": "50%"},
                        ),
                    ]
                ),
                html.Div(
                    [
                        dcc.Markdown(
                            "Radar (Optional):",
                            style={"display": "inline-block", "width": "25%"},
                        ),
                        dcc.Dropdown(
                            id="receivers_radar_drape",
                            style={"display": "inline-block", "width": "50%"},
                        ),
                    ]
                ),
            ],
            style={"display": "inline-block", "vertical-align": "top", "width": "50%"},
        ),
    ],
    style={
        "border": "2px black solid",
        "padding-left": "10px",
        "padding-right": "10px",
    },
)

mesh = html.Div(
    id="mesh_div",
    children=[
        dcc.Markdown("###### **Mesh**"),
        dcc.Markdown("Object:"),
        dcc.Dropdown(id="mesh_object"),
        dcc.Markdown("**Core cell size (u, v, w)**"),
        dcc.Input(id="u_cell_size", style={"width": "40%", "margin-right": "60%"}),
        dcc.Input(id="v_cell_size", style={"width": "40%", "margin-right": "60%"}),
        dcc.Input(id="w_cell_size", style={"width": "40%", "margin-right": "60%"}),
        dcc.Markdown("**Refinement Layers**"),
        html.Div(
            [
                dcc.Markdown(
                    "Number of cells below topography",
                    style={"display": "inline-block", "width": "40%"},
                ),
                dcc.Input(
                    id="octree_levels_topo",
                    style={"display": "inline-block", "width": "40%"},
                ),
            ]
        ),
        html.Div(
            [
                dcc.Markdown(
                    "Number of cells below sensors",
                    style={"display": "inline-block", "width": "40%"},
                ),
                dcc.Input(
                    id="octree_levels_obs",
                    style={"display": "inline-block", "width": "40%"},
                ),
            ]
        ),
        html.Div(
            [
                dcc.Markdown(
                    "Maximum distance (m)",
                    style={"display": "inline-block", "width": "40%"},
                ),
                dcc.Input(
                    id="max_distance",
                    style={"display": "inline-block", "width": "40%"},
                ),
            ]
        ),
        dcc.Markdown("**Dimensions**"),
        html.Div(
            [
                dcc.Markdown(
                    "Horizontal padding (m)",
                    style={"display": "inline-block", "width": "30%"},
                ),
                dcc.Input(
                    id="horizontal_padding",
                    style={"display": "inline-block", "width": "40%"},
                ),
            ]
        ),
        html.Div(
            [
                dcc.Markdown(
                    "Vertical padding (m)",
                    style={"display": "inline-block", "width": "30%"},
                ),
                dcc.Input(
                    id="vertical_padding",
                    style={"display": "inline-block", "width": "40%"},
                ),
            ]
        ),
        html.Div(
            [
                dcc.Markdown(
                    "Minimum depth (m)",
                    style={"display": "inline-block", "width": "30%"},
                ),
                dcc.Input(
                    id="depth_core", style={"display": "inline-block", "width": "40%"}
                ),
            ]
        ),
    ],
    style={"display": "inline-block", "vertical-align": "top", "width": "50%"},
)

regularization = html.Div(
    id="regularization_div",
    children=[
        dcc.Markdown("###### **Regularization**"),
        html.Div(
            [
                html.Div(
                    [
                        dcc.Markdown(
                            "**Scaling (alphas)**",
                            style={
                                "display": "inline-block",
                                "width": "30%",
                                "margin-left": "30%",
                            },
                        ),
                        dcc.Markdown(
                            "**Lp-norms**",
                            style={"display": "inline-block", "width": "30%"},
                        ),
                    ]
                ),
                html.Div(
                    [
                        dcc.Markdown(
                            "Reference Model (s)",
                            style={"display": "inline-block", "width": "30%"},
                        ),
                        dcc.Input(
                            id="alpha_s",
                            style={"display": "inline-block", "width": "30%"},
                        ),
                        dcc.Input(
                            id="s_norm",
                            style={"display": "inline-block", "width": "30%"},
                        ),
                    ]
                ),
                html.Div(
                    [
                        dcc.Markdown(
                            "EW-gradient (x)",
                            style={"display": "inline-block", "width": "30%"},
                        ),
                        dcc.Input(
                            id="alpha_x",
                            style={"display": "inline-block", "width": "30%"},
                        ),
                        dcc.Input(
                            id="x_norm",
                            style={"display": "inline-block", "width": "30%"},
                        ),
                    ]
                ),
                html.Div(
                    [
                        dcc.Markdown(
                            "NS_gradient (y)",
                            style={"display": "inline-block", "width": "30%"},
                        ),
                        dcc.Input(
                            id="alpha_y",
                            style={"display": "inline-block", "width": "30%"},
                        ),
                        dcc.Input(
                            id="y_norm",
                            style={"display": "inline-block", "width": "30%"},
                        ),
                    ]
                ),
                html.Div(
                    [
                        dcc.Markdown(
                            "Vertical-gradient (z)",
                            style={"display": "inline-block", "width": "30%"},
                        ),
                        dcc.Input(
                            id="alpha_z",
                            style={"display": "inline-block", "width": "30%"},
                        ),
                        dcc.Input(
                            id="z_norm",
                            style={"display": "inline-block", "width": "30%"},
                        ),
                    ]
                ),
            ],
        ),
    ],
    style={"display": "inline-block", "vertical-align": "top", "width": "50%"},
)

upper_lower_bounds = html.Div(
    id="upper_lower_bound_div",
    children=[
        dcc.Markdown("###### **Upper-Lower Bounds**"),
        generate_model_component("lower_bound", "Lower Bounds", "Units"),
        generate_model_component("upper_bound", "Upper Bounds", "Units"),
    ],
    style={"display": "inline-block", "vertical-align": "top", "width": "50%"},
)

ignore_values = html.Div(
    id="ignore_values_div",
    children=[
        dcc.Markdown("###### **Ignore Values**"),
        html.Div(
            [
                dcc.Markdown(
                    "Value (i.e. '<0' for no negatives)",
                    style={"display": "inline-block", "width": "55%"},
                ),
                dcc.Input(
                    id="ignore_values",
                    style={"display": "inline-block", "width": "40%"},
                ),
            ]
        ),
    ],
    style={"display": "inline-block", "vertical-align": "top", "width": "50%"},
)

optimization = html.Div(
    id="optimization_div",
    children=[
        dcc.Markdown("###### **Optimization**"),
        html.Div(
            [
                dcc.Markdown(
                    "Max beta Iterations",
                    style={"display": "inline-block", "width": "50%"},
                ),
                dcc.Input(
                    id="max_iterations",
                    style={"display": "inline-block", "width": "50%"},
                ),
            ]
        ),
        html.Div(
            [
                dcc.Markdown(
                    "Target misfit", style={"display": "inline-block", "width": "50%"}
                ),
                dcc.Input(
                    id="chi_factor",
                    style={"display": "inline-block", "width": "50%"},
                ),
            ]
        ),
        html.Div(
            [
                dcc.Markdown(
                    "Beta ratio (phi_d/phi_m)",
                    style={"display": "inline-block", "width": "50%"},
                ),
                dcc.Input(
                    id="initial_beta_ratio",
                    style={"display": "inline-block", "width": "50%"},
                ),
            ]
        ),
        html.Div(
            [
                dcc.Markdown(
                    "Max CG Iterations",
                    style={"display": "inline-block", "width": "50%"},
                ),
                dcc.Input(
                    id="max_cg_iterations",
                    style={"display": "inline-block", "width": "50%"},
                ),
            ]
        ),
        html.Div(
            [
                dcc.Markdown(
                    "CG Tolerance", style={"display": "inline-block", "width": "50%"}
                ),
                dcc.Input(
                    id="tol_cg", style={"display": "inline-block", "width": "50%"}
                ),
            ]
        ),
        html.Div(
            [
                dcc.Markdown(
                    "Max CPUs", style={"display": "inline-block", "width": "50%"}
                ),
                dcc.Input(
                    id="n_cpu", style={"display": "inline-block", "width": "50%"}
                ),
            ]
        ),
        html.Div(
            [
                dcc.Markdown(
                    "Number of tiles", style={"display": "inline-block", "width": "50%"}
                ),
                dcc.Input(
                    id="tile_spatial", style={"display": "inline-block", "width": "50%"}
                ),
            ]
        ),
    ],
    style={"display": "inline-block", "vertical-align": "top", "width": "50%"},
)

detrend = html.Div(
    id="detrend_div",
    children=[
        dcc.Markdown("###### **Detrend**"),
        html.Div(
            [
                html.Div(
                    [
                        dcc.Markdown(
                            "Method", style={"display": "inline-block", "width": "30%"}
                        ),
                        dcc.Dropdown(
                            id="detrend_type",
                            options=["all", "perimeter"],
                            style={"display": "inline-block", "width": "70%"},
                        ),
                    ]
                ),
                html.Div(
                    [
                        dcc.Markdown(
                            "Order", style={"display": "inline-block", "width": "40%"}
                        ),
                        dcc.Input(
                            id="detrend_order",
                            style={"display": "inline-block", "width": "60%"},
                        ),
                    ]
                ),
            ],
        ),
    ],
    style={"display": "inline-block", "vertical-align": "top", "width": "60%"},
)

inversion_parameters_dropdown = html.Div(
    [
        dcc.Checklist(
            id="forward_only", options=[{"label": "Forward only", "value": True}]
        ),
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
    ],
    style={
        "display": "inline-block",
        "vertical-align": "top",
        "width": "30%",
        "margin-right": "10%",
    },
)

output_layout = html.Div(
    [
        dcc.Markdown("##### **Output**"),
        html.Div(
            [
                dcc.Markdown(
                    "Save as:", style={"display": "inline-block", "width": "10%"}
                ),
                dcc.Input(
                    id="out_group",
                    style={"display": "inline-block", "width": "10%"},
                ),
            ]
        ),
        html.Div(
            [
                dcc.Markdown(
                    "Output path:", style={"display": "inline-block", "width": "15%"}
                ),
                dcc.Input(
                    id="monitoring_directory",
                    style={"display": "inline-block", "width": "50%"},
                ),
            ]
        ),
        html.Button(
            "Write input",
            id="write_input",
            style={"width": "15%", "margin-right": "85%"},
        ),
        html.Button(
            "Compute", id="compute", style={"width": "15%", "margin-right": "85%"}
        ),
    ],
    style={
        "border": "2px black solid",
        "padding-left": "10px",
        "padding-right": "10px",
    },
)
