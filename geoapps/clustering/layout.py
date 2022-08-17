#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import dash_daq as daq
from dash import dash_table, dcc, html

from geoapps.scatter_plot.layout import axis_layout, plot_layout, workspace_layout

# Layout for histogram, stats table, confusion matrix
norm_tabs_layout = html.Div(
    id="norm_tabs",
    children=[
        dcc.Tabs(
            [
                dcc.Tab(
                    label="Histogram",
                    children=[
                        html.Div(
                            [
                                html.Div(
                                    [
                                        dcc.Markdown("Data: "),
                                        dcc.Dropdown(
                                            id="channel",
                                        ),
                                        dcc.Markdown("Scale: "),
                                        dcc.Slider(
                                            id="scale",
                                            min=1,
                                            max=10,
                                            step=1,
                                            marks=None,
                                            tooltip={
                                                "placement": "bottom",
                                                "always_visible": True,
                                            },
                                        ),
                                        dcc.Markdown("Lower bound: "),
                                        dcc.Input(
                                            id="lower_bounds",
                                        ),
                                        dcc.Markdown("Upper bound: "),
                                        dcc.Input(
                                            id="upper_bounds",
                                        ),
                                    ],
                                    style={
                                        "width": "200px",
                                        "display": "inline-block",
                                        "vertical-align": "middle",
                                        "margin-right": "50px",
                                    },
                                ),
                                dcc.Graph(
                                    id="histogram",
                                    style={
                                        "width": "70%",
                                        "display": "inline-block",
                                        "vertical-align": "middle",
                                    },
                                ),
                            ]
                        )
                    ],
                ),
                dcc.Tab(
                    label="Boxplot",
                    children=[
                        html.Div(
                            [
                                dcc.Graph(
                                    id="boxplot",
                                )
                            ],
                            style={"width": "50%", "margin": "auto"},
                        )
                    ],
                ),
                dcc.Tab(
                    label="Statistics",
                    children=[
                        html.Div(
                            [
                                dash_table.DataTable(
                                    id="stats_table",
                                    style_data={
                                        "color": "black",
                                        "backgroundColor": "white",
                                    },
                                    style_data_conditional=[
                                        {
                                            "if": {"row_index": "odd"},
                                            "backgroundColor": "rgb(220, 220, 220)",
                                        }
                                    ],
                                    style_header={
                                        "backgroundColor": "rgb(210, 210, 210)",
                                        "color": "black",
                                        "fontWeight": "bold",
                                    },
                                )
                            ],
                            style={
                                "margin-top": "20px",
                                "margin-bottom": "20px",
                            },
                        )
                    ],
                ),
                dcc.Tab(
                    label="Confusion Matrix",
                    children=[
                        html.Div(
                            [
                                dcc.Graph(
                                    id="matrix",
                                ),
                            ],
                            style={"width": "50%", "margin": "auto"},
                        )
                    ],
                ),
            ]
        )
    ],
)

# Layout for crossplot, boxplot, inertia plot
cluster_tabs_layout = html.Div(
    [
        dcc.Tabs(
            [
                dcc.Tab(
                    label="Crossplot",
                    children=[
                        html.Div(
                            [axis_layout],
                            style={
                                "width": "45%",
                                "display": "inline-block",
                                "vertical-align": "middle",
                            },
                        ),
                        html.Div(
                            [
                                plot_layout,
                            ],
                            style={
                                "width": "55%",
                                "display": "inline-block",
                                "vertical-align": "middle",
                            },
                        ),
                    ],
                ),
                dcc.Tab(
                    label="Inertia",
                    children=[
                        html.Div(
                            [
                                dcc.Graph(
                                    id="inertia",
                                )
                            ],
                            style={"width": "50%", "margin": "auto"},
                        )
                    ],
                ),
            ]
        )
    ]
)

# Full app layout
cluster_layout = html.Div(
    [
        html.Div(
            [
                # Workspace, object, downsampling, data subset selection
                workspace_layout,
                dcc.Markdown("Data subset: "),
                dcc.Dropdown(
                    id="data_subset",
                    multi=True,
                ),
            ],
            style={
                "width": "40%",
                "display": "inline-block",
                "vertical-align": "top",
                "margin-right": "50px",
                "margin-bottom": "45px",
            },
        ),
        html.Div(
            [
                # Dash components for clustering parameters
                dcc.Markdown("Number of clusters: "),
                dcc.Slider(
                    id="n_clusters",
                    min=2,
                    max=100,
                    step=1,
                    marks=None,
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True,
                    },
                ),
                dcc.Checklist(
                    id="show_color_picker",
                    options=["Select cluster color"],
                    value=[],
                    style={"margin-bottom": "20px"},
                ),
                dcc.Checklist(
                    id="live_link",
                    options=[
                        {"label": "Geoscience ANALYST Pro - Live link", "value": True}
                    ],
                    value=[],
                    style={"margin-bottom": "20px"},
                ),
                dcc.Markdown("Group name:"),
                dcc.Input(
                    id="ga_group_name",
                    style={"margin-bottom": "20px"},
                ),
                dcc.Markdown("Output path:"),
                dcc.Input(
                    id="monitoring_directory",
                    style={"margin-bottom": "20px"},
                ),
                html.Button("Export", id="export"),
                dcc.Markdown(id="export_message"),
            ],
            style={
                "width": "25%",
                "display": "inline-block",
                "vertical-align": "top",
            },
        ),
        html.Div(
            id="color_select_div",
            children=[
                dcc.Markdown("Cluster: "),
                dcc.Dropdown(
                    id="select_cluster",
                    style={"margin-bottom": "20px"},
                    clearable=False,
                ),
                daq.ColorPicker(  # pylint: disable=E1102
                    id="color_picker",
                    value=dict(hex="#000000"),
                ),
            ],
            style={
                "width": "25%",
                "display": "none",
                "vertical-align": "top",
            },
        ),
        # Checkbox to hide/show the normalization plots
        dcc.Checklist(
            id="show_norm_tabs",
            options=["Show Analytics & Normalization"],
            value=[],
            style={"margin-bottom": "10px"},
        ),
        norm_tabs_layout,
        cluster_tabs_layout,
        # Creating stored variables that can be passed through callbacks.
        dcc.Store(id="dataframe"),
        dcc.Store(id="full_scales", data={}),
        dcc.Store(id="full_lower_bounds", data={}),
        dcc.Store(id="full_upper_bounds", data={}),
        dcc.Store(id="color_pickers"),
        dcc.Store(id="kmeans"),
        dcc.Store(id="clusters", data={}),
        dcc.Store(id="indices"),
        dcc.Store(id="mapping"),
        dcc.Store(id="ui_json_data"),
    ],
    style={"width": "70%", "margin-left": "50px", "margin-top": "30px"},
)
