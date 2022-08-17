#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import plotly.express as px
from dash import dcc, html

workspace_layout = html.Div(
    [
        dcc.Upload(
            id="upload",
            children=html.Button("Upload Workspace/ui.json"),
            style={"margin-bottom": "20px"},
        ),
        dcc.Markdown(children="Object: "),
        dcc.Dropdown(
            id="objects",
            style={"margin-bottom": "20px"},
            clearable=False,
        ),
        html.Div(
            [
                dcc.Markdown(
                    children="Population Downsampling (%): ",
                    style={
                        "display": "inline-block",
                        "margin-right": "5px",
                    },
                ),
                dcc.Slider(
                    id="downsampling",
                    min=1,
                    max=100,
                    step=1,
                    marks=None,
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True,
                    },
                ),
            ],
            style={"margin-bottom": "20px"},
        ),
    ]
)
axis_layout = html.Div(
    [
        html.Div(
            [
                dcc.Markdown(children="Axis: "),
                dcc.Dropdown(
                    id="axes_panels",
                    options=[
                        {"label": "X-axis", "value": "x"},
                        {"label": "Y-axis", "value": "y"},
                        {"label": "Z-axis", "value": "z"},
                        {"label": "Color", "value": "color"},
                        {"label": "Size", "value": "size"},
                    ],
                    value="x",
                    style={"margin-bottom": "20px"},
                    clearable=False,
                ),
            ],
            style={
                "display": "block",
                "vertical-align": "top",
            },
        ),
        html.Div(
            id="x_div",
            children=[
                dcc.Markdown(children="Data: "),
                dcc.Dropdown(
                    id="x",
                    style={"margin-bottom": "20px"},
                ),
                html.Div(
                    [
                        dcc.Markdown(
                            children="Threshold: ",
                            style={
                                "display": "inline-block",
                                "margin-right": "5px",
                            },
                        ),
                        dcc.Input(
                            id="x_thresh",
                            type="number",
                            style={
                                "display": "inline-block",
                                "margin-right": "20px",
                            },
                        ),
                        dcc.Checklist(
                            id="x_log",
                            options=[{"label": "Log10", "value": True}],
                            style={"display": "inline-block"},
                        ),
                    ],
                    style={"margin-bottom": "20px"},
                ),
                html.Div(
                    [
                        dcc.Markdown(
                            children="Min: ",
                            style={
                                "display": "inline-block",
                                "margin-right": "5px",
                            },
                        ),
                        dcc.Input(
                            id="x_min",
                            type="number",
                            style={
                                "display": "inline-block",
                                "margin-right": "20px",
                            },
                        ),
                    ],
                    style={"margin-bottom": "20px"},
                ),
                html.Div(
                    [
                        dcc.Markdown(
                            children="Max: ",
                            style={
                                "display": "inline-block",
                                "margin-right": "5px",
                            },
                        ),
                        dcc.Input(
                            id="x_max",
                            type="number",
                            style={"display": "inline-block"},
                        ),
                    ],
                ),
            ],
            style={
                "display": "block",
                "width": "40%",
                "vertical-align": "top",
                "margin-bottom": "20px",
            },
        ),
        html.Div(
            id="y_div",
            children=[
                dcc.Markdown(children="Data: "),
                dcc.Dropdown(
                    id="y",
                    style={"margin-bottom": "20px"},
                ),
                html.Div(
                    [
                        dcc.Markdown(
                            children="Threshold: ",
                            style={
                                "display": "inline-block",
                                "margin-right": "5px",
                            },
                        ),
                        dcc.Input(
                            id="y_thresh",
                            type="number",
                            style={
                                "display": "inline-block",
                                "margin-right": "20px",
                            },
                        ),
                        dcc.Checklist(
                            id="y_log",
                            options=[{"label": "Log10", "value": True}],
                            style={"display": "inline-block"},
                        ),
                    ],
                    style={"margin-bottom": "20px"},
                ),
                html.Div(
                    [
                        dcc.Markdown(
                            children="Min: ",
                            style={
                                "display": "inline-block",
                                "margin-right": "5px",
                            },
                        ),
                        dcc.Input(
                            id="y_min",
                            type="number",
                            style={
                                "display": "inline-block",
                                "margin-right": "20px",
                            },
                        ),
                    ],
                    style={"margin-bottom": "20px"},
                ),
                html.Div(
                    [
                        dcc.Markdown(
                            children="Max: ",
                            style={
                                "display": "inline-block",
                                "margin-right": "5px",
                            },
                        ),
                        dcc.Input(
                            id="y_max",
                            type="number",
                            style={"display": "inline-block"},
                        ),
                    ],
                ),
            ],
            style={
                "display": "none",
                "width": "40%",
                "vertical-align": "top",
                "margin-bottom": "20px",
            },
        ),
        html.Div(
            id="z_div",
            children=[
                dcc.Markdown(children="Data: "),
                dcc.Dropdown(
                    id="z",
                    style={"margin-bottom": "20px"},
                ),
                html.Div(
                    [
                        dcc.Markdown(
                            children="Threshold: ",
                            style={
                                "display": "inline-block",
                                "margin-right": "5px",
                            },
                        ),
                        dcc.Input(
                            id="z_thresh",
                            type="number",
                            style={
                                "display": "inline-block",
                                "margin-right": "20px",
                            },
                        ),
                        dcc.Checklist(
                            id="z_log",
                            options=[{"label": "Log10", "value": True}],
                            style={"display": "inline-block"},
                        ),
                    ],
                    style={"margin-bottom": "20px"},
                ),
                html.Div(
                    [
                        dcc.Markdown(
                            children="Min: ",
                            style={
                                "display": "inline-block",
                                "margin-right": "5px",
                            },
                        ),
                        dcc.Input(
                            id="z_min",
                            type="number",
                            style={
                                "display": "inline-block",
                                "margin-right": "20px",
                            },
                        ),
                    ],
                    style={"margin-bottom": "20px"},
                ),
                html.Div(
                    [
                        dcc.Markdown(
                            children="Max: ",
                            style={
                                "display": "inline-block",
                                "margin-right": "5px",
                            },
                        ),
                        dcc.Input(
                            id="z_max",
                            type="number",
                            style={"display": "inline-block"},
                        ),
                    ],
                ),
            ],
            style={
                "display": "none",
                "width": "40%",
                "vertical-align": "top",
                "margin-bottom": "20px",
            },
        ),
        html.Div(
            id="color_div",
            children=[
                dcc.Markdown(children="Data: "),
                dcc.Dropdown(
                    id="color",
                    style={"margin-bottom": "20px"},
                ),
                dcc.Markdown(children="Colormap: "),
                dcc.Dropdown(
                    id="color_maps",
                    style={"margin-bottom": "20px"},
                    options=px.colors.named_colorscales(),
                ),
                html.Div(
                    [
                        dcc.Markdown(
                            children="Threshold: ",
                            style={
                                "display": "inline-block",
                                "margin-right": "5px",
                            },
                        ),
                        dcc.Input(
                            id="color_thresh",
                            type="number",
                            style={
                                "display": "inline-block",
                                "margin-right": "20px",
                            },
                        ),
                        dcc.Checklist(
                            id="color_log",
                            options=[{"label": "Log10", "value": True}],
                            style={"display": "inline-block"},
                        ),
                    ],
                    style={"margin-bottom": "20px"},
                ),
                html.Div(
                    [
                        dcc.Markdown(
                            children="Min: ",
                            style={
                                "display": "inline-block",
                                "margin-right": "5px",
                            },
                        ),
                        dcc.Input(
                            id="color_min",
                            type="number",
                            style={
                                "display": "inline-block",
                                "margin-right": "20px",
                            },
                        ),
                    ],
                    style={"margin-bottom": "20px"},
                ),
                html.Div(
                    [
                        dcc.Markdown(
                            children="Max: ",
                            style={
                                "display": "inline-block",
                                "margin-right": "5px",
                            },
                        ),
                        dcc.Input(
                            id="color_max",
                            type="number",
                            style={"display": "inline-block"},
                        ),
                    ],
                ),
            ],
            style={
                "display": "none",
                "width": "40%",
                "vertical-align": "top",
                "margin-bottom": "20px",
            },
        ),
        html.Div(
            id="size_div",
            children=[
                dcc.Markdown(children="Data: "),
                dcc.Dropdown(
                    id="size",
                    style={"margin-bottom": "20px"},
                ),
                html.Div(
                    [
                        dcc.Markdown(
                            children="Marker Size: ",
                            style={
                                "display": "inline-block",
                                "margin-right": "5px",
                            },
                        ),
                        dcc.Slider(
                            id="size_markers",
                            min=1,
                            max=100,
                            step=1,
                            marks=None,
                            tooltip={
                                "placement": "bottom",
                                "always_visible": True,
                            },
                        ),
                    ],
                    style={"width": "80%", "margin-bottom": "20px"},
                ),
                html.Div(
                    [
                        dcc.Markdown(
                            children="Threshold: ",
                            style={
                                "display": "inline-block",
                                "margin-right": "5px",
                            },
                        ),
                        dcc.Input(
                            id="size_thresh",
                            type="number",
                            style={
                                "display": "inline-block",
                                "margin-right": "20px",
                            },
                        ),
                        dcc.Checklist(
                            id="size_log",
                            options=[{"label": "Log10", "value": True}],
                            style={"display": "inline-block"},
                        ),
                    ],
                    style={"margin-bottom": "20px"},
                ),
                html.Div(
                    [
                        dcc.Markdown(
                            children="Min: ",
                            style={
                                "display": "inline-block",
                                "margin-right": "5px",
                            },
                        ),
                        dcc.Input(
                            id="size_min",
                            type="number",
                            style={
                                "display": "inline-block",
                                "margin-right": "20px",
                            },
                        ),
                    ],
                    style={"margin-bottom": "20px"},
                ),
                html.Div(
                    [
                        dcc.Markdown(
                            children="Max: ",
                            style={
                                "display": "inline-block",
                                "margin-right": "5px",
                            },
                        ),
                        dcc.Input(
                            id="size_max",
                            type="number",
                            style={"display": "inline-block"},
                        ),
                    ],
                ),
            ],
        ),
    ]
)
plot_layout = html.Div(
    [
        dcc.Graph(
            id="crossplot",
            style={"margin-bottom": "20px"},
        ),
    ]
)
scatter_layout = html.Div(
    [
        html.Div(
            [workspace_layout, axis_layout],
            style={
                "width": "40%",
                "display": "inline-block",
                "margin-bottom": "20px",
                "vertical-align": "bottom",
                "margin-right": "20px",
            },
        ),
        html.Div(
            [
                plot_layout,
                dcc.Markdown("Output path: "),
                html.Div(
                    [
                        dcc.Input(id="monitoring_directory"),
                        html.Button("Download as html", id="export"),
                    ]
                ),
            ],
            style={
                "width": "55%",
                "display": "inline-block",
                "margin-bottom": "20px",
                "vertical-align": "bottom",
            },
        ),
        dcc.Store(id="ui_json_data"),
    ],
    style={"width": "70%", "margin-left": "50px", "margin-top": "30px"},
)
