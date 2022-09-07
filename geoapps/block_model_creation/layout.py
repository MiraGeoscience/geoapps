#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


from dash import dcc, html

block_model_layout = html.Div(
    [
        dcc.Upload(
            id="upload",
            children=html.Button("Upload Workspace/ui.json"),
            style={"margin_bottom": "20px"},
        ),
        html.Div(
            [
                dcc.Markdown(
                    children="Object:",
                    style={
                        "width": "25%",
                        "display": "inline-block",
                        "margin-top": "20px",
                    },
                ),
                dcc.Dropdown(
                    id="objects",
                    style={
                        "width": "75%",
                        "display": "inline-block",
                        "margin_bottom": "20px",
                    },
                ),
            ]
        ),
        html.Div(
            [
                dcc.Markdown(
                    children="Name:",
                    style={"width": "25%", "display": "inline-block"},
                ),
                dcc.Input(
                    id="new_grid",
                    style={
                        "width": "50%",
                        "display": "inline-block",
                        "margin_bottom": "20px",
                    },
                ),
            ]
        ),
        html.Div(
            [
                dcc.Markdown(
                    children="Minimum x cell size:",
                    style={"width": "25%", "display": "inline-block"},
                ),
                dcc.Input(
                    id="cell_size_x",
                    type="number",
                    min=1e-14,
                    style={
                        "width": "50%",
                        "display": "inline-block",
                        "margin_bottom": "20px",
                    },
                ),
            ]
        ),
        html.Div(
            [
                dcc.Markdown(
                    children="Minimum y cell size:",
                    style={"width": "25%", "display": "inline-block"},
                ),
                dcc.Input(
                    id="cell_size_y",
                    type="number",
                    min=1e-14,
                    style={
                        "width": "50%",
                        "display": "inline-block",
                        "margin_bottom": "20px",
                    },
                ),
            ]
        ),
        html.Div(
            [
                dcc.Markdown(
                    children="Minimum z cell size:",
                    style={"width": "25%", "display": "inline-block"},
                ),
                dcc.Input(
                    id="cell_size_z",
                    type="number",
                    min=1e-14,
                    style={
                        "width": "50%",
                        "display": "inline-block",
                        "margin_bottom": "20px",
                    },
                ),
            ]
        ),
        html.Div(
            [
                dcc.Markdown(
                    children="Core depth (m):",
                    style={"width": "25%", "display": "inline-block"},
                ),
                dcc.Input(
                    id="depth_core",
                    type="number",
                    min=0.0,
                    style={
                        "width": "50%",
                        "display": "inline-block",
                        "margin_bottom": "20px",
                    },
                ),
            ]
        ),
        html.Div(
            [
                dcc.Markdown(
                    children="Horizontal padding:",
                    style={"width": "25%", "display": "inline-block"},
                ),
                dcc.Input(
                    id="horizontal_padding",
                    type="number",
                    min=0.0,
                    style={
                        "width": "50%",
                        "display": "inline-block",
                        "margin_bottom": "20px",
                    },
                ),
            ]
        ),
        html.Div(
            [
                dcc.Markdown(
                    children="Bottom padding:",
                    style={"width": "25%", "display": "inline-block"},
                ),
                dcc.Input(
                    id="bottom_padding",
                    type="number",
                    min=0.0,
                    style={
                        "width": "50%",
                        "display": "inline-block",
                        "margin_bottom": "20px",
                    },
                ),
            ]
        ),
        html.Div(
            [
                dcc.Markdown(
                    children="Expansion factor:",
                    style={"width": "25%", "display": "inline-block"},
                ),
                dcc.Input(
                    id="expansion_fact",
                    type="number",
                    style={
                        "width": "50%",
                        "display": "inline-block",
                        "margin_bottom": "20px",
                    },
                ),
            ]
        ),
        html.Div(
            [
                dcc.Markdown(
                    children="Output path:",
                    style={"width": "25%", "display": "inline-block"},
                ),
                dcc.Input(
                    id="monitoring_directory",
                    style={
                        "width": "50%",
                        "display": "inline-block",
                        "margin_bottom": "20px",
                    },
                ),
            ]
        ),
        dcc.Checklist(
            id="live_link",
            options=[{"label": "Geoscience ANALYST Pro - Live link", "value": True}],
            value=[],
            style={"margin_bottom": "20px"},
        ),
        html.Button("Export", id="export"),
        dcc.Markdown(id="output_message"),
        dcc.Store(id="ui_json_data"),
    ],
    style={
        "margin_left": "20px",
        "margin_top": "20px",
        "width": "75%",
    },
)
