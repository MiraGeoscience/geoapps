# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024-2025 Mira Geoscience Ltd.                                '
#                                                                              '
#  This file is part of geoapps.                                               '
#                                                                              '
#  geoapps is distributed under the terms and conditions of the MIT License    '
#  (see LICENSE file at the root of this source code package).                 '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from __future__ import annotations

from dash import dcc, html


workspace_layout = html.Div(
    [
        dcc.Upload(
            id="upload",
            children=html.Button("Upload Workspace/ui.json"),
            style={"margin_bottom": "40px"},
        ),
        html.Div(
            [
                dcc.Markdown(
                    children="Object:",
                    style={
                        "width": "20%",
                        "display": "inline-block",
                        "margin-top": "20px",
                        "vertical-align": "bottom",
                    },
                ),
                dcc.Dropdown(
                    id="objects",
                    style={
                        "width": "65%",
                        "display": "inline-block",
                        "margin_bottom": "40px",
                        "vertical-align": "bottom",
                    },
                ),
            ]
        ),
    ]
)

launch_qt_layout = html.Div(
    [
        html.Div(
            [
                html.Button("Launch App", id="launch_app", n_clicks=0),
                dcc.Markdown(
                    children="",
                    id="launch_app_markdown",
                    style={"width": "50%", "display": "inline-block"},
                ),
            ],
            style={"margin_top": "40px"},
        ),
        dcc.Store(id="ui_json_data"),
    ]
)

object_selection_layout = html.Div([workspace_layout, launch_qt_layout])
