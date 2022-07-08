#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import base64
import io
import os
import sys

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import callback_context, dcc, html, no_update
from dash.dependencies import Input, Output
from flask import Flask
from geoh5py.objects.object_base import ObjectBase
from geoh5py.ui_json import InputFile
from geoh5py.workspace import Workspace
from jupyter_dash import JupyterDash

from geoapps.base.dash_application import BaseDashApplication
from geoapps.grid_creation.constants import app_initializer
from geoapps.grid_creation.driver import GridCreationDriver
from geoapps.grid_creation.params import GridCreationParams


class GridCreation(BaseDashApplication):
    _param_class = GridCreationParams

    def __init__(self, ui_json=None, **kwargs):
        app_initializer.update(kwargs)
        if ui_json is not None and os.path.exists(ui_json.path):
            self.params = self._param_class(ui_json)
        else:
            self.params = self._param_class(**app_initializer)

        super().__init__()
        # Initial values for the dash components
        defaults = self.get_defaults()

        # Set up the layout with the dash components
        self.app.layout = html.Div(
            [
                dcc.Upload(
                    id="upload",
                    children=html.Button("Upload Workspace/ui.json"),
                    style={"margin_bottom": "20px"},
                ),
                dcc.Markdown(
                    children="Input object",
                    style={"width": "25%", "display": "inline-block"},
                ),
                dcc.Dropdown(
                    id="objects",
                    value=defaults["objects_name"],
                    options=defaults["objects_options"],
                    style={
                        "width": "75%",
                        "display": "inline-block",
                        "margin_bottom": "20px",
                    },
                ),
                dcc.Markdown(
                    children="Lateral extent",
                    style={"width": "25%", "display": "inline-block"},
                ),
                dcc.Dropdown(
                    id="xy_reference",
                    value=defaults["xy_reference_name"],
                    options=defaults["xy_reference_options"],
                    style={
                        "width": "75%",
                        "display": "inline-block",
                        "margin_bottom": "20px",
                    },
                ),
                dcc.Markdown(
                    children="Name",
                    style={"width": "25%", "display": "inline-block"},
                ),
                dcc.Input(
                    id="new_grid",
                    value=defaults["new_grid"],
                    style={
                        "width": "70%",
                        "display": "inline-block",
                        "margin_bottom": "20px",
                    },
                ),
                dcc.Markdown(
                    children="Smallest cells",
                    style={"width": "25%", "display": "inline-block"},
                ),
                dcc.Input(
                    id="core_cell_size",
                    value=defaults["core_cell_size"],
                    style={
                        "width": "70%",
                        "display": "inline-block",
                        "margin_bottom": "20px",
                    },
                ),
                dcc.Markdown(
                    children="Core depth (m)",
                    style={"width": "25%", "display": "inline-block"},
                ),
                dcc.Input(
                    id="depth_core",
                    type="number",
                    value=defaults["depth_core"],
                    style={
                        "width": "70%",
                        "display": "inline-block",
                        "margin_bottom": "20px",
                    },
                ),
                dcc.Markdown(
                    children="Pad distance (W, E, S, N, D, U)",
                    style={"width": "25%", "display": "inline-block"},
                ),
                dcc.Input(
                    id="padding_distance",
                    value=defaults["padding_distance"],
                    style={
                        "width": "70%",
                        "display": "inline-block",
                        "margin_bottom": "20px",
                    },
                ),
                dcc.Markdown(
                    children="Expansion factor",
                    style={"width": "25%", "display": "inline-block"},
                ),
                dcc.Input(
                    id="expansion_fact",
                    type="number",
                    value=defaults["expansion_fact"],
                    style={
                        "width": "70%",
                        "display": "inline-block",
                        "margin_bottom": "20px",
                    },
                ),
                html.Button("Export", id="export"),
                dcc.Markdown(id="output_message"),
                # dcc.Store(
                #    id="param_dict",
                #    data=defaults["param_dict"]
                # ),
            ],
            style={
                "margin_left": "20px",
                "margin_top": "20px",
                "width": "40%",
            },
        )

        # Set up callbacks
        self.app.callback(
            # Output(component_id="param_dict", component_property="data"),
            Output(component_id="new_grid", component_property="value"),
            Output(component_id="objects", component_property="value"),
            Output(component_id="objects", component_property="options"),
            Output(component_id="xy_reference", component_property="value"),
            Output(component_id="xy_reference", component_property="options"),
            Output(component_id="core_cell_size", component_property="value"),
            Output(component_id="depth_core", component_property="value"),
            Output(component_id="padding_distance", component_property="value"),
            Output(component_id="expansion_fact", component_property="value"),
            Output(component_id="upload", component_property="filename"),
            Output(component_id="upload", component_property="contents"),
            # Input(component_id="param_dict", component_property="data"),
            Input(component_id="upload", component_property="filename"),
            Input(component_id="upload", component_property="contents"),
            Input(component_id="new_grid", component_property="value"),
            Input(component_id="objects", component_property="value"),
            Input(component_id="xy_reference", component_property="value"),
            Input(component_id="core_cell_size", component_property="value"),
            Input(component_id="depth_core", component_property="value"),
            Input(component_id="padding_distance", component_property="value"),
            Input(component_id="expansion_fact", component_property="value"),
        )(self.update_params)
        self.app.callback(
            Output(component_id="output_message", component_property="children"),
            Input(component_id="export", component_property="n_clicks"),
            # Input(component_id="param_dict", component_property="data"),
        )(self.create_block_model)

    def update_params(
        self,
        filename,
        contents,
        new_grid,
        objects,
        xy_reference,
        core_cell_size,
        depth_core,
        padding_distance,
        expansion_fact,
    ):
        param_list = [
            "new_grid",
            "objects_name",
            "objects_options",
            "xy_reference_name",
            "xy_reference_options",
            "core_cell_size",
            "depth_core",
            "padding_distance",
            "expansion_fact",
            "filename",
            "contents",
        ]

        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]
        update_dict = {}
        if trigger == "upload":
            if (filename.endswith("ui.json") or filename.endswith(".geoh5")) and (
                contents is not None
            ):
                if filename.endswith(".ui.json"):
                    update_dict.update(self.update_from_ui_json(contents, param_list))
                    ws = update_dict["geoh5"]
                elif filename.endswith(".geoh5"):
                    content_type, content_string = contents.split(",")
                    decoded = io.BytesIO(base64.b64decode(content_string))
                    ws = Workspace(decoded)
                update_dict.update(self.update_object_options(ws, "objects"))
                update_dict.update(self.update_object_options(ws, "xy_reference"))
                update_dict["filename"] = None
                update_dict["contents"] = None
            else:
                print("Uploaded file must be a workspace or ui.json.")
        elif trigger + "_name" in param_list:
            update_dict[trigger + "_name"] = locals()[trigger]
        elif trigger != "":
            update_dict[trigger] = locals()[trigger]

        self.update_param_dict(update_dict)
        outputs = self.get_outputs(param_list, update_dict)

        return outputs

    def create_block_model(self, n_clicks):

        if callback_context.triggered[0]["prop_id"].split(".")[0] == "export":
            ifile = InputFile(
                ui_json=self.params.input_file.ui_json,
                validation_options={"disabled": True},
            )

            ifile.data = self.params.to_dict()
            new_params = GridCreationParams(input_file=ifile)
            new_params.write_input_file()

            driver = GridCreationDriver(new_params)
            driver.run()


grid_creation = GridCreation()
grid_creation.run()

if __name__ == "__main__":
    print("Loading geoh5 file . . .")
    file = sys.argv[1]
    ifile = InputFile.read_ui_json(file)
    app = GridCreation(ui_json=ifile)
    print("Loaded. Creating block model . . .")
    app.run()
    print("Done")
