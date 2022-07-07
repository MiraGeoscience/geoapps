#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

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

        # Initial values for the dash components
        defaults = self.get_defaults()
        super().__init__()

        # Set up the layout with the dash components
        self.app.layout = html.Div(
            [
                dcc.Upload(
                    id="upload",
                    children=html.Button("Upload Workspace/ui.json"),
                ),
                dcc.Markdown("Name"),
                dcc.Input(id="new_grid"),
                dcc.Markdown("Lateral Extent"),
                dcc.Dropdown(id="objects"),
                dcc.Markdown("Smallest cells"),
                dcc.Input(id="core_cell_size"),
                dcc.Markdown("Core depth (m)"),
                dcc.Input(id="depth_core", type="number"),
                dcc.Markdown("Pad distance (W, E, S, N, D, U)"),
                dcc.Input(id="padding_distance"),
                dcc.Markdown("Expansion factor"),
                dcc.Input(id="expansion_fact", type="number"),
                html.Button("Export", id="export"),
                dcc.Markdown(id="output_message"),
                dcc.Store(id="param_dict"),
            ]
        )

        # Set up callbacks
        self.app.callback(
            Output(component_id="param_dict", component_property="data"),
            Output(component_id="new_grid", component_property="value"),
            Output(component_id="objects", component_property="value"),
            Output(component_id="objects", component_property="options"),
            Output(component_id="core_cell_size", component_property="value"),
            Output(component_id="depth_core", component_property="value"),
            Output(component_id="padding_distance", component_property="value"),
            Output(component_id="expansion_fact", component_property="value"),
            Output(component_id="upload", component_property="filename"),
            Output(component_id="upload", component_property="contents"),
            Input(component_id="param_dict", component_property="data"),
            Input(component_id="upload", component_property="value"),
            Input(component_id="new_grid", component_property="value"),
            Input(component_id="objects", component_property="value"),
            Input(component_id="core_cell_size", component_property="value"),
            Input(component_id="depth_core", component_property="value"),
            Input(component_id="padding_distance", component_property="value"),
            Input(component_id="expansion_fact", component_property="value"),
        )(self.update_params)
        self.app.callback(
            Output(component_id="output_message", component_property="children"),
            Input(component_id="export", component_property="n_clicks"),
            Input(component_id="param_dict", component_property="data"),
        )(self.create_block_model)

    def update_params(
        self,
        param_dict,
        upload,
        new_grid,
        objects,
        core_cell_size,
        depth_core,
        padding_distance,
        expansion_fact,
    ):
        param_list = [
            "new_grid",
            "objects_name",
            "objects_options",
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
            if filename.endswith(".ui.json"):
                update_dict.update(self.update_from_uijson(contents))
            elif filename.endswith(".geoh5"):
                update_dict.update(self.update_object_options(contents))
            else:
                print("Uploaded file must be a workspace or ui.json.")
            update_dict["filename"] = None
            update_dict["contents"] = None
        elif trigger + "_name" in param_list:
            update_dict[trigger + "_name"] = getattr(locals(), trigger)
        else:
            update_dict[trigger] = getattr(locals(), trigger)

        new_param_dict = self.update_param_dict(param_dict, update_dict)
        outputs = self.get_outputs(param_list, update_dict)

        return new_param_dict, outputs

    def create_block_model(self, n_clicks, param_dict):

        if callback_context.triggered[0]["prop_id"].split(".")[0] == "export":
            ifile = InputFile(
                ui_json=self.params.input_file.ui_json,
                validation_options={"disabled": True},
            )

            # Write uijson ***

            ifile.data = param_dict
            new_params = InterpGridParams(input_file=ifile)

            driver = InterpGridDriver(new_params)
            driver.run()


if __name__ == "__main__":
    print("Loading geoh5 file . . .")
    file = sys.argv[1]
    ifile = InputFile.read_ui_json(file)
    app = InterpGrid(ui_json=ifile)
    print("Loaded. Creating block model . . .")
    app.run()
    print("Done")
