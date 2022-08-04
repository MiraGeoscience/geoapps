#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import os
from time import time

from dash import callback_context, no_update
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
from flask import Flask
from geoh5py.objects.object_base import ObjectBase
from jupyter_dash import JupyterDash

from geoapps.base.application import BaseApplication
from geoapps.base.dash_application import BaseDashApplication
from geoapps.block_model_creation.constants import app_initializer
from geoapps.block_model_creation.driver import BlockModelDriver
from geoapps.block_model_creation.layout import block_model_layout
from geoapps.block_model_creation.params import BlockModelParams


class BlockModelCreation(BaseDashApplication):
    """
    Dash app used for the creation of a BlockModel.
    """

    _param_class = BlockModelParams

    def __init__(self, ui_json=None, **kwargs):
        app_initializer.update(kwargs)
        if ui_json is not None and os.path.exists(ui_json.path):
            self.params = self._param_class(ui_json)
        else:
            self.params = self._param_class(**app_initializer)

        external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
        server = Flask(__name__)
        self.app = JupyterDash(
            server=server,
            url_base_pathname=os.environ.get("JUPYTERHUB_SERVICE_PREFIX", "/"),
            external_stylesheets=external_stylesheets,
        )

        super().__init__(**kwargs)

        self.app.layout = block_model_layout

        # Set up callbacks
        self.app.callback(
            Output(component_id="objects", component_property="options"),
            Output(component_id="objects", component_property="value"),
            Output(component_id="ui_json", component_property="data"),
            Output(component_id="upload", component_property="filename"),
            Output(component_id="upload", component_property="contents"),
            Input(component_id="upload", component_property="filename"),
            Input(component_id="upload", component_property="contents"),
        )(self.update_object_options)
        self.app.callback(
            Output(component_id="new_grid", component_property="value"),
            Output(component_id="cell_size_x", component_property="value"),
            Output(component_id="cell_size_y", component_property="value"),
            Output(component_id="cell_size_z", component_property="value"),
            Output(component_id="depth_core", component_property="value"),
            Output(component_id="horizontal_padding", component_property="value"),
            Output(component_id="bottom_padding", component_property="value"),
            Output(component_id="expansion_fact", component_property="value"),
            Output(component_id="output_path", component_property="value"),
            Input(component_id="ui_json", component_property="data"),
        )(self.update_remainder_from_ui_json)
        self.app.callback(
            Output(component_id="live_link", component_property="value"),
            Input(component_id="export", component_property="n_clicks"),
            Input(component_id="new_grid", component_property="value"),
            Input(component_id="objects", component_property="value"),
            Input(component_id="cell_size_x", component_property="value"),
            Input(component_id="cell_size_y", component_property="value"),
            Input(component_id="cell_size_z", component_property="value"),
            Input(component_id="depth_core", component_property="value"),
            Input(component_id="horizontal_padding", component_property="value"),
            Input(component_id="bottom_padding", component_property="value"),
            Input(component_id="expansion_fact", component_property="value"),
            Input(component_id="live_link", component_property="value"),
            Input(component_id="output_path", component_property="value"),
        )(self.trigger_click)

    def update_remainder_from_ui_json(
        self, ui_json: dict, output_ids: list | None = None
    ) -> tuple:
        """
        Update parameters from uploaded ui_json that aren't involved in another callback.

        :param ui_json: Uploaded ui_json.
        :param param_list: List of parameters to update. Used by tests.

        :return outputs: List of outputs corresponding to the callback expected outputs.
        """
        # List of outputs for the callback
        if output_ids is None:
            output_ids = [
                item["id"] + "_" + item["property"]
                for item in callback_context.outputs_list
            ]
        update_dict = self.update_param_list_from_ui_json(ui_json, output_ids)
        outputs = BaseDashApplication.get_outputs(output_ids, update_dict)

        return outputs

    def trigger_click(
        self,
        n_clicks: int,
        new_grid: str,
        objects: str,
        cell_size_x: float,
        cell_size_y: float,
        cell_size_z: float,
        depth_core: float,
        horizontal_padding: float,
        bottom_padding: float,
        expansion_fact: float,
        live_link: list,
        output_path: str,
        trigger: str = None,
    ) -> list:
        """
        When the export button is pressed, run block model driver to export block model.

        :param n_clicks: Triggers callback for pressing export button.
        :param new_grid: Name for exported block model.
        :param objects_uid: Input object uid.
        :param cell_size_x: X cell size for the core mesh.
        :param cell_size_y: Y cell size for the core mesh.
        :param cell_size_z: Z cell size for the core mesh.
        :param depth_core: Depth of core mesh below input object.
        :param horizontal_padding: Horizontal padding distance.
        :param bottom_padding: Bottom padding distance.
        :param expansion_fact: Expansion factor for padding cells.
        :param live_link: Checkbox for using monitoring directory.
        :param output_path: Output path for exporting block model.
        :param trigger: Dash component which triggered the callback.

        :return live_link: Checkbox for using monitoring directory.
        """
        if trigger is None:
            trigger = callback_context.triggered[0]["prop_id"].split(".")[0]
        if trigger == "export":
            # Get output path.
            if (
                (output_path is not None)
                and (output_path != "")
                and (os.path.exists(os.path.abspath(output_path)))
            ):
                output_path = os.path.abspath(output_path)
            else:
                print("Invalid output path.")
                raise PreventUpdate

            # Update self.params from dash component values
            param_dict = self.get_params_dict(locals())

            # Get output workspace.
            temp_geoh5 = f"BlockModel_{time():.0f}.geoh5"
            ws, param_dict["live_link"] = BaseApplication.get_output_workspace(
                param_dict["live_link"], output_path, temp_geoh5
            )

            with self.workspace.open():
                with ws as new_workspace:
                    # Put entities in output workspace.
                    param_dict["geoh5"] = new_workspace
                    for key, value in param_dict.items():
                        if isinstance(value, ObjectBase):
                            param_dict[key] = value.copy(
                                parent=new_workspace, copy_children=True
                            )

            # Write output uijson.
            new_params = BlockModelParams(**param_dict)
            new_params.write_input_file(
                name=temp_geoh5.replace(".geoh5", ".ui.json"),
                path=new_params.output_path,
                validate=False,
            )
            # Run driver.
            driver = BlockModelDriver(new_params)
            print("Creating block model . . .")
            driver.run()

            if new_params.live_link:
                print("Live link active. Check your ANALYST session for new mesh.")
                return ["Geoscience ANALYST Pro - Live link"]
            else:
                print("Saved to " + new_params.output_path)
                return []
        else:
            return no_update
