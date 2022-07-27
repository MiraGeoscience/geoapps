#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import os
from time import time

from dash import callback_context, dcc, html, no_update
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
from geoh5py.objects.object_base import ObjectBase

from geoapps.base.application import BaseApplication
from geoapps.base.dash_application import BaseDashApplication
from geoapps.block_model_creation.constants import app_initializer
from geoapps.block_model_creation.driver import BlockModelDriver
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

        super().__init__(**kwargs)

        # Set up the layout with the dash components
        self.app.layout = html.Div(
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
                            id="output_path",
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
                    options=["Geoscience ANALYST Pro - Live link"],
                    value=[],
                    style={"margin_bottom": "20px"},
                ),
                html.Button("Export", id="export"),
                dcc.Markdown(id="output_message"),
                dcc.Store(id="ui_json"),
            ],
            style={
                "margin_left": "20px",
                "margin_top": "20px",
                "width": "75%",
            },
        )

        # Set up callbacks
        self.app.callback(
            Output(component_id="ui_json", component_property="data"),
            Output(component_id="objects", component_property="options"),
            Input(component_id="upload", component_property="filename"),
            Input(component_id="upload", component_property="contents"),
        )(self.update_object_options)
        self.app.callback(
            Output(component_id="new_grid", component_property="value"),
            Output(component_id="objects", component_property="value"),
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

    def update_remainder_from_ui_json(self, ui_json: dict) -> tuple:
        """
        Update parameters from uploaded ui_json that aren't involved in another callback.

        :param ui_json: Uploaded ui_json.

        :return outputs: List of outputs corresponding to the callback expected outputs.
        """
        # List of outputs for the callback
        param_list = [i["id"] for i in callback_context.outputs_list]
        update_dict = self.update_param_list_from_ui_json(ui_json, param_list)
        outputs = BaseDashApplication.get_outputs(param_list, update_dict)

        return outputs

    def trigger_click(
        self,
        n_clicks: int,
        new_grid: str,
        objects_name: str,
        cell_size_x: float,
        cell_size_y: float,
        cell_size_z: float,
        depth_core: int,
        horizontal_padding: float,
        bottom_padding: float,
        expansion_fact: float,
        live_link: list,
        output_path: str,
    ) -> list:
        """
        When the export button is pressed, run block model driver to export block model.

        :param n_clicks: Triggers callback for pressing export button.
        :param new_grid: Name for exported block model.
        :param objects_name: Input object name.
        :param cell_size_x: X cell size for the core mesh.
        :param cell_size_y: Y cell size for the core mesh.
        :param cell_size_z: Z cell size for the core mesh.
        :param depth_core: Depth of core mesh below input object.
        :param horizontal_padding: Horizontal padding distance.
        :param bottom_padding: Bottom padding distance.
        :param expansion_fact: Expansion factor for padding cells.
        :param live_link: Checkbox for using monitoring directory.
        :param output_path: Output path for exporting block model.

        :return live_link: Checkbox for using monitoring directory.
        """

        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]

        if trigger == "export":
            # Update self.params from dash component values
            self.update_params(locals())

            temp_geoh5 = f"BlockModel_{time():.0f}.geoh5"

            # Get output path.
            if (
                (self.params.output_path is not None)
                and (self.params.output_path != "")
                and (os.path.exists(os.path.abspath(self.params.output_path)))
            ):
                self.params.output_path = os.path.abspath(self.params.output_path)
            else:
                print("Invalid output path.")
                raise PreventUpdate

            # Get output workspace.
            ws, self.params.live_link = BaseApplication.get_output_workspace(
                self.params.live_link, self.params.output_path, temp_geoh5
            )

            param_dict = self.params.to_dict()

            with ws as workspace:
                # Put entities in output workspace.
                param_dict["geoh5"] = workspace
                for key, value in param_dict.items():
                    if isinstance(value, ObjectBase):
                        param_dict[key] = value.copy(
                            parent=workspace, copy_children=True
                        )

            # Write output uijson.
            new_params = BlockModelParams(**param_dict)
            new_params.write_input_file(
                name=temp_geoh5.replace(".geoh5", ".ui.json"),
                path=self.params.output_path,
                validate=False,
            )
            # Run driver.
            driver = BlockModelDriver(new_params)
            print("Creating block model . . .")
            driver.run()

            if self.params.live_link:
                print("Live link active. Check your ANALYST session for new mesh.")
                return ["Geoscience ANALYST Pro - Live link"]
            else:
                print("Saved to " + self.params.output_path)
                return []
        else:
            return no_update
