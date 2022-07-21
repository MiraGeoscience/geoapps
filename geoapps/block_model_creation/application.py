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
from time import time

from dash import callback_context, dcc, html
from dash.dependencies import Input, Output
from geoh5py.objects.object_base import ObjectBase
from geoh5py.workspace import Workspace

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
                            value=getattr(self.params.objects, "name", None),
                            options=[
                                {
                                    "label": obj.parent.name + "/" + obj.name,
                                    "value": obj.name,
                                }
                                for obj in self.params.geoh5.objects
                            ],
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
                            value=self.params.new_grid,
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
                            value=self.params.cell_size_x,
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
                            value=self.params.cell_size_y,
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
                            value=self.params.cell_size_z,
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
                            value=self.params.depth_core,
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
                            value=self.params.horizontal_padding,
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
                            value=self.params.bottom_padding,
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
                            value=self.params.expansion_fact,
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
                            value=self.params.output_path,
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
            ],
            style={
                "margin_left": "20px",
                "margin_top": "20px",
                "width": "75%",
            },
        )

        # Set up callbacks
        self.app.callback(
            Output(component_id="new_grid", component_property="value"),
            Output(component_id="objects", component_property="value"),
            Output(component_id="objects", component_property="options"),
            Output(component_id="cell_size_x", component_property="value"),
            Output(component_id="cell_size_y", component_property="value"),
            Output(component_id="cell_size_z", component_property="value"),
            Output(component_id="depth_core", component_property="value"),
            Output(component_id="horizontal_padding", component_property="value"),
            Output(component_id="bottom_padding", component_property="value"),
            Output(component_id="expansion_fact", component_property="value"),
            Output(component_id="output_path", component_property="value"),
            Output(component_id="upload", component_property="filename"),
            Output(component_id="upload", component_property="contents"),
            Input(component_id="upload", component_property="filename"),
            Input(component_id="upload", component_property="contents"),
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
        )(self.update_params)
        self.app.callback(
            Output(component_id="live_link", component_property="value"),
            Input(component_id="export", component_property="n_clicks"),
            prevent_initial_call=True,
        )(self.trigger_click)

    def update_params(
        self,
        filename: str,
        contents: str,
        new_grid: str,
        objects: ObjectBase,
        cell_size_x: float,
        cell_size_y: float,
        cell_size_z: float,
        depth_core: int,
        horizontal_padding: float,
        bottom_padding: float,
        expansion_fact: float,
        live_link: list,
        output_path: str,
    ) -> (
        str,
        str,
        dict,
        float,
        float,
        float,
        int,
        float,
        float,
        float,
        str,
        str,
        str,
    ):
        """
        Update self.params and dash components from user input, including ones that depend indirectly.

        :param filename: Input file filename. Workspace or ui_json.
        :param contents: Input file contents. Workspace or ui_json.
        :param new_grid: Name for exported block model.
        :param objects: Input object.
        :param cell_size_x: X cell size for the core mesh.
        :param cell_size_y: Y cell size for the core mesh.
        :param cell_size_z: Z cell size for the core mesh.
        :param depth_core: Depth of core mesh below input object.
        :param horizontal_padding: Horizontal padding distance.
        :param bottom_padding: Bottom padding distance.
        :param expansion_fact: Expansion factor for padding cells.
        :param live_link: Checkbox for using monitoring directory.
        :param output_path: Output path for exporting block model.

        :return new_grid: Name for exported block model.
        :return objects_name: Name for input object.
        :return objects_options: Dropdown options for input object.
        :return cell_size_x: X cell size for the core mesh.
        :return cell_size_y: Y cell size for the core mesh.
        :return cell_size_z: Z cell size for the core mesh.
        :return depth_core: Depth of core mesh below input object.
        :return horizontal_padding: Horizontal padding distance.
        :return bottom_padding: Bottom padding distance.
        :return expansion_fact: Expansion factor for padding cells.
        :return output_path: Output path for exporting block model.
        :return filename: Input file filename. Workspace or ui_json.
        :return contents: Input file contents. Workspace or ui_json.
        """
        param_list = [
            "new_grid",
            "objects_name",
            "objects_options",
            "cell_size_x",
            "cell_size_y",
            "cell_size_z",
            "depth_core",
            "horizontal_padding",
            "bottom_padding",
            "expansion_fact",
            "output_path",
            "filename",
            "contents",
        ]
        # Dash converts .0 numbers to int. Making sure typing is correct for floats:
        cell_size_x = float(cell_size_x)
        cell_size_y = float(cell_size_y)
        cell_size_z = float(cell_size_z)
        horizontal_padding = float(horizontal_padding)
        bottom_padding = float(bottom_padding)
        expansion_fact = float(expansion_fact)
        depth_core = float(depth_core)

        # Get the dash component that triggered the callback
        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]

        update_dict = {}
        if trigger == "upload":
            if (filename.endswith("ui.json") or filename.endswith(".geoh5")) and (
                contents is not None
            ):
                if filename.endswith(".ui.json"):
                    update_dict.update(self.update_from_ui_json(contents, param_list))
                elif filename.endswith(".geoh5"):
                    content_type, content_string = contents.split(",")
                    decoded = io.BytesIO(base64.b64decode(content_string))
                    update_dict["geoh5"] = Workspace(decoded)
                ws = update_dict["geoh5"]
                update_dict.update(self.update_object_options(ws))
                update_dict["filename"] = None
                update_dict["contents"] = None
            else:
                print("Uploaded file must be a workspace or ui.json.")
        elif trigger + "_name" in param_list:
            update_dict[trigger + "_name"] = locals()[trigger]
        elif trigger != "":
            update_dict[trigger] = locals()[trigger]

        # Update self.params with the new parameter values.
        self.update_param_dict(update_dict)
        outputs = self.get_outputs(param_list, update_dict)

        return outputs

    def trigger_click(self, _) -> list:
        """
        When the export button is pressed, run block model driver to export block model.

        :return live_link: Checkbox for using monitoring directory.
        """
        # self.params should be up to date whenever create_block_model is called.
        param_dict = self.params.to_dict()
        temp_geoh5 = f"BlockModel_{time():.0f}.geoh5"

        # Get output path.
        if self.params.output_path is not None and os.path.exists(
            os.path.abspath(self.params.output_path)
        ):
            output_path = os.path.abspath(self.params.output_path)
        else:
            output_path = os.path.dirname(self.params.geoh5.h5file)

        # Get output workspace.
        ws, self.params.live_link = BaseApplication.get_output_workspace(
            self.params.live_link, output_path, temp_geoh5
        )

        with ws as workspace:
            # Put entities in output workspace.
            param_dict["geoh5"] = workspace
            for key, value in param_dict.items():
                if isinstance(value, ObjectBase):
                    param_dict[key] = value.copy(parent=workspace, copy_children=True)

        # Write output uijson.
        new_params = BlockModelParams(**param_dict)
        new_params.write_input_file(
            name=temp_geoh5.replace(".geoh5", ".ui.json"),
            path=output_path,
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
            print("Saved to " + os.path.abspath(output_path))
            return []
