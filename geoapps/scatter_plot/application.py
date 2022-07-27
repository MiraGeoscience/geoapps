#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import os
import sys
from time import time

import numpy as np
import plotly.graph_objects as go
from dash import callback_context, dcc, html
from dash.dependencies import Input, Output
from geoh5py.objects import ObjectBase
from geoh5py.ui_json import InputFile
from geoh5py.workspace import Workspace

from geoapps.base.application import BaseApplication
from geoapps.base.dash_application import BaseDashApplication
from geoapps.scatter_plot.constants import app_initializer
from geoapps.scatter_plot.driver import ScatterPlotDriver
from geoapps.scatter_plot.params import ScatterPlotParams


class ScatterPlots(BaseDashApplication):
    _param_class = ScatterPlotParams

    def __init__(self, ui_json=None, **kwargs):
        app_initializer.update(kwargs)
        if ui_json is not None and os.path.exists(ui_json.path):
            self.params = self._param_class(ui_json)
        else:
            self.params = self._param_class(**app_initializer)

        super().__init__(**kwargs)

        # Set up the layout with the dash components
        self.workspace_layout = html.Div(
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
        self.axis_layout = html.Div(
            [
                html.Div(
                    [
                        dcc.Markdown(children="Axis: "),
                        dcc.Dropdown(
                            id="axes_pannels",
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
                        dcc.Dropdown(
                            id="color_maps",
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
        self.plot_layout = html.Div(
            [
                dcc.Graph(
                    id="crossplot",
                    style={"margin-bottom": "20px"},
                ),
            ]
        )
        self.app.layout = html.Div(
            [
                html.Div(
                    [self.workspace_layout, self.axis_layout],
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
                        self.plot_layout,
                        dcc.Markdown("Output path: "),
                        html.Div(
                            [
                                dcc.Input(id="output_path"),
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
                dcc.Store(id="ui_json"),
            ],
            style={"width": "70%", "margin-left": "50px", "margin-top": "30px"},
        )

        # Set up callbacks
        self.app.callback(
            Output(component_id="x_div", component_property="style"),
            Output(component_id="y_div", component_property="style"),
            Output(component_id="z_div", component_property="style"),
            Output(component_id="color_div", component_property="style"),
            Output(component_id="size_div", component_property="style"),
            Input(component_id="axes_pannels", component_property="value"),
        )(ScatterPlots.update_visibility)
        self.app.callback(
            Output(component_id="ui_json", component_property="data"),
            Output(component_id="objects", component_property="options"),
            Output(component_id="upload", component_property="filename"),
            Output(component_id="upload", component_property="contents"),
            Input(component_id="upload", component_property="filename"),
            Input(component_id="upload", component_property="contents"),
        )(self.update_object_options)
        self.app.callback(
            Output(component_id="x", component_property="options"),
            Output(component_id="y", component_property="options"),
            Output(component_id="z", component_property="options"),
            Output(component_id="color", component_property="options"),
            Output(component_id="size", component_property="options"),
            Input(component_id="ui_json", component_property="data"),
            Input(component_id="objects", component_property="value"),
        )(self.update_data_options)
        self.app.callback(
            Output(component_id="x_min", component_property="value"),
            Output(component_id="x_max", component_property="value"),
            Output(component_id="y_min", component_property="value"),
            Output(component_id="y_max", component_property="value"),
            Output(component_id="z_min", component_property="value"),
            Output(component_id="z_max", component_property="value"),
            Output(component_id="color_min", component_property="value"),
            Output(component_id="color_max", component_property="value"),
            Output(component_id="size_min", component_property="value"),
            Output(component_id="size_max", component_property="value"),
            Input(component_id="ui_json", component_property="data"),
            Input(component_id="x", component_property="value"),
            Input(component_id="y", component_property="value"),
            Input(component_id="z", component_property="value"),
            Input(component_id="color", component_property="value"),
            Input(component_id="size", component_property="value"),
        )(self.update_channel_bounds)
        self.app.callback(
            Output(component_id="objects", component_property="value"),
            Output(component_id="downsampling", component_property="value"),
            Output(component_id="x", component_property="value"),
            Output(component_id="x_log", component_property="value"),
            Output(component_id="x_thresh", component_property="value"),
            Output(component_id="y", component_property="value"),
            Output(component_id="y_log", component_property="value"),
            Output(component_id="y_thresh", component_property="value"),
            Output(component_id="z", component_property="value"),
            Output(component_id="z_log", component_property="value"),
            Output(component_id="z_thresh", component_property="value"),
            Output(component_id="color", component_property="value"),
            Output(component_id="color_log", component_property="value"),
            Output(component_id="color_thresh", component_property="value"),
            Output(component_id="color_maps", component_property="options"),
            Output(component_id="color_maps", component_property="value"),
            Output(component_id="size", component_property="value"),
            Output(component_id="size_log", component_property="value"),
            Output(component_id="size_thresh", component_property="value"),
            Output(component_id="size_markers", component_property="value"),
            Output(component_id="output_path", component_property="value"),
            Input(component_id="ui_json", component_property="data"),
        )(self.update_remainder_from_ui_json)
        self.app.callback(
            Output(component_id="crossplot", component_property="figure"),
            Input(component_id="downsampling", component_property="value"),
            Input(component_id="x", component_property="value"),
            Input(component_id="x_log", component_property="value"),
            Input(component_id="x_thresh", component_property="value"),
            Input(component_id="x_min", component_property="value"),
            Input(component_id="x_max", component_property="value"),
            Input(component_id="y", component_property="value"),
            Input(component_id="y_log", component_property="value"),
            Input(component_id="y_thresh", component_property="value"),
            Input(component_id="y_min", component_property="value"),
            Input(component_id="y_max", component_property="value"),
            Input(component_id="z", component_property="value"),
            Input(component_id="z_log", component_property="value"),
            Input(component_id="z_thresh", component_property="value"),
            Input(component_id="z_min", component_property="value"),
            Input(component_id="z_max", component_property="value"),
            Input(component_id="color", component_property="value"),
            Input(component_id="color_log", component_property="value"),
            Input(component_id="color_thresh", component_property="value"),
            Input(component_id="color_min", component_property="value"),
            Input(component_id="color_max", component_property="value"),
            Input(component_id="color_maps", component_property="value"),
            Input(component_id="size", component_property="value"),
            Input(component_id="size_log", component_property="value"),
            Input(component_id="size_thresh", component_property="value"),
            Input(component_id="size_min", component_property="value"),
            Input(component_id="size_max", component_property="value"),
            Input(component_id="size_markers", component_property="value"),
        )(self.update_plot)
        self.app.callback(
            Output(component_id="export", component_property="n_clicks"),
            Input(component_id="export", component_property="n_clicks"),
            Input(component_id="output_path", component_property="value"),
            Input(component_id="crossplot", component_property="figure"),
            prevent_initial_call=True,
        )(self.save_figure)

    @staticmethod
    def update_visibility(axis):
        # Change the visibility of the dash components depending on the axis selected
        if axis == "x":
            return (
                {"display": "block"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
            )
        elif axis == "y":
            return (
                {"display": "none"},
                {"display": "block"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
            )
        elif axis == "z":
            return (
                {"display": "none"},
                {"display": "none"},
                {"display": "block"},
                {"display": "none"},
                {"display": "none"},
            )
        elif axis == "color":
            return (
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "block"},
                {"display": "none"},
            )
        elif axis == "size":
            return (
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "block"},
            )

    def get_channel_bounds(self, channel):
        """
        Set the min and max values for the given axis channel
        """
        cmin, cmax = None, None
        if self.params.geoh5.get_entity(channel)[0] is not None:
            data = self.params.geoh5.get_entity(channel)[0]
            cmin = float(f"{np.nanmin(data.values):.2e}")
            cmax = float(f"{np.nanmax(data.values):.2e}")

        return cmin, cmax

    def update_channel_bounds(self, ui_json, x, y, z, color, size):
        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]

        if trigger == "ui_json":
            x_min, x_max = ui_json["x_min"]["value"], ui_json["x_max"]["value"]
            y_min, y_max = ui_json["y_min"]["value"], ui_json["y_max"]["value"]
            z_min, z_max = ui_json["z_min"]["value"], ui_json["z_max"]["value"]
            color_min, color_max = (
                ui_json["color_min"]["value"],
                ui_json["color_max"]["value"],
            )
            size_min, size_max = (
                ui_json["size_min"]["value"],
                ui_json["size_max"]["value"],
            )
        else:
            x_min, x_max = self.get_channel_bounds(x)
            y_min, y_max = self.get_channel_bounds(y)
            z_min, z_max = self.get_channel_bounds(z)
            color_min, color_max = self.get_channel_bounds(color)
            size_min, size_max = self.get_channel_bounds(size)

        return (
            x_min,
            x_max,
            y_min,
            y_max,
            z_min,
            z_max,
            color_min,
            color_max,
            size_min,
            size_max,
        )

    def update_remainder_from_ui_json(self, ui_json):
        # List of outputs for the callback'
        output_ids = [
            item["id"] + "_" + item["property"]
            for item in callback_context.outputs_list
        ]
        update_dict = self.update_param_list_from_ui_json(ui_json, output_ids)
        outputs = BaseDashApplication.get_outputs(output_ids, update_dict)

        return outputs

    def update_plot(
        self,
        downsampling,
        x_name,
        x_log,
        x_thresh,
        x_min,
        x_max,
        y_name,
        y_log,
        y_thresh,
        y_min,
        y_max,
        z_name,
        z_log,
        z_thresh,
        z_min,
        z_max,
        color_name,
        color_log,
        color_thresh,
        color_min,
        color_max,
        color_maps,
        size_name,
        size_log,
        size_thresh,
        size_min,
        size_max,
        size_markers,
        clustering=False,
    ):
        self.update_params(locals())

        new_params = ScatterPlotParams(**self.params.to_dict())
        # Run driver.
        driver = ScatterPlotDriver(new_params)
        figure = go.FigureWidget(driver.run())

        return figure

    def save_figure(self, n_clicks, output_path, figure):
        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]

        if trigger == "export":
            # Get output path.
            if (
                (output_path is not None)
                and (output_path != "")
                and (os.path.exists(os.path.abspath(output_path)))
            ):
                temp_geoh5 = f"Scatterplot_{time():.0f}.geoh5"

                output_path = os.path.abspath(output_path)
                go.Figure(figure).write_html(
                    os.path.join(output_path, temp_geoh5.replace(".geoh5", ".html"))
                )

                param_dict = self.params.to_dict()

                ws, _ = BaseApplication.get_output_workspace(
                    False, output_path, temp_geoh5
                )

                with ws as workspace:
                    # Put entities in output workspace.
                    param_dict["geoh5"] = workspace

                    for key, value in param_dict.items():
                        if isinstance(value, ObjectBase):
                            param_dict[key] = value.copy(
                                parent=workspace, copy_children=True
                            )

                    # Write output uijson.
                    new_params = ScatterPlotParams(**param_dict)
                    new_params.write_input_file(
                        name=temp_geoh5.replace(".geoh5", ".ui.json"),
                        path=output_path,
                        validate=False,
                    )

                print("Saved to " + output_path)
            else:
                print("Invalid output path.")

        return 0


if __name__ == "__main__":
    print("Loading geoh5 file . . .")
    file = sys.argv[1]
    ifile = InputFile.read_ui_json(file)
    app = ScatterPlots(ui_json=ifile)
    print("Loaded. Building the plotly scatterplot . . .")
    app.run()
    print("Done")
