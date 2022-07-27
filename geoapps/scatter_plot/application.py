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

from geoapps.base.application import BaseApplication
from geoapps.base.dash_application import BaseDashApplication
from geoapps.scatter_plot.constants import app_initializer
from geoapps.scatter_plot.driver import ScatterPlotDriver
from geoapps.scatter_plot.params import ScatterPlotParams


class ScatterPlots(BaseDashApplication):
    """
    Dash app to make a scatter plot.
    """

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
            Input(component_id="axes_panels", component_property="value"),
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
    def update_visibility(axis: str) -> (dict, dict, dict, dict, dict):
        """
        Change the visibility of the dash components depending on the axis selected.

        :param axis: Selected data axis.

        :return x-style: X axis style dict.
        :return y-style: Y axis style dict.
        :return z-style: Z axis style dict.
        :return color-style: Color axis style dict.
        :return size-style: Size axis style dict.
        """
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

    def get_channel_bounds(self, channel: str) -> (float, float):
        """
        Set the min and max values for the given axis channel.

        :param channel: Name of channel to find data for.

        :return cmin: Minimum value for input channel.
        :return cmax: Maximum value for input channel.
        """
        cmin, cmax = None, None
        if self.params.geoh5.get_entity(channel)[0] is not None:
            data = self.params.geoh5.get_entity(channel)[0]
            cmin = float(f"{np.nanmin(data.values):.2e}")
            cmax = float(f"{np.nanmax(data.values):.2e}")

        return cmin, cmax

    def update_channel_bounds(
        self, ui_json: dict, x: str, y: str, z: str, color: str, size: str
    ):
        """
        Update min and max for all channels, either from uploaded ui.json or from change of data.

        :param ui_json: Uploaded ui.json.
        :param x: Name of selected x data.
        :param y: Name of selected y data.
        :param z: Name of selected z data.
        :param color: Name of selected color data.
        :param size: Name of selected size data.

        :return x_min: Minimum value for x data.
        :return x_max: Maximum value for x data.
        :return y_min: Minimum value for y data.
        :return y_max: Maximum value for y data.
        :return z_min: Minimum value for z data.
        :return z_max: Maximum value for z data.
        :return color_min: Minimum value for color data.
        :return color_max: Maximum value for color data.
        :return size_min: Minimum value for size data.
        :return size_max: Maximum value for size data.
        """
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

    def update_remainder_from_ui_json(
        self, ui_json: dict
    ) -> (
        str,
        int,
        str,
        list,
        float,
        str,
        list,
        float,
        str,
        list,
        float,
        str,
        list,
        float,
        list,
        str,
        str,
        list,
        float,
        int,
        str,
    ):
        """
        Update parameters from uploaded ui.json, which aren't involved in other callbacks.

        :param ui_json: Uploaded ui.json.

        :return objects: Name of selected object.
        :return downsampling: Percent of total values to plot.
        :return x: Name of selected x data.
        :return x_log: Whether or not to plot the log of x data.
        :return x_thresh: X threshold.
        :return y: Name of selected y data.
        :return y_log: Whether or not to plot the log of y data.
        :return y_thresh: Y threshold.
        :return z: Name of selected z data.
        :return z_log: Whether or not to plot the log of z data.
        :return z_thresh: Z threshold.
        :return color: Name of selected color data.
        :return color_log: Whether or not to plot the log of color data.
        :return color_thresh: Color threshold.
        :return color_maps_options: Color map dropdown options.
        :return color_maps: Selected color map.
        :return size: Name of selected size data.
        :return size_log: Whether or not to plot the log of size data.
        :return size_thresh: Size threshold.
        :return size_markers: Max marker size.
        :return output_path: Output path for exporting scatter plot.
        """
        # List of outputs for the callback
        output_ids = [
            item["id"] + "_" + item["property"]
            for item in callback_context.outputs_list
        ]
        update_dict = self.update_param_list_from_ui_json(ui_json, output_ids)
        outputs = BaseDashApplication.get_outputs(output_ids, update_dict)

        return outputs

    def update_plot(
        self,
        downsampling: int,
        x_name: str,
        x_log: list,
        x_thresh: float,
        x_min: float,
        x_max: float,
        y_name: str,
        y_log: list,
        y_thresh: float,
        y_min: float,
        y_max: float,
        z_name: str,
        z_log: list,
        z_thresh: float,
        z_min: float,
        z_max: float,
        color_name: str,
        color_log: list,
        color_thresh: float,
        color_min: float,
        color_max: float,
        color_maps: str,
        size_name: str,
        size_log: list,
        size_thresh: float,
        size_min: float,
        size_max: float,
        size_markers: int,
    ) -> go.FigureWidget:
        """
        Update self.params, then run the scatter plot driver with the new params.

        :param downsampling: Percent of total values to plot.
        :param x_name: Name of selected x data.
        :param x_log: Whether or not to plot the log of x data.
        :param x_thresh: X threshold.
        :param x_min: Minimum value for x data.
        :param x_max: Maximum value for x data.
        :param y_name: Name of selected y data.
        :param y_log: Whether or not to plot the log of y data.
        :param y_thresh: Y threshold.
        :param y_min: Minimum value for y data.
        :param y_max: Maximum value for y data.
        :param z_name: Name of selected z data.
        :param z_log: Whether or not to plot the log of z data.
        :param z_thresh: Z threshold.
        :param z_min: Minimum value for z data.
        :param z_max: Maximum value for x data.
        :param color_name: Name of selected color data.
        :param color_log: Whether or not to plot the log of color data.
        :param color_thresh: Color threshold.
        :param color_min: Minimum value for color data.
        :param color_max: Maximum value for color data.
        :param color_maps: Color map.
        :param size_name: Name of selected size data.
        :param size_log: Whether or not to plot the log of size data.
        :param size_thresh: Size threshold.
        :param size_min: Minimum value for size data.
        :param size_max: Maximum value for size data.
        :param size_markers: Max size for markers.

        :return figure: Scatter plot.
        """
        self.update_params(locals())

        new_params = ScatterPlotParams(**self.params.to_dict())
        # Run driver.
        driver = ScatterPlotDriver(new_params)
        figure = go.FigureWidget(driver.run())

        return figure

    def save_figure(self, n_clicks: int, output_path: str, figure: go.FigureWidget):
        """
        Save scatter plot to output path as html.

        :param n_clicks: Triggers callback for pressing export button.
        :param output_path: Path to download scatter plot.
        :param figure: Scatter plot.

        :return n_clicks: Placeholder for callback.
        """
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
