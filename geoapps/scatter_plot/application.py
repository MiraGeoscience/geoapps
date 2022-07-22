#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import ast
import base64
import io
import json
import os
import sys
import uuid
import webbrowser
from os import environ

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

from geoapps.scatter_plot.constants import app_initializer
from geoapps.scatter_plot.driver import ScatterPlotDriver
from geoapps.scatter_plot.params import ScatterPlotParams


class ScatterPlots:
    _param_class = ScatterPlotParams

    def __init__(self, ui_json=None, clustering=False, **kwargs):
        app_initializer.update(kwargs)
        if ui_json is not None and os.path.exists(ui_json.path):
            self.params = self._param_class(ui_json)
        else:
            self.params = self._param_class(**app_initializer)

        self.data_channels = {}
        # Initial values for the dash components
        self.defaults = self.get_defaults()
        if clustering:
            self.defaults["color_maps_options"] = px.colors.named_colorscales() + [
                "kmeans"
            ]
            self.defaults["data_options"].append("kmeans")

        external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
        server = Flask(__name__)
        self.app = JupyterDash(
            server=server,
            url_base_pathname=environ.get("JUPYTERHUB_SERVICE_PREFIX", "/"),
            external_stylesheets=external_stylesheets,
        )

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
                    options=self.defaults["objects_options"],
                    value=self.defaults["objects_name"],
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
                            value=self.defaults["downsampling"],
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
                        # "width": "40%",
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
                            options=self.defaults["data_options"],
                            value=self.defaults["x_name"],
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
                                    value=self.defaults["x_thresh"],
                                    style={
                                        "display": "inline-block",
                                        "margin-right": "20px",
                                    },
                                ),
                                dcc.Checklist(
                                    id="x_log",
                                    options=["Log10"],
                                    value=self.defaults["x_log"],
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
                                    value=self.defaults["x_min"],
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
                                    value=self.defaults["x_max"],
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
                            options=self.defaults["data_options"],
                            value=self.defaults["y_name"],
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
                                    value=self.defaults["y_thresh"],
                                    style={
                                        "display": "inline-block",
                                        "margin-right": "20px",
                                    },
                                ),
                                dcc.Checklist(
                                    id="y_log",
                                    options=["Log10"],
                                    value=self.defaults["y_log"],
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
                                    value=self.defaults["y_min"],
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
                                    value=self.defaults["y_max"],
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
                            options=self.defaults["data_options"],
                            value=self.defaults["z_name"],
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
                                    value=self.defaults["z_thresh"],
                                    style={
                                        "display": "inline-block",
                                        "margin-right": "20px",
                                    },
                                ),
                                dcc.Checklist(
                                    id="z_log",
                                    options=["Log10"],
                                    value=self.defaults["z_log"],
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
                                    value=self.defaults["z_min"],
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
                                    value=self.defaults["z_max"],
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
                            options=self.defaults["data_options"],
                            value=self.defaults["color_name"],
                            style={"margin-bottom": "20px"},
                        ),
                        dcc.Markdown(children="Colormap: "),
                        dcc.Dropdown(
                            id="color_maps",
                            options=self.defaults["color_maps_options"],
                            value=self.defaults["color_maps"],
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
                                    value=self.defaults["color_thresh"],
                                    style={
                                        "display": "inline-block",
                                        "margin-right": "20px",
                                    },
                                ),
                                dcc.Checklist(
                                    id="color_log",
                                    options=["Log10"],
                                    value=self.defaults["color_log"],
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
                                    value=self.defaults["color_min"],
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
                                    value=self.defaults["color_max"],
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
                            options=self.defaults["data_options"],
                            value=self.defaults["size_name"],
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
                                    value=self.defaults["size_markers"],
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
                                    value=self.defaults["size_thresh"],
                                    style={
                                        "display": "inline-block",
                                        "margin-right": "20px",
                                    },
                                ),
                                dcc.Checklist(
                                    id="size_log",
                                    options=["Log10"],
                                    value=self.defaults["size_log"],
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
                                    value=self.defaults["size_min"],
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
                                    value=self.defaults["size_max"],
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
                        html.A(
                            html.Button("Download as HTML"),
                            id="download",
                            download="Crossplot.html",
                            style={"margin-left": "30%"},
                        ),
                    ],
                    style={
                        "width": "55%",
                        "display": "inline-block",
                        "margin-bottom": "20px",
                        "vertical-align": "bottom",
                    },
                ),
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
        )(self.update_visibility)
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
            Output(component_id="objects", component_property="options"),
            Output(component_id="objects", component_property="value"),
            Output(component_id="downsampling", component_property="value"),
            Output(component_id="x", component_property="options"),
            Output(component_id="x", component_property="value"),
            Output(component_id="x_log", component_property="value"),
            Output(component_id="x_thresh", component_property="value"),
            Output(component_id="x_min", component_property="value"),
            Output(component_id="x_max", component_property="value"),
            Output(component_id="y", component_property="options"),
            Output(component_id="y", component_property="value"),
            Output(component_id="y_log", component_property="value"),
            Output(component_id="y_thresh", component_property="value"),
            Output(component_id="y_min", component_property="value"),
            Output(component_id="y_max", component_property="value"),
            Output(component_id="z", component_property="options"),
            Output(component_id="z", component_property="value"),
            Output(component_id="z_log", component_property="value"),
            Output(component_id="z_thresh", component_property="value"),
            Output(component_id="z_min", component_property="value"),
            Output(component_id="z_max", component_property="value"),
            Output(component_id="color", component_property="options"),
            Output(component_id="color", component_property="value"),
            Output(component_id="color_log", component_property="value"),
            Output(component_id="color_thresh", component_property="value"),
            Output(component_id="color_min", component_property="value"),
            Output(component_id="color_max", component_property="value"),
            Output(component_id="color_maps", component_property="value"),
            Output(component_id="size", component_property="options"),
            Output(component_id="size", component_property="value"),
            Output(component_id="size_log", component_property="value"),
            Output(component_id="size_thresh", component_property="value"),
            Output(component_id="size_min", component_property="value"),
            Output(component_id="size_max", component_property="value"),
            Output(component_id="size_markers", component_property="value"),
            Output(component_id="upload", component_property="filename"),
            Output(component_id="upload", component_property="contents"),
            Input(component_id="upload", component_property="filename"),
            Input(component_id="upload", component_property="contents"),
            Input(component_id="objects", component_property="value"),
            Input(component_id="x", component_property="value"),
            Input(component_id="y", component_property="value"),
            Input(component_id="z", component_property="value"),
            Input(component_id="color", component_property="value"),
            Input(component_id="size", component_property="value"),
            prevent_initial_call=True,
        )(self.update_params)
        self.app.callback(
            Output(component_id="download", component_property="href"),
            Input(component_id="crossplot", component_property="figure"),
        )(self.save_figure)

    def update_visibility(self, axis):
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

    def get_defaults(self):
        defaults = {}
        # Get initial values to initialize the dash components
        for key, value in self.params.to_dict().items():
            if key in ["x", "y", "z", "color", "size"]:
                defaults[key + "_name"] = getattr(value, "name", None)
            elif key == "objects":
                if value is None:
                    defaults[key + "_name"] = None
                    defaults["data_options"] = []
                else:
                    defaults[key + "_name"] = value.name
                    data_options = ["None"]
                    data_options.extend(value.get_data_list())
                    if "Visual Parameters" in data_options:
                        data_options.remove("Visual Parameters")
                    defaults["data_options"] = data_options
                for channel in defaults["data_options"]:
                    self.get_channel(channel)
            elif key in ["x_log", "y_log", "z_log", "color_log", "size_log"]:
                if value is True:
                    defaults[key] = ["Log10"]
                else:
                    defaults[key] = []
            else:
                defaults[key] = value

            if key == "geoh5":
                if value is None:
                    defaults["objects_options"] = []
                else:
                    defaults["objects_options"] = [
                        {"label": obj.parent.name + "/" + obj.name, "value": obj.name}
                        for obj in value.objects
                    ]
        defaults["color_maps_options"] = px.colors.named_colorscales()

        if "plot_kmeans" in self.params.to_dict().keys():
            plot_kmeans = ast.literal_eval(self.params.plot_kmeans)
            defaults["plot_kmeans"] = plot_kmeans
            axis_list = ["x", "y", "z", "color", "size"]
            for i in range(len(plot_kmeans)):
                if plot_kmeans[i]:
                    defaults[axis_list[i] + "_name"] = "kmeans"

        return defaults

    def update_plot(
        self,
        downsampling,
        x,
        x_log,
        x_thresh,
        x_min,
        x_max,
        y,
        y_log,
        y_thresh,
        y_min,
        y_max,
        z,
        z_log,
        z_thresh,
        z_min,
        z_max,
        color,
        color_log,
        color_thresh,
        color_min,
        color_max,
        color_maps,
        size,
        size_log,
        size_thresh,
        size_min,
        size_max,
        size_markers,
        clustering=False,
    ):
        new_params_dict = {}
        for key, value in self.params.to_dict().items():

            if key in locals():
                param = locals()[key]
            else:
                param = None

            if param is None:
                if key in ["x", "y", "z", "color", "size"]:
                    new_params_dict[key] = None
                else:
                    new_params_dict[key] = value
            elif (
                (key == "x")
                | (key == "y")
                | (key == "z")
                | (key == "color")
                | (key == "size")
            ) & (clustering is False):
                if (param != "None") & (param in self.data_channels.keys()):
                    new_params_dict[key] = self.data_channels[param]
                else:
                    new_params_dict[key] = None
            else:
                new_params_dict[key] = param

        ifile = InputFile(
            ui_json=self.params.input_file.ui_json,
            validation_options={"disabled": True},
        )

        ifile.data = new_params_dict
        new_params = ScatterPlotParams(input_file=ifile)

        driver = ScatterPlotDriver(new_params)
        figure = go.FigureWidget(driver.run())

        return figure

    def update_from_uijson(self, contents):
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        ui_json = json.loads(decoded)

        for key, value in ui_json.items():
            if hasattr(self.params, key):
                if key == "geoh5":
                    if value == "":
                        setattr(self.params, key, None)
                    else:
                        setattr(self.params, key, value)
                elif type(value) is dict:
                    if key in ["objects", "x", "y", "z", "color", "size"]:
                        if (
                            (value["value"] is None)
                            | (value["value"] == "")
                            | (self.params.geoh5 is None)
                        ):
                            setattr(self.params, key, None)
                        elif self.params.geoh5.get_entity(uuid.UUID(value["value"])):
                            setattr(
                                self.params,
                                key,
                                self.params.geoh5.get_entity(uuid.UUID(value["value"]))[
                                    0
                                ],
                            )
                    elif value["value"]:
                        setattr(self.params, key, value["value"])
                    else:
                        setattr(self.params, key, None)
                elif key == "channels":
                    setattr(self.params, key, value.replace("'", "").split(", "))
                else:
                    setattr(self.params, key, value)

        self.data_channels = {}
        defaults = self.get_defaults()

        return defaults

    def update_object_options(self, contents):
        objects, value = None, None
        if contents is not None:
            content_type, content_string = contents.split(",")
            decoded = io.BytesIO(base64.b64decode(content_string))
            self.params.geoh5 = Workspace(decoded)

        obj_list = self.params.geoh5.objects

        options = [
            {"label": obj.parent.name + "/" + obj.name, "value": obj.name}
            for obj in obj_list
        ]
        if len(options) > 0:
            value = options[0]["value"]

        return {"objects_options": options, "objects_name": value}

    def update_data_options(self, object):
        obj = None
        if getattr(
            self.params, "geoh5", None
        ) is not None and self.params.geoh5.get_entity(object):
            for entity in self.params.geoh5.get_entity(object):
                if isinstance(entity, ObjectBase):
                    obj = entity

        channel_list = ["None"]
        channel_list.extend(obj.get_data_list())

        if "Visual Parameters" in channel_list:
            channel_list.remove("Visual Parameters")

        self.data_channels = {}
        for channel in channel_list:
            self.get_channel(channel)

        options = list(self.data_channels.keys())

        return {
            "data_options": options,
            "x_name": "None",
            "y_name": "None",
            "z_name": "None",
            "color_name": "None",
            "size_name": "None",
        }

    def get_channel(self, channel):
        if channel is None:
            return None

        if channel not in self.data_channels.keys():
            if channel == "None":
                data = None
            elif self.params.geoh5.get_entity(channel):
                data = self.params.geoh5.get_entity(channel)[0]
            else:
                return

            self.data_channels[channel] = data

    def get_channel_bounds(self, channel):
        """
        Set the min and max values for the given axis channel
        """
        self.get_channel(channel)

        cmin, cmax = None, None
        if (channel in self.data_channels.keys()) & (channel != "None"):
            values = self.data_channels[channel].values
            values = values[~np.isnan(values)]
            cmin = float(f"{np.min(values):.2e}")
            cmax = float(f"{np.max(values):.2e}")

        return cmin, cmax

    def set_channel_bounds(self, x, y, z, color, size):
        x_min, x_max = self.get_channel_bounds(x)
        y_min, y_max = self.get_channel_bounds(y)
        z_min, z_max = self.get_channel_bounds(z)
        color_min, color_max = self.get_channel_bounds(color)
        size_min, size_max = self.get_channel_bounds(size)

        return {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "z_min": z_min,
            "z_max": z_max,
            "color_min": color_min,
            "color_max": color_max,
            "size_min": size_min,
            "size_max": size_max,
        }

    def update_params(
        self,
        filename,
        contents,
        objects,
        x,
        y,
        z,
        color,
        size,
    ):
        param_list = [
            "objects_options",
            "objects_name",
            "downsampling",
            "data_options",
            "x_name",
            "x_log",
            "x_thresh",
            "x_min",
            "x_max",
            "data_options",
            "y_name",
            "y_log",
            "y_thresh",
            "y_min",
            "y_max",
            "data_options",
            "z_name",
            "z_log",
            "z_thresh",
            "z_min",
            "z_max",
            "data_options",
            "color_name",
            "color_log",
            "color_thresh",
            "color_min",
            "color_max",
            "color_maps",
            "data_options",
            "size_name",
            "size_log",
            "size_thresh",
            "size_min",
            "size_max",
            "size_markers",
            "filename",
            "contents",
        ]

        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]
        update_dict = {}
        if trigger == "upload":
            if filename.endswith(".ui.json"):
                update_dict = self.update_from_uijson(contents)
            elif filename.endswith(".geoh5"):
                update_dict = self.update_object_options(contents)
                update_dict.update(
                    self.update_data_options(update_dict["objects_name"])
                )
            else:
                print("Uploaded file must be a workspace or ui.json.")
            update_dict["filename"] = None
            update_dict["contents"] = None
        elif trigger == "objects":
            update_dict = self.update_data_options(objects)
        elif trigger in ["x", "y", "z", "color", "size"]:
            update_dict = self.set_channel_bounds(x, y, z, color, size)

        outputs = []
        for param in param_list:
            if param in update_dict.keys():
                outputs.append(update_dict[param])
            else:
                outputs.append(no_update)
        return tuple(outputs)

    def save_figure(self, fig):
        buffer = io.StringIO()
        go.Figure(fig).write_html(buffer)
        html_bytes = buffer.getvalue().encode()
        encoded = base64.b64encode(html_bytes).decode()
        href = "data:text/html;base64," + encoded

        return href

    def run(self):
        # The reloader has not yet run - open the browser
        if not environ.get("WERKZEUG_RUN_MAIN"):
            webbrowser.open_new("http://127.0.0.1:8050/")

        # Otherwise, continue as normal
        self.app.run_server(host="127.0.0.1", port=8050, debug=False)


if __name__ == "__main__":
    print("Loading geoh5 file . . .")
    file = sys.argv[1]
    ifile = InputFile.read_ui_json(file)
    app = ScatterPlots(ui_json=ifile)
    print("Loaded. Building the plotly scatterplot . . .")
    app.run()
    print("Done")
