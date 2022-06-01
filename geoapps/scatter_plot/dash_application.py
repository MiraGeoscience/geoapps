#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import os
import webbrowser
from os import environ

import dash
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output
from flask import Flask
from geoh5py.ui_json import InputFile

from geoapps.base.selection import ObjectDataSelection
from geoapps.scatter_plot.constants import app_initializer
from geoapps.scatter_plot.driver import ScatterPlotDriver
from geoapps.scatter_plot.params import ScatterPlotParams


class ScatterPlots(ObjectDataSelection):
    """
    Application for 2D and 3D crossplots of data using symlog scaling
    """

    _param_class = ScatterPlotParams
    _select_multiple = True
    _add_groups = False
    _downsampling = None
    _color = None
    _x = None
    _y = None
    _z = None
    _size = None

    def __init__(self, ui_json=None, **kwargs):
        app_initializer.update(kwargs)
        if ui_json is not None and os.path.exists(ui_json):
            self.params = self._param_class(InputFile(ui_json))
        else:
            self.params = self._param_class(**app_initializer)

        self.defaults = {}
        for key, value in self.params.to_dict().items():
            self.defaults[key] = value

        self.data_channels = {}

        self.trigger.on_click(self.trigger_click)
        self.trigger.description = "Save HTML"

        super().__init__(**self.defaults)

    def get_channel(self, channel):
        obj, _ = self.get_selected_entities()

        if channel is None:
            return None

        if channel not in self.data_channels.keys():
            if channel == "None":
                data = None
            elif self.workspace.get_entity(channel):
                data = self.workspace.get_entity(channel)[0]
            else:
                return

            self.data_channels[channel] = data

    @staticmethod
    def get_channel_bounds(name):
        """
        Set the min and max values for the given axis channel
        """
        scatter.refresh.value = False

        channel = getattr(scatter, "_" + name).value
        scatter.get_channel(channel)

        if channel in scatter.data_channels.keys():
            if channel == "None":
                cmin = 0
                cmax = 0
            else:
                values = scatter.data_channels[channel].values
                values = values[~np.isnan(values)]
                cmin = f"{np.min(values):.2e}"
                cmax = f"{np.max(values):.2e}"

        scatter.refresh.value = True

        return cmin, cmax


scatter = ScatterPlots()

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

server = Flask(__name__)
app = dash.Dash(
    server=server,
    url_base_pathname=environ.get("JUPYTERHUB_SERVICE_PREFIX", "/"),
    external_stylesheets=external_stylesheets,
)


app.layout = html.Div(
    [
        html.Div(
            [
                dcc.Markdown(
                    """
                    Scatter Plots
                    """
                ),
            ],
            style={"width": "100%", "margin-left": "200px", "margin-bottom": "20px"},
        ),
        html.Div(
            [
                dcc.Upload(id="workspace"),
                dcc.Dropdown(
                    id="objects",
                    options=["New York City", "Montreal", "San Francisco"],
                    value="Montreal",
                ),
                dcc.Slider(id="downsampling", value=10, min=0, max=20, step=5),
                dcc.Dropdown(
                    id="axes_pannels",
                    options=[
                        {"label": "X-axis", "value": "x"},
                        {"label": "Y-axis", "value": "y"},
                        {"label": "Z-axis", "value": "z"},
                        {"label": "Color", "value": "color"},
                        {"label": "Size", "value": "size"},
                    ],
                    value="X-axis",
                ),
            ],
            style={"width": "100%"},
        ),
        html.Div(
            id="x_div",
            children=[
                dcc.Dropdown(
                    id="x",
                    options=["New York City", "Montreal", "San Francisco"],
                    value="Montreal",
                ),
                html.Div(
                    [
                        dcc.Checklist(id="x_log", options=["Log10"], value=["Log10"]),
                        dcc.Input(id="x_thresh", type="number", value=0.1),
                    ],
                    style={"width": "100%"},
                ),
                html.Div(
                    [
                        dcc.Input(id="x_min", type="number", value=-17),
                        dcc.Input(id="x_max", type="number", value=25.5),
                    ],
                    style={"width": "100%"},
                ),
            ],
            style={"display": "block"},
        ),
        html.Div(
            id="y_div",
            children=[
                dcc.Dropdown(
                    id="y",
                    options=["New York City", "Montreal", "San Francisco"],
                    value="Montreal",
                ),
                html.Div(
                    [
                        dcc.Checklist(id="y_log", options=["Log10"], value=["Log10"]),
                        dcc.Input(id="y_thresh", type="number", value=0.1),
                    ],
                    style={"width": "100%"},
                ),
                html.Div(
                    [
                        dcc.Input(id="y_min", type="number", value=-17),
                        dcc.Input(id="y_max", type="number", value=25.5),
                    ],
                    style={"width": "100%"},
                ),
            ],
            style={"display": "block"},
        ),
        html.Div(
            id="z_div",
            children=[
                dcc.Dropdown(
                    id="z",
                    options=["New York City", "Montreal", "San Francisco"],
                    value="Montreal",
                ),
                html.Div(
                    [
                        dcc.Checklist(id="z_log", options=["Log10"], value=["Log10"]),
                        dcc.Input(id="z_thresh", type="number", value=0.1),
                    ],
                    style={"width": "100%"},
                ),
                html.Div(
                    [
                        dcc.Input(id="z_min", type="number", value=-17),
                        dcc.Input(id="z_max", type="number", value=25.5),
                    ],
                    style={"width": "100%"},
                ),
            ],
            style={"display": "block"},
        ),
        html.Div(
            id="color_div",
            children=[
                dcc.Dropdown(
                    id="color",
                    options=["New York City", "Montreal", "San Francisco"],
                    value="Montreal",
                ),
                dcc.Dropdown(
                    id="color_maps",
                    options=["New York City", "Montreal", "San Francisco"],
                    value="Montreal",
                ),
                html.Div(
                    [
                        dcc.Checklist(
                            id="color_log", options=["Log10"], value=["Log10"]
                        ),
                        dcc.Input(id="color_thresh", type="number", value=0.1),
                    ],
                    style={"width": "100%"},
                ),
                html.Div(
                    [
                        dcc.Input(id="color_min", type="number", value=-17),
                        dcc.Input(id="color_max", type="number", value=25.5),
                    ],
                    style={"width": "100%"},
                ),
            ],
            style={"display": "block"},
        ),
        html.Div(
            id="size_div",
            children=[
                dcc.Dropdown(
                    id="size",
                    options=["New York City", "Montreal", "San Francisco"],
                    value="Montreal",
                ),
                dcc.Slider(id="size_markers", value=10, min=0, max=20, step=5),
                html.Div(
                    [
                        dcc.Checklist(id="size_log", options=["Log10"], value=[]),
                        dcc.Input(id="size_thresh", type="number", value=0.1),
                    ],
                    style={"width": "100%"},
                ),
                html.Div(
                    [
                        dcc.Input(id="size_min", type="number", value=-17),
                        dcc.Input(id="size_max", type="number", value=25.5),
                    ],
                    style={"width": "100%"},
                ),
            ],
            style={"display": "block"},
        ),
        html.Div(
            [
                dcc.Graph(
                    id="plot",
                ),
            ],
            style={"width": "100%", "display": "inline-block"},
        ),
    ],
    style={"width": "1000px"},
)


# https://stackoverflow.com/questions/50213761/changing-visibility-of-a-dash-component-by-updating-other-component
@app.callback(
    Output(component_id="x_div", component_property="style"),
    Output(component_id="y_div", component_property="style"),
    Output(component_id="z_div", component_property="style"),
    Output(component_id="color_div", component_property="style"),
    Output(component_id="size_div", component_property="style"),
    Input(component_id="axes_pannels", component_property="value"),
)
def update_visibility(axis):
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


@app.callback(
    Output(component_id="x_min", component_property="value"),
    Output(component_id="x_max", component_property="value"),
    Input(component_id="x", component_property="value"),
)
def set_channel_bounds(x):
    scatter.refresh.value = False
    cmin, cmax = ScatterPlots.get_channel_bounds("x")
    scatter.refresh.value = True
    return cmin, cmax


@app.callback(
    Output(component_id="y_min", component_property="value"),
    Output(component_id="y_max", component_property="value"),
    Input(component_id="y", component_property="value"),
)
def set_channel_bounds(y):
    scatter.refresh.value = False
    cmin, cmax = ScatterPlots.get_channel_bounds("y")
    scatter.refresh.value = True
    return cmin, cmax


@app.callback(
    Output(component_id="z_min", component_property="value"),
    Output(component_id="z_max", component_property="value"),
    Input(component_id="z", component_property="value"),
)
def set_channel_bounds(z):
    scatter.refresh.value = False
    cmin, cmax = ScatterPlots.get_channel_bounds("z")
    scatter.refresh.value = True
    return cmin, cmax


@app.callback(
    Output(component_id="color_min", component_property="value"),
    Output(component_id="color_max", component_property="value"),
    Input(component_id="color", component_property="value"),
)
def set_channel_bounds(color):
    scatter.refresh.value = False
    cmin, cmax = ScatterPlots.get_channel_bounds("color")
    scatter.refresh.value = True
    return cmin, cmax


@app.callback(
    Output(component_id="size_min", component_property="value"),
    Output(component_id="size_max", component_property="value"),
    Input(component_id="size", component_property="value"),
)
def set_channel_bounds(size):
    scatter.refresh.value = False
    cmin, cmax = ScatterPlots.get_channel_bounds("size")
    scatter.refresh.value = True
    return cmin, cmax


@app.callback(
    Output(component_id="plot", component_property="figure"),
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
)
def plot_selection(
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
):
    if not scatter.refresh.value:
        return None

    new_params_dict = {}
    for key, value in scatter.params.to_dict().items():
        # param = getattr(scatter, key, None)
        param = locals()[key]

        if param is None:
            new_params_dict[key] = value
        elif hasattr(param, "value") is False:
            new_params_dict[key] = param
        elif (
            (key == "x")
            | (key == "y")
            | (key == "z")
            | (key == "color")
            | (key == "size")
        ):
            if (param.value != "None") & (param.value in scatter.data_channels.keys()):
                new_params_dict[key] = scatter.data_channels[param.value]
            else:
                new_params_dict[key] = None
        else:
            new_params_dict[key] = param.value

    ifile = InputFile(
        ui_json=scatter.params.input_file.ui_json,
        validation_options={"disabled": True},
    )

    ifile.data = new_params_dict
    new_params = ScatterPlotParams(input_file=ifile)

    driver = ScatterPlotDriver(new_params)
    figure = go.FigureWidget(driver.run())

    return figure


def main():
    # The reloader has not yet run - open the browser
    if not environ.get("WERKZEUG_RUN_MAIN"):
        webbrowser.open_new("http://127.0.0.1:8050/")

    # Otherwise, continue as normal
    app.run_server(host="127.0.0.1", port=8050, debug=True)


if __name__ == "__main__":
    main()
