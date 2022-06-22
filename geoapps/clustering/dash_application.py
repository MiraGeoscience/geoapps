#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import os
import sys
import time
import webbrowser
from os import environ, makedirs, path

import dash_daq as daq
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import callback_context, dash_table, dcc, html, no_update
from dash.dependencies import Input, Output
from flask import Flask
from geoh5py.ui_json import InputFile
from geoh5py.workspace import Workspace
from jupyter_dash import JupyterDash
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

from geoapps.clustering.constants import app_initializer
from geoapps.clustering.params import ClusteringParams
from geoapps.scatter_plot.application import ScatterPlots
from geoapps.shared_utils.utils import colors, hex_to_rgb
from geoapps.utils.statistics import random_sampling


class Clustering(ScatterPlots):
    _param_class = ClusteringParams

    def __init__(self, ui_json=None, **kwargs):
        app_initializer.update(kwargs)
        if ui_json is not None and os.path.exists(ui_json.path):
            self.params = self._param_class(ui_json)
        else:
            self.params = self._param_class(**app_initializer)

        self.clusters = {}
        self.data_channels = {}
        self.kmeans = None
        self.scalings = {}
        self.lower_bounds = {}
        self.upper_bounds = {}
        self.indices = []
        self.mapping = None
        self.color_pickers = colors
        self.live_link = False
        # Initial values for the dash components
        super().__init__(**self.params.to_dict())
        self.defaults = self.get_cluster_defaults()

        external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
        server = Flask(__name__)
        self.app = JupyterDash(
            server=server,
            url_base_pathname=environ.get("JUPYTERHUB_SERVICE_PREFIX", "/"),
            external_stylesheets=external_stylesheets,
        )

        self.norm_tabs_layout = html.Div(
            id="norm_tabs",
            children=[
                dcc.Tabs(
                    [
                        dcc.Tab(
                            label="Histogram",
                            children=[
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                dcc.Markdown("Data: "),
                                                dcc.Dropdown(
                                                    id="channel",
                                                    value=self.defaults["channel"].name,
                                                    options=[
                                                        self.defaults["x"].name,
                                                        self.defaults["y"].name,
                                                        self.defaults["z"].name,
                                                        self.defaults["color"].name,
                                                        self.defaults["size"].name,
                                                    ],
                                                ),
                                                dcc.Markdown("Scale: "),
                                                dcc.Slider(
                                                    id="scale",
                                                    min=1,
                                                    max=10,
                                                    step=1,
                                                    value=self.defaults["scale"],
                                                    marks=None,
                                                    tooltip={
                                                        "placement": "bottom",
                                                        "always_visible": True,
                                                    },
                                                ),
                                                dcc.Markdown("Lower bound: "),
                                                dcc.Input(
                                                    id="lower_bounds",
                                                    value=self.defaults["lower_bounds"],
                                                ),
                                                dcc.Markdown("Upper bound: "),
                                                dcc.Input(
                                                    id="upper_bounds",
                                                    value=self.defaults["upper_bounds"],
                                                ),
                                            ],
                                            style={
                                                "width": "200px",
                                                "display": "inline-block",
                                                "vertical-align": "middle",
                                                "margin-right": "50px",
                                            },
                                        ),
                                        dcc.Graph(
                                            id="histogram",
                                            style={
                                                "width": "70%",
                                                "display": "inline-block",
                                                "vertical-align": "middle",
                                            },
                                        ),
                                    ]
                                )
                            ],
                        ),
                        dcc.Tab(
                            label="Statistics",
                            children=[
                                html.Div(
                                    [
                                        dash_table.DataTable(
                                            id="stats_table",
                                            style_data={
                                                "color": "black",
                                                "backgroundColor": "white",
                                            },
                                            style_data_conditional=[
                                                {
                                                    "if": {"row_index": "odd"},
                                                    "backgroundColor": "rgb(220, 220, 220)",
                                                }
                                            ],
                                            style_header={
                                                "backgroundColor": "rgb(210, 210, 210)",
                                                "color": "black",
                                                "fontWeight": "bold",
                                            },
                                        )
                                    ],
                                    style={"margin-top": "20px"},
                                )
                            ],
                        ),
                        dcc.Tab(
                            label="Confusion Matrix",
                            children=[
                                html.Div(
                                    [
                                        dcc.Graph(
                                            id="matrix",
                                        ),
                                    ],
                                    style={"width": "50%", "margin": "auto"},
                                )
                            ],
                        ),
                    ]
                )
            ],
        )

        self.cluster_tabs_layout = html.Div(
            [
                dcc.Tabs(
                    [
                        dcc.Tab(
                            label="Crossplot",
                            children=[
                                html.Div(
                                    [self.axis_layout],
                                    style={
                                        "width": "45%",
                                        "display": "inline-block",
                                        "vertical-align": "middle",
                                    },
                                ),
                                html.Div(
                                    [
                                        self.plot_layout,
                                    ],
                                    style={
                                        "width": "55%",
                                        "display": "inline-block",
                                        "vertical-align": "middle",
                                    },
                                ),
                            ],
                        ),
                        dcc.Tab(
                            label="Boxplot",
                            children=[
                                html.Div(
                                    [
                                        dcc.Graph(
                                            id="boxplot",
                                        )
                                    ],
                                    style={"width": "50%", "margin": "auto"},
                                )
                            ],
                        ),
                        dcc.Tab(
                            label="Inertia",
                            children=[
                                html.Div(
                                    [
                                        dcc.Graph(
                                            id="inertia",
                                        )
                                    ],
                                    style={"width": "50%", "margin": "auto"},
                                )
                            ],
                        ),
                    ]
                )
            ]
        )

        self.app.layout = html.Div(
            [
                html.Div(
                    [
                        self.workspace_layout,
                        dcc.Markdown("Data subset: "),
                        dcc.Dropdown(
                            id="channels",
                            value=[
                                self.defaults["x"].name,
                                self.defaults["y"].name,
                                self.defaults["z"].name,
                                self.defaults["color"].name,
                                self.defaults["size"].name,
                            ],
                            options=self.defaults["data_options"],
                            multi=True,
                        ),
                    ],
                    style={
                        "width": "40%",
                        "display": "inline-block",
                        "vertical-align": "top",
                        "margin-right": "50px",
                        "margin-bottom": "20px",
                    },
                ),
                html.Div(
                    [
                        dcc.Markdown("Number of clusters: "),
                        dcc.Slider(
                            id="n_clusters",
                            min=2,
                            max=100,
                            step=1,
                            value=self.defaults["n_clusters"],
                            marks=None,
                            tooltip={
                                "placement": "bottom",
                                "always_visible": True,
                            },
                        ),
                        dcc.Checklist(
                            id="show_color_picker",
                            options=["Select cluster color"],
                            value=[],
                        ),
                        dcc.Input(id="ga_group", value="test"),
                        html.Button("Export", id="export"),
                        dcc.Markdown(id="export_message"),
                    ],
                    style={
                        "width": "25%",
                        "display": "inline-block",
                        "vertical-align": "top",
                    },
                ),
                html.Div(
                    id="color_select_div",
                    children=[
                        dcc.Markdown("Cluster: "),
                        dcc.Dropdown(
                            id="select_cluster",
                            options=np.arange(0, 101, 1),
                            value=0,
                            style={"margin-bottom": "20px"},
                        ),
                        daq.ColorPicker(
                            id="color_picker",
                            value=dict(hex="#000000"),
                        ),
                    ],
                    style={
                        "width": "25%",
                        "display": "inline-block",
                        "vertical-align": "top",
                    },
                ),
                dcc.Checklist(
                    id="show_norm_tabs",
                    options=["Data Normalization"],
                    value=[],
                ),
                self.norm_tabs_layout,
                self.cluster_tabs_layout,
                dcc.Store(id="dataframe", data={}),
            ],
            style={"width": "70%", "margin-left": "50px", "margin-top": "30px"},
        )

        self.app.callback(
            Output(component_id="x_div", component_property="style"),
            Output(component_id="y_div", component_property="style"),
            Output(component_id="z_div", component_property="style"),
            Output(component_id="color_div", component_property="style"),
            Output(component_id="size_div", component_property="style"),
            Input(component_id="axes_pannels", component_property="value"),
        )(self.update_visibility)
        self.app.callback(
            Output(component_id="color_select_div", component_property="style"),
            Input(component_id="show_color_picker", component_property="value"),
        )(self.update_color_select)
        self.app.callback(
            Output(component_id="norm_tabs", component_property="style"),
            Input(component_id="show_norm_tabs", component_property="value"),
        )(self.update_norm_tabs)

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
            Output(component_id="channel", component_property="options"),
            Output(component_id="scale", component_property="value"),
            Output(component_id="lower_bounds", component_property="value"),
            Output(component_id="upper_bounds", component_property="value"),
            Output(component_id="color_picker", component_property="value"),
            Output(component_id="dataframe", component_property="data"),
            Input(component_id="upload", component_property="filename"),
            Input(component_id="upload", component_property="contents"),
            Input(component_id="objects", component_property="value"),
            Input(component_id="x", component_property="value"),
            Input(component_id="y", component_property="value"),
            Input(component_id="z", component_property="value"),
            Input(component_id="color", component_property="value"),
            Input(component_id="size", component_property="value"),
            Input(component_id="channel", component_property="value"),
            Input(component_id="channels", component_property="value"),
            Input(component_id="scale", component_property="value"),
            Input(component_id="lower_bounds", component_property="value"),
            Input(component_id="upper_bounds", component_property="value"),
            Input(component_id="downsampling", component_property="value"),
            Input(component_id="select_cluster", component_property="value"),
            # prevent_initial_call=True,
        )(self.update_cluster_params)
        self.app.callback(
            Output(component_id="crossplot", component_property="figure"),
            Output(component_id="stats_table", component_property="data"),
            Output(component_id="matrix", component_property="figure"),
            Output(component_id="histogram", component_property="figure"),
            Output(component_id="boxplot", component_property="figure"),
            Output(component_id="inertia", component_property="figure"),
            Input(component_id="n_clusters", component_property="value"),
            Input(component_id="select_cluster", component_property="value"),
            Input(component_id="dataframe", component_property="data"),
            Input(component_id="channel", component_property="value"),
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
            Input(component_id="color_picker", component_property="value"),
            Input(component_id="size", component_property="value"),
            Input(component_id="size_log", component_property="value"),
            Input(component_id="size_thresh", component_property="value"),
            Input(component_id="size_min", component_property="value"),
            Input(component_id="size_max", component_property="value"),
            Input(component_id="size_markers", component_property="value"),
        )(self.update_plots)
        self.app.callback(
            Output(component_id="export_message", component_property="children"),
            Input(component_id="export", component_property="n_clicks"),
            Input(component_id="dataframe", component_property="data"),
            Input(component_id="objects", component_property="value"),
            Input(component_id="n_clusters", component_property="value"),
            Input(component_id="ga_group", component_property="value"),
            prevent_initial_call=True,
        )(self.export_clusters)

    def get_cluster_defaults(self):
        defaults = {}
        # Get initial values to initialize the dash components

        for key, value in self.params.to_dict().items():
            if key == "data":
                if value is None:
                    defaults[key] = []
                else:
                    defaults[key] = [value.name]
            elif key == "objects":
                if value is None:
                    defaults["data_options"] = []
                else:
                    data_options = value.get_data_list()
                    if "Visual Parameters" in data_options:
                        data_options.remove("Visual Parameters")
                    defaults["data_options"] = data_options
                for channel in defaults["data_options"]:
                    self.get_channel(channel)
            else:
                defaults[key] = value
        self.scalings[defaults["channel"].name] = defaults["scale"]
        self.lower_bounds[defaults["channel"].name] = defaults["lower_bounds"]
        self.upper_bounds[defaults["channel"].name] = defaults["upper_bounds"]
        return defaults

    def update_color_select(self, checkbox):
        if not checkbox:
            return {"display": "none"}
        else:
            return {"width": "25%", "display": "inline-block", "vertical-align": "top"}

    def update_norm_tabs(self, checkbox):
        if not checkbox:
            return {"display": "none"}
        else:
            return {"display": "block"}

    def get_data_channels(self, channels):
        data_channels = {}

        for channel in channels:
            if channel not in data_channels.keys():
                if channel == "None":
                    data_channels[channel] = None
                elif self.params.geoh5.get_entity(channel):
                    data_channels[channel] = self.params.geoh5.get_entity(channel)[0]

        return data_channels

    def update_channels(self, channels):
        self.data_channels = self.get_data_channels(channels)

        for channel in channels:
            self.update_properties(channel)

        for channel in self.scalings.keys():
            if channel not in channels:
                del self.scalings[channel]
        for channel in self.lower_bounds.keys():
            if channel not in channels:
                del self.lower_bounds[channel]
        for channel in self.upper_bounds.keys():
            if channel not in channels:
                del self.upper_bounds[channel]

        return channels, channels, channels, channels, channels, channels

    def update_properties(self, channel):
        if channel not in self.scalings.keys():
            self.scalings[channel] = 1
        scale = self.scalings[channel]

        if channel not in self.lower_bounds.keys():
            self.lower_bounds[channel] = np.nanmin(self.data_channels[channel].values)
        lower_bounds = self.lower_bounds[channel]

        if channel not in self.upper_bounds.keys():
            self.upper_bounds[channel] = np.nanmax(self.data_channels[channel].values)
        upper_bounds = self.upper_bounds[channel]

        return scale, lower_bounds, upper_bounds

    def update_cluster_params(
        self,
        filename,
        contents,
        objects,
        x,
        y,
        z,
        color,
        size,
        channel,
        channels,
        scale,
        lower_bounds,
        upper_bounds,
        downsampling,
        select_cluster,
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
            "channels_options",
            "scale",
            "lower_bounds",
            "upper_bounds",
            "color_picker",
            "dataframe",
        ]

        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]
        update_dict = {}
        if trigger == "upload":
            if filename.endswith(".ui.json"):
                # Update params from uploaded uijson
                update_dict = self.update_from_uijson(contents)
            elif filename.endswith(".geoh5"):
                # Update object and data options from uploaded workspace
                update_dict = self.update_object_options(contents)
                # update data subset ***
                update_dict.update(
                    {
                        "channels_options": self.update_data_options(objects)[
                            "data_options"
                        ]
                    }
                )
            else:
                print("Uploaded file must be a workspace or ui.json.")
            update_dict["filename"] = None
            update_dict["contents"] = None

        elif trigger == "objects":
            # Update data subset options from object change
            update_dict = {
                "channels_options": self.update_data_options(objects)["data_options"]
            }

        elif trigger == "select_cluster":
            # Update color displayed by the dash colorpicker
            update_dict = self.update_color_picker(select_cluster)
            # Output(component_id="color_picker", component_property="value"),

        elif trigger in ["x", "y", "z", "color", "size"]:
            # Update min, max values in scatter plot
            update_dict = self.set_channel_bounds(x, y, z, color, size)

        else:
            print("test")
            # Update dataframe
            update_dict = {
                "dataframe": self.update_dataframe(
                    downsampling, channel, channels, scale, lower_bounds, upper_bounds
                )
            }
            if trigger == "channel":
                # Update displayed scale and bounds from stored values
                update_dict.update(self.update_properties(channel))
            elif trigger == "channels":
                # Update data options from data subset
                update_dict.update(self.update_channels(channels))

        outputs = []
        for param in param_list:
            if param in update_dict.keys():
                outputs.append(update_dict[param])
            else:
                outputs.append(no_update)

        return tuple(outputs)

    def update_plots(
        self,
        n_clusters,
        select_cluster,
        dataframe_dict,
        channel,
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
        color_picker,
        size,
        size_log,
        size_thresh,
        size_min,
        size_max,
        size_markers,
    ):
        if dataframe_dict:
            dataframe = pd.DataFrame(dataframe_dict["dataframe"])
            self.run_clustering(n_clusters, dataframe)

            color_maps = self.update_colormap(n_clusters, color_picker, select_cluster)

            crossplot = self.update_plot(
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
            )
            stats_table = self.make_stats_table(dataframe).to_dict("records")
            matrix = self.make_heatmap(dataframe)
            histogram = self.make_hist_plot(dataframe, channel)
            boxplot = self.make_boxplot(n_clusters, dataframe, channel)
            inertia = self.make_inertia_plot(n_clusters)

            return crossplot, stats_table, matrix, histogram, boxplot, inertia

        else:
            return None, None, None, None, None, None

    def update_color_picker(self, select_cluster):
        return dict(hex=self.color_pickers[select_cluster])

    def update_colormap(self, n_clusters, new_color, select_cluster):
        """
        Change the colormap for clusters
        """
        self.color_pickers[select_cluster] = new_color["hex"]
        colormap = {}
        for ii in range(n_clusters):
            colorpicker = self.color_pickers[ii]
            if "#" in colorpicker:
                color = colorpicker.lstrip("#")
                colormap[ii] = [
                    np.min([ii / (n_clusters - 1), 1]),
                    "rgb("
                    + ",".join([f"{int(color[i:i + 2], 16)}" for i in (0, 2, 4)])
                    + ")",
                ]
            else:
                colormap[ii] = [
                    np.min([ii / (n_clusters - 1), 1]),
                    colorpicker,
                ]

        # self.custom_colormap = list(self.colormap.values())
        return list(colormap.values())

    def run_clustering(self, n_clusters, dataframe):
        """
        Normalize the the selected data and perform the kmeans clustering.
        """
        if dataframe is None:
            return
        # Prime the app with clusters
        # Normalize values and run
        values = []
        for field in dataframe.columns:
            vals = dataframe[field].values.copy()

            nns = ~np.isnan(vals)
            vals[nns] = (
                (vals[nns] - min(vals[nns]))
                / (max(vals[nns]) - min(vals[nns]) + 1e-32)
                * self.scalings[field]
            )
            values += [vals]

        for val in [2, 4, 8, 16, 32, n_clusters]:
            kmeans = KMeans(n_clusters=val, random_state=0).fit(np.vstack(values).T)
            self.clusters[val] = kmeans

        cluster_ids = self.clusters[n_clusters].labels_.astype(float)
        # self.data_channels["kmeans"] = cluster_ids[self.mapping]
        self.kmeans = cluster_ids[self.mapping]

        """
        self.update_axes(refresh_plot=False)
        self.color_max.value = self.n_clusters.value
        self.update_colormap(None, refresh_plot=False)
        self.color.value = "kmeans"
        self.color_active.value = True
        """

    def make_inertia_plot(self, n_clusters):
        """
        Generate an inertia plot
        """
        if n_clusters in self.clusters.keys():
            ind = np.sort(list(self.clusters.keys()))
            inertias = [self.clusters[ii].inertia_ for ii in ind]
            clusters = ind
            line = go.Scatter(x=clusters, y=inertias, mode="lines")
            point = go.Scatter(
                x=[n_clusters],
                y=[self.clusters[n_clusters].inertia_],
            )

            inertia_plot = go.Figure([line, point])

            inertia_plot.update_layout(
                {
                    "xaxis": {"title": "Number of clusters"},
                    "showlegend": False,
                }
            )
            return inertia_plot
        else:
            return None

    def make_hist_plot(self, dataframe, channel):
        """
        Generate an histogram plot for the selected data channel.
        """
        if dataframe is not None:
            histogram = go.Figure(
                data=[
                    go.Histogram(
                        x=dataframe[channel].values,
                        histnorm="percent",
                        name=channel,
                    )
                ]
            )
            return histogram
        else:
            return None

    def make_boxplot(self, n_clusters, dataframe, channel):
        """
        Generate a box plot for each cluster.
        """
        if dataframe is not None and self.kmeans is not None:
            field = channel

            boxes = []
            for ii in range(n_clusters):

                cluster_ind = self.kmeans[self.indices] == ii
                x = np.ones(np.sum(cluster_ind)) * ii
                y = self.data_channels[field].values[self.indices][cluster_ind]

                boxes.append(
                    go.Box(
                        x=x,
                        y=y,
                        fillcolor=self.color_pickers[ii],
                        marker_color=self.color_pickers[ii],
                        line_color=self.color_pickers[ii],
                        showlegend=False,
                    )
                )

            boxplot = go.FigureWidget()

            boxplot.data = []
            for box in boxes:
                boxplot.add_trace(box)

            boxplot.update_layout(
                {
                    "xaxis": {"title": "Cluster #"},
                    "yaxis": {"title": field},
                    "height": 600,
                    "width": 600,
                }
            )
            return boxplot
        else:
            return None

    def make_stats_table(self, dataframe):
        """
        Generate a table of statistics using pandas
        """
        if dataframe is not None:
            stats_df = dataframe.describe(percentiles=None, include=None, exclude=None)
            stats_df.insert(0, "", stats_df.index)
            return stats_df
        else:
            return None

    def make_heatmap(self, dataframe):
        """
        Generate a confusion matrix
        """
        if dataframe is not None:
            df = dataframe.copy()
            corrs = df.corr()

            matrix = go.Figure(
                data=[
                    go.Heatmap(
                        x=list(corrs.columns),
                        y=list(corrs.index),
                        z=corrs.values,
                        type="heatmap",
                        colorscale="Viridis",
                        zsmooth=False,
                    )
                ]
            )

            matrix.update_scenes(aspectratio=dict(x=1, y=1, z=0.7), aspectmode="manual")
            matrix.update_layout(
                width=500,
                height=500,
                autosize=False,
                margin=dict(t=0, b=0, l=0, r=0),
                template="plotly_white",
                updatemenus=[
                    {
                        "buttons": [
                            {
                                "args": ["type", "heatmap"],
                                "label": "Heatmap",
                                "method": "restyle",
                            },
                            {
                                "args": ["type", "surface"],
                                "label": "3D Surface",
                                "method": "restyle",
                            },
                        ],
                        "direction": "down",
                        "pad": {"r": 10, "t": 10},
                        "showactive": True,
                        "x": 0.01,
                        "xanchor": "left",
                        "y": 1.15,
                        "yanchor": "top",
                    },
                    {
                        "buttons": [
                            {
                                "args": ["colorscale", label],
                                "label": label,
                                "method": "restyle",
                            }
                            for label in [
                                "Viridis",
                                "Rainbow",
                                "Cividis",
                                "Blues",
                                "Greens",
                            ]
                        ],
                        "direction": "down",
                        "pad": {"r": 10, "t": 10},
                        "showactive": True,
                        "x": 0.32,
                        "xanchor": "left",
                        "y": 1.15,
                        "yanchor": "top",
                    },
                ],
                yaxis={"autorange": "reversed"},
            )
            return matrix
        else:
            return None

    def update_dataframe(
        self, downsampling, channel, channels, scale, lower_bounds, upper_bounds
    ):
        """
        Normalize the the selected data and perform the kmeans clustering.
        """
        self.kmeans = None

        self.scalings[channel] = scale
        self.lower_bounds[channel] = lower_bounds
        self.upper_bounds[channel] = upper_bounds

        indices, values = self.get_indices(channels, downsampling)
        n_values = values.shape[0]

        dataframe = pd.DataFrame(
            values[indices, :],
            columns=channels,
        )

        tree = cKDTree(dataframe.values)
        inactive_set = np.ones(n_values, dtype="bool")
        inactive_set[indices] = False
        out_values = values[inactive_set, :]
        for ii in range(values.shape[1]):
            out_values[np.isnan(out_values[:, ii]), ii] = np.mean(values[indices, ii])

        _, ind_out = tree.query(out_values)
        del tree

        self.mapping = np.empty(n_values, dtype="int")
        self.mapping[inactive_set] = ind_out
        self.mapping[indices] = np.arange(len(indices))

        # self._inactive_set = np.where(np.all(np.isnan(values), axis=1))[0]
        # options = [[self.data.uid_name_map[key], key] for key in fields]
        # self.channels_plot_options.options = options
        return {"dataframe": dataframe.to_dict("records")}

    def get_indices(self, channels, downsampling):
        values = []
        non_nan = []
        for channel in channels:
            if channel is not None:
                values.append(
                    np.asarray(self.data_channels[channel].values, dtype=float)
                )
                non_nan.append(~np.isnan(self.data_channels[channel].values))

        values = np.vstack(values)
        non_nan = np.vstack(non_nan)

        percent = downsampling / 100

        # Number of values that are not nan along all three axes
        size = np.sum(np.all(non_nan, axis=0))

        indices = random_sampling(
            values.T,
            int(percent * size),
            bandwidth=2.0,
            rtol=1e0,
            method="histogram",
        )
        return indices, values.T

    def get_output_workspace(self, workpath: str = "./", name: str = "Temp.geoh5"):
        """
        Create an active workspace with check for GA monitoring directory
        """
        if not name.endswith(".geoh5"):
            name += ".geoh5"

        workspace = Workspace(path.join(workpath, name))
        workspace.close()
        live_link = False
        time.sleep(1)
        # Check if GA digested the file already
        if not path.exists(workspace.h5file):
            workpath = path.join(workpath, ".working")
            if not path.exists(workpath):
                makedirs(workpath)
            workspace = Workspace(path.join(workpath, name))
            workspace.close()
            live_link = True
            if not self.live_link:
                print(
                    "ANALYST Pro active live link found. Switching to monitoring directory..."
                )
        elif self.live_link:
            print(
                "ANALYST Pro 'monitoring directory' inactive. Reverting to standalone mode..."
            )

        self.live_link = live_link

        workspace.open()
        return workspace

    def export_clusters(self, n_clicks, dataframe, objects, n_clusters, group_name):
        """
        Write cluster groups to the target geoh5 object.
        """
        if (
            self.kmeans is not None
            and callback_context.triggered[0]["prop_id"].split(".")[0] == "export"
        ):
            obj = self.params.objects  # ***

            # Create reference values and color_map
            group_map, color_map = {}, []
            cluster_values = self.kmeans + 1
            # cluster_values = self.data_channels["kmeans"] + 1
            # cluster_values[self._inactive_set] = 0
            for ii in range(n_clusters):
                colorpicker = self.color_pickers[ii]
                color = colorpicker.lstrip("#")
                group_map[ii + 1] = f"Cluster_{ii}"
                color_map += [[ii + 1] + hex_to_rgb(color) + [1]]

            color_map = np.core.records.fromarrays(
                np.vstack(color_map).T,
                names=["Value", "Red", "Green", "Blue", "Alpha"],
            )

            # Create reference values and color_map
            group_map, color_map = {}, []
            for ii in range(n_clusters):
                colorpicker = self.color_pickers[ii]
                color = colorpicker.lstrip("#")
                group_map[ii + 1] = f"Cluster_{ii}"
                color_map += [[ii + 1] + hex_to_rgb(color) + [1]]

            color_map = np.core.records.fromarrays(
                np.vstack(color_map).T,
                names=["Value", "Red", "Green", "Blue", "Alpha"],
            )

            if self.params.monitoring_directory:
                output_path = self.params.monitoring_directory
                # monitored_directory_copy(self.export_directory.selected_path, obj)
            else:
                output_path = os.path.dirname(self.params.geoh5.h5file)

            temp_geoh5 = f"Clustering_{time.time():.3f}.geoh5"
            with self.get_output_workspace(output_path, temp_geoh5) as workspace:
                obj = obj.copy(parent=workspace)
                cluster_groups = obj.add_data(
                    {
                        group_name: {
                            "type": "referenced",
                            "values": cluster_values,
                            "value_map": group_map,
                        }
                    }
                )
                cluster_groups.entity_type.color_map = {
                    "name": "Cluster Groups",
                    "values": color_map,
                }

            return "Saved to " + output_path + "/" + temp_geoh5

    def run(self):
        # The reloader has not yet run - open the browser
        if not environ.get("WERKZEUG_RUN_MAIN"):
            webbrowser.open_new("http://127.0.0.1:8050/")

        # Otherwise, continue as normal
        self.app.run_server(host="127.0.0.1", port=8050, debug=False)


app = Clustering()
app.app.run()


if __name__ == "__main__":
    print("Loading geoh5 file . . .")
    file = sys.argv[1]
    ifile = InputFile.read_ui_json(file)
    app = Clustering(ui_json=ifile)
    print("Loaded. Building the clustering app . . .")
    app.run()
    print("Done")
