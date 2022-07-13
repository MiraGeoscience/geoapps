#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import ast
import os
import sys
import time
import webbrowser
from multiprocessing import cpu_count
from os import environ, makedirs, path

from geoh5py.objects import ObjectBase

from geoapps.clustering.driver import ClusteringDriver

os.environ["OMP_NUM_THREADS"] = str(cpu_count())

import dash_daq as daq
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import callback_context, dash_table, dcc, html, no_update
from dash.dependencies import Input, Output
from flask import Flask
from geoh5py.ui_json import InputFile
from geoh5py.workspace import Workspace
from jupyter_dash import JupyterDash

from geoapps.clustering.constants import app_initializer
from geoapps.clustering.params import ClusteringParams
from geoapps.clustering.plot_data import PlotData
from geoapps.scatter_plot.application import ScatterPlots
from geoapps.shared_utils.utils import colors


class Clustering(ScatterPlots):
    _param_class = ClusteringParams

    def __init__(self, ui_json=None, **kwargs):
        app_initializer.update(kwargs)
        if ui_json is not None and os.path.exists(ui_json.path):
            self.params = self._param_class(ui_json)
        else:
            self.params = self._param_class(**app_initializer)

        self.data_channels = {}
        super().__init__(clustering=True, **self.params.to_dict())
        # Initial values for the dash components
        self.defaults.update(self.get_cluster_defaults())

        external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
        server = Flask(__name__)
        self.app = JupyterDash(
            server=server,
            url_base_pathname=environ.get("JUPYTERHUB_SERVICE_PREFIX", "/"),
            external_stylesheets=external_stylesheets,
        )

        # Layout for histogram, stats table, confusion matrix
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
                                                    value=self.defaults["channel"],
                                                    options=self.defaults["channels"],
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
                                    style={
                                        "margin-top": "20px",
                                        "margin-bottom": "20px",
                                    },
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

        # Layout for crossplot, boxplot, inertia plot
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

        # Full app layout
        self.app.layout = html.Div(
            [
                html.Div(
                    [
                        # Workspace, object, downsampling, data subset selection
                        self.workspace_layout,
                        dcc.Markdown("Data subset: "),
                        dcc.Dropdown(
                            id="channels",
                            value=self.defaults["channels"],
                            options=self.defaults["channels_options"],
                            multi=True,
                        ),
                    ],
                    style={
                        "width": "40%",
                        "display": "inline-block",
                        "vertical-align": "top",
                        "margin-right": "50px",
                        "margin-bottom": "45px",
                    },
                ),
                html.Div(
                    [
                        # Dash components for clustering parameters
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
                            style={"margin-bottom": "20px"},
                        ),
                        dcc.Checklist(
                            id="live_link",
                            options=["Geoscience ANALYST Pro - Live link"],
                            value=[],
                            style={"margin-bottom": "20px"},
                        ),
                        dcc.Markdown("Group: "),
                        dcc.Input(
                            id="ga_group",
                            value=self.defaults["ga_group_name"],
                            style={"margin-bottom": "20px"},
                        ),
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
                        "display": "none",
                        "vertical-align": "top",
                    },
                ),
                # Checkbox to hide/show the normalization plots
                dcc.Checklist(
                    id="show_norm_tabs",
                    options=["Show Analytics & Normalization"],
                    value=[],
                    style={"margin-bottom": "10px"},
                ),
                self.norm_tabs_layout,
                self.cluster_tabs_layout,
                # Stored variables
                dcc.Store(id="dataframe", data=self.defaults["dataframe"]),
                dcc.Store(id="full_scales", data=self.defaults["full_scales"]),
                dcc.Store(
                    id="full_lower_bounds", data=self.defaults["full_lower_bounds"]
                ),
                dcc.Store(
                    id="full_upper_bounds", data=self.defaults["full_upper_bounds"]
                ),
                dcc.Store(id="color_pickers", data=self.defaults["color_pickers"]),
                dcc.Store(id="plot_kmeans", data=self.defaults["plot_kmeans"]),
                dcc.Store(id="kmeans", data=self.defaults["kmeans"]),
                dcc.Store(id="clusters", data=self.defaults["clusters"]),
                dcc.Store(id="indices", data=self.defaults["indices"]),
                dcc.Store(id="mapping", data=self.defaults["mapping"]),
            ],
            style={"width": "70%", "margin-left": "50px", "margin-top": "30px"},
        )

        # Callbacks relating to layout
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
        )(Clustering.update_color_select)
        self.app.callback(
            Output(component_id="norm_tabs", component_property="style"),
            Input(component_id="show_norm_tabs", component_property="value"),
        )(Clustering.update_norm_tabs)

        # Callback to update any params
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
            Output(component_id="channel", component_property="value"),
            Output(component_id="channels", component_property="options"),
            Output(component_id="channels", component_property="value"),
            Output(component_id="scale", component_property="value"),
            Output(component_id="lower_bounds", component_property="value"),
            Output(component_id="upper_bounds", component_property="value"),
            Output(component_id="color_maps", component_property="options"),
            Output(component_id="color_picker", component_property="value"),
            Output(component_id="n_clusters", component_property="value"),
            Output(component_id="dataframe", component_property="data"),
            Output(component_id="full_scales", component_property="data"),
            Output(component_id="full_lower_bounds", component_property="data"),
            Output(component_id="full_upper_bounds", component_property="data"),
            Output(component_id="color_pickers", component_property="data"),
            Output(component_id="plot_kmeans", component_property="data"),
            Output(component_id="kmeans", component_property="data"),
            Output(component_id="clusters", component_property="data"),
            Output(component_id="indices", component_property="data"),
            Output(component_id="mapping", component_property="data"),
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
            Input(component_id="n_clusters", component_property="value"),
            Input(component_id="full_scales", component_property="data"),
            Input(component_id="full_lower_bounds", component_property="data"),
            Input(component_id="full_upper_bounds", component_property="data"),
            Input(component_id="color_picker", component_property="value"),
            Input(component_id="color_pickers", component_property="data"),
            Input(component_id="kmeans", component_property="data"),
            Input(component_id="indices", component_property="data"),
            Input(component_id="clusters", component_property="data"),
        )(self.update_cluster_params)
        # Callback to update all the plots
        self.app.callback(
            Output(component_id="crossplot", component_property="figure"),
            Output(component_id="stats_table", component_property="data"),
            Output(component_id="matrix", component_property="figure"),
            Output(component_id="histogram", component_property="figure"),
            Output(component_id="boxplot", component_property="figure"),
            Output(component_id="inertia", component_property="figure"),
            Input(component_id="n_clusters", component_property="value"),
            Input(component_id="dataframe", component_property="data"),
            Input(component_id="channel", component_property="value"),
            Input(component_id="lower_bounds", component_property="value"),
            Input(component_id="upper_bounds", component_property="value"),
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
            Input(component_id="color_pickers", component_property="data"),
            Input(component_id="size", component_property="value"),
            Input(component_id="size_log", component_property="value"),
            Input(component_id="size_thresh", component_property="value"),
            Input(component_id="size_min", component_property="value"),
            Input(component_id="size_max", component_property="value"),
            Input(component_id="size_markers", component_property="value"),
            Input(component_id="kmeans", component_property="data"),
            Input(component_id="indices", component_property="data"),
            Input(component_id="clusters", component_property="data"),
        )(self.update_plots)
        # Callback to export the clusters as a geoh5 file
        self.app.callback(
            Output(component_id="live_link", component_property="value"),
            Input(component_id="export", component_property="n_clicks"),
            prevent_initial_call=True,
        )(self.export_clusters)

    def get_cluster_defaults(self):
        # Get initial values to initialize the dash components
        defaults = {}
        defaults["kmeans"] = None
        defaults["clusters"] = {}
        defaults["mapping"] = None
        defaults["indices"] = None
        # If there is no default data subset list, set it from selected scatter plot data
        self.params.channels = ast.literal_eval(self.params.channels)
        if not self.params.channels:
            plot_data = [
                self.defaults["x_name"],
                self.defaults["y_name"],
                self.defaults["z_name"],
                self.defaults["color_name"],
                self.defaults["size_name"],
            ]
            self.params.channels = list(filter(None, plot_data))
        defaults["channels"] = self.params.channels

        # Set the initial histogram data to be the first data in the data subset
        if len(defaults["channels"]) > 0:
            self.params.channel = defaults["channels"][0]
        else:
            self.params.channel = None
        defaults["channel"] = self.params.channel

        # Loop through self.params to set defaults
        for key, value in self.params.to_dict().items():
            if (key != "channels") & (key != "channel"):
                if key == "objects":
                    # Get default data subset from self.params.objects
                    if value is None:
                        defaults["channels_options"] = []
                    else:
                        channels_options = value.get_data_list()
                        if "Visual Parameters" in channels_options:
                            channels_options.remove("Visual Parameters")
                        defaults["channels_options"] = channels_options
                    for channel in defaults["channels_options"]:
                        self.get_channel(channel)
                elif key in ["full_scales", "full_lower_bounds", "full_upper_bounds"]:
                    # Reconstruct scaling and bounds dicts from uijson input lists.
                    out_dict = {}
                    full_list = ast.literal_eval(getattr(self.params, key))
                    for i in range(len(defaults["channels"])):
                        if (full_list is None) | (not full_list):
                            if key == "full_scales":
                                out_dict[defaults["channels"][i]] = 1
                            else:
                                out_dict[defaults["channels"][i]] = None
                        else:
                            out_dict[defaults["channels"][i]] = full_list[i]
                    defaults[key] = out_dict
                elif key == "color_pickers":
                    full_list = ast.literal_eval(getattr(self.params, key))
                    if (full_list is None) | (not full_list):
                        defaults[key] = colors
                    else:
                        defaults[key] = full_list
                else:
                    defaults[key] = value

        channel = defaults["channel"]
        # Set up initial dataframe and clustering
        defaults.update(
            self.update_clustering(
                channel,
                defaults["channels"],
                defaults["full_scales"],
                defaults["full_lower_bounds"],
                defaults["full_upper_bounds"],
                defaults["downsampling"],
                defaults["n_clusters"],
                defaults["kmeans"],
                defaults["clusters"],
                defaults["indices"],
                defaults["geoh5"],
                True,
            )
        )
        # Get initial scale and bounds for histogram plot
        defaults["scale"], defaults["lower_bounds"], defaults["upper_bounds"] = (
            None,
            None,
            None,
        )
        if channel in defaults["full_scales"]:
            defaults["scale"] = defaults["full_scales"][channel]
        if channel in defaults["full_lower_bounds"]:
            defaults["lower_bounds"] = defaults["full_lower_bounds"][channel]
        if channel in defaults["full_upper_bounds"]:
            defaults["upper_bounds"] = defaults["full_upper_bounds"][channel]

        return defaults

    @staticmethod
    def update_color_select(checkbox):
        # Updating visibility for cluster color picker
        if not checkbox:
            return {"display": "none"}
        else:
            return {"width": "25%", "display": "inline-block", "vertical-align": "top"}

    @staticmethod
    def update_norm_tabs(checkbox):
        # Update visibility for normalization plots
        if not checkbox:
            return {"display": "none"}
        else:
            return {"display": "block"}

    def get_data_channels(self, channels):
        # Loop through channels and add them to the data channels dict with name and object of all the current data
        data_channels = {}
        for channel in channels:
            if channel not in data_channels.keys():
                if channel == "None":
                    data_channels[channel] = None
                elif self.params.geoh5.get_entity(channel):
                    data_channels[channel] = self.params.geoh5.get_entity(channel)[0]

        return data_channels

    def update_channels(
        self,
        channel,
        channels,
        full_scales,
        full_lower_bounds,
        full_upper_bounds,
        kmeans,
        indices,
    ):
        # Update the data options for the scatter plot and histogram from the data subset
        if channels is None:
            self.data_channels = {}
            return {
                "channel_options": [],
                "color_maps_options": px.colors.named_colorscales(),
                "data_options": {},
                "full_scales": {},
                "full_lower_bounds": {},
                "full_upper_bounds": {},
                "channel": None,
            }
        else:
            self.data_channels = self.get_data_channels(channels)
            channels = list(filter(None, channels))
            # Update the full scales and bounds dicts with the new data subset
            for chan in channels:
                dict = self.update_properties(
                    chan, full_scales, full_lower_bounds, full_upper_bounds
                )
                full_scales = dict["full_scales"]
                full_lower_bounds = dict["full_lower_bounds"]
                full_upper_bounds = dict["full_upper_bounds"]

            new_scales = {}
            for chan, value in full_scales.items():
                if chan in channels:
                    new_scales[chan] = value
            new_lower_bounds = {}
            for chan, value in full_lower_bounds.items():
                if chan in channels:
                    new_lower_bounds[chan] = value
            new_upper_bounds = {}
            for chan, value in full_upper_bounds.items():
                if chan in channels:
                    new_upper_bounds[chan] = value

            if channel not in channels:
                channel = None

            # Add kmeans to the data selection for the scatter plot
            if kmeans is not None:
                data_options = channels + ["kmeans"]
                color_maps_options = px.colors.named_colorscales() + ["kmeans"]
                self.data_channels.update(
                    {"kmeans": PlotData("kmeans", kmeans[np.array(indices)])}
                )
            else:
                data_options = channels
                color_maps_options = px.colors.named_colorscales()
                self.data_channels.pop("kmeans", None)

            return {
                "channel_options": channels,
                "color_maps_options": color_maps_options,
                "data_options": data_options,
                "full_scales": new_scales,
                "full_lower_bounds": new_lower_bounds,
                "full_upper_bounds": new_upper_bounds,
                "channel": channel,
            }

    def update_properties(
        self, channel, full_scales, full_lower_bounds, full_upper_bounds
    ):
        # Get stored scale and bounds for a given channel. If there's no stored value, set a default.
        if channel is not None:
            if channel not in full_scales.keys():
                full_scales[channel] = 1
            scale = full_scales[channel]

            if (channel not in full_lower_bounds.keys()) or (
                full_lower_bounds[channel] is None
            ):
                full_lower_bounds[channel] = np.nanmin(
                    self.data_channels[channel].values
                )
            lower_bounds = float(full_lower_bounds[channel])

            if (channel not in full_upper_bounds.keys()) or (
                full_upper_bounds[channel] is None
            ):
                full_upper_bounds[channel] = np.nanmax(
                    self.data_channels[channel].values
                )
            upper_bounds = float(full_upper_bounds[channel])
        else:
            scale, lower_bounds, upper_bounds = None, None, None

        return {
            "scale": scale,
            "lower_bounds": lower_bounds,
            "upper_bounds": upper_bounds,
            "full_scales": full_scales,
            "full_lower_bounds": full_lower_bounds,
            "full_upper_bounds": full_upper_bounds,
        }

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
        n_clusters,
        full_scales,
        full_lower_bounds,
        full_upper_bounds,
        color_picker,
        color_pickers,
        kmeans,
        indices,
        clusters,
    ):
        # List of params that will be outputted
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
            "channel_options",
            "channel",
            "channels_options",
            "channels",
            "scale",
            "lower_bounds",
            "upper_bounds",
            "color_maps_options",
            "color_picker",
            "n_clusters",
            "dataframe",
            "full_scales",
            "full_lower_bounds",
            "full_upper_bounds",
            "color_pickers",
            "plot_kmeans",
            "kmeans",
            "clusters",
            "indices",
            "mapping",
        ]
        # Trigger is which variable triggered the callback
        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]

        if full_scales is None:
            full_scales = {}
        if full_lower_bounds is None:
            full_lower_bounds = {}
        if full_upper_bounds is None:
            full_upper_bounds = {}

        indices = np.array(indices)
        kmeans = np.array(kmeans)

        update_dict = {}
        if trigger == "upload":
            if filename.endswith(".ui.json"):
                # Update params from uploaded uijson
                update_dict = self.update_from_uijson(contents)
                if "plot_kmeans" in update_dict:
                    plot_kmeans = ast.literal_eval(update_dict["plot_kmeans"])
                    update_dict.update({"plot_kmeans": plot_kmeans})
                    axis_list = ["x", "y", "z", "color", "size"]
                    for i in range(len(plot_kmeans)):
                        if plot_kmeans[i]:
                            update_dict.update({axis_list[i] + "_name": "kmeans"})
                if "color_pickers" in update_dict:
                    color_pickers = ast.literal_eval(update_dict["color_pickers"])
                    if color_pickers:
                        update_dict.update({"color_pickers": color_pickers})
                    else:
                        update_dict.update({"color_pickers": colors})
                # Update full scales, bounds
                if "channels" in update_dict:
                    channels = update_dict["channels"]
                    if not channels:
                        update_dict.update(
                            {
                                "full_scales": {},
                                "full_lower_bounds": {},
                                "full_upper_bounds": {},
                            }
                        )
                    else:
                        for key in [
                            "full_scales",
                            "full_lower_bounds",
                            "full_upper_bounds",
                        ]:
                            # Reconstruct full scales and bounds dict from lists in uijson.
                            if key in update_dict:
                                out_dict = {}
                                full_list = ast.literal_eval(update_dict[key])
                                if full_list:
                                    for i in range(len(channels)):
                                        out_dict[channels[i]] = full_list[i]
                                update_dict.update({key: out_dict})
                if "downsampling" in update_dict:
                    downsampling = update_dict["downsampling"]
                if "n_clusters" in update_dict:
                    n_clusters = update_dict["n_clusters"]
                # Create new dataframe and run clustering for new variables.
                self.params.geoh5 = Workspace(update_dict["geoh5"])
                update_dict.update(
                    self.update_clustering(
                        channel,
                        channels,
                        update_dict["full_scales"],
                        update_dict["full_lower_bounds"],
                        update_dict["full_upper_bounds"],
                        downsampling,
                        n_clusters,
                        kmeans,
                        clusters,
                        indices,
                        self.params.geoh5,
                        True,
                    )
                )

            elif filename.endswith(".geoh5"):
                # Update object and data subset options from uploaded workspace
                update_dict = self.update_object_options(contents)
                channels_options = self.update_data_options(
                    update_dict["objects_name"]
                )["data_options"]
                update_dict.update(
                    {
                        "channels_options": channels_options,
                        "channels": None,
                    }
                )
                update_dict.update(
                    self.update_channels(
                        None,
                        None,
                        full_scales,
                        full_lower_bounds,
                        full_upper_bounds,
                        kmeans,
                        indices,
                    )
                )
            else:
                print("Uploaded file must be a workspace or ui.json.")
            # Reset file properties so the same file can be uploaded twice in a row.
            update_dict["filename"] = None
            update_dict["contents"] = None
        elif trigger == "objects":
            # Update data subset options from object change
            update_dict = {
                "channels_options": self.update_data_options(objects)["data_options"]
            }
        elif trigger == "select_cluster":
            # Update color displayed by the dash colorpicker
            update_dict = Clustering.update_color_picker(select_cluster, color_pickers)
            # Output(component_id="color_picker", component_property="value"),
        elif trigger == "color_picker":
            # Update color_pickers with new color selection
            color_pickers[select_cluster] = color_picker["hex"]
            update_dict.update({"color_pickers": color_pickers})
        elif trigger in ["x", "y", "z", "color", "size"]:
            # Update min, max values in scatter plot
            update_dict = {
                "x_name": x,
                "y_name": y,
                "z_name": z,
                "color_name": color,
                "size_name": size,
            }
            update_dict.update(self.set_channel_bounds(x, y, z, color, size))
            update_dict.update(
                {"plot_kmeans": [i == "kmeans" for i in [x, y, z, color, size]]}
            )
        elif trigger in [
            "downsampling",
            "channel",
            "channels",
            "scale",
            "lower_bounds",
            "upper_bounds",
            "n_clusters",
            "",
        ]:
            update_dict.update({"channel_name": channel})

            if trigger in ["scale", "lower_bounds", "upper_bounds"]:
                full_scales[channel] = scale
                full_lower_bounds[channel] = lower_bounds
                full_upper_bounds[channel] = upper_bounds
                update_dict.update(
                    {
                        "full_scales": full_scales,
                        "full_lower_bounds": full_lower_bounds,
                        "full_upper_bounds": full_upper_bounds,
                    }
                )
            elif trigger in ["channels", "downsampling", "n_clusters", ""]:
                update_dict.update(
                    {
                        "channels": channels,
                        "downsampling": downsampling,
                        "n_clusters": n_clusters,
                    }
                )
                # Update data options from data subset
                update_all_clusters = trigger != "n_clusters"
                update_dict.update(
                    self.update_clustering(
                        channel,
                        channels,
                        full_scales,
                        full_lower_bounds,
                        full_upper_bounds,
                        downsampling,
                        n_clusters,
                        kmeans,
                        clusters,
                        indices,
                        self.params.geoh5,
                        update_all_clusters,
                    )
                )
            elif trigger == "channel":
                update_dict.update({"channel": channel})
                # Update displayed scale and bounds from stored values
                update_dict.update(
                    self.update_properties(
                        channel, full_scales, full_lower_bounds, full_upper_bounds
                    )
                )

        # Update param dict from update_dict
        self.update_param_dict(update_dict)

        outputs = []
        for param in param_list:
            if param in update_dict.keys():
                outputs.append(update_dict[param])
            else:
                outputs.append(no_update)

        return tuple(outputs)

    def update_param_dict(self, update_dict):
        if "plot_kmeans" in update_dict.keys():
            plot_kmeans = update_dict["plot_kmeans"]
        else:
            plot_kmeans = ast.literal_eval(self.params.plot_kmeans)
        if len(plot_kmeans) == 0:
            plot_kmeans = [False, False, False, False, False]
        axis_list = ["x", "y", "z", "color", "size"]
        # Update self.params from update_dict.
        for key, value in self.params.to_dict().items():
            if key in update_dict:
                if key in ["full_scales", "full_lower_bounds", "full_upper_bounds"]:
                    # Convert dict of scales and bounds to lists to store in uijson.
                    if "channels" in update_dict:
                        channels = update_dict["channels"]
                    else:
                        channels = self.params.channels
                    outlist = []
                    if bool(channels) & bool(update_dict[key]):
                        for channel in channels:
                            outlist.append(update_dict[key][channel])
                    setattr(self.params, key, str(outlist))
                elif key == "color_pickers":
                    setattr(self.params, key, str(update_dict[key]))
                elif key in ["x_log", "y_log", "z_log", "color_log", "size_log"]:
                    if value is None:
                        setattr(self.params, key, False)
                    else:
                        setattr(self.params, key, value)
                elif key == "plot_kmeans":
                    setattr(self.params, key, str(update_dict[key]))
                else:
                    setattr(self.params, key, update_dict[key])
            elif key in ["x", "y", "z", "color", "size"]:
                if key + "_name" in update_dict:
                    if update_dict[key + "_name"] in self.data_channels:
                        if (
                            self.data_channels[update_dict[key + "_name"]].name
                            == "kmeans"
                        ):
                            index = axis_list.index(key)
                            plot_kmeans[index] = True
                            setattr(self.params, key, None)
                        else:
                            setattr(
                                self.params,
                                key,
                                self.data_channels[update_dict[key + "_name"]],
                            )
            elif key == "objects":
                if "objects_name" in update_dict:
                    obj = self.params.geoh5.get_entity(update_dict["objects_name"])[0]
                    self.params.objects = obj

    def update_plots(
        self,
        n_clusters,
        dataframe_dict,
        channel,
        lower_bounds,
        upper_bounds,
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
        color_pickers,
        size,
        size_log,
        size_thresh,
        size_min,
        size_max,
        size_markers,
        kmeans,
        indices,
        clusters,
    ):
        # Read in stored dataframe.
        print("A")
        dataframe = pd.DataFrame(dataframe_dict)
        print("B")
        if not dataframe.empty:
            if color_maps == "kmeans":
                # Update color_maps
                color_maps = Clustering.update_colormap(n_clusters, color_pickers)
            elif color_maps is None:
                color_maps = [[0.0, "rgb(0,0,0)"]]
            print("C")
            # Input downsampled data to scatterplot so it doesn't regenerate data every time a parameter changes.
            axis_values = []
            for axis in [x, y, z, color, size]:
                if axis == "kmeans":
                    axis_values.append(self.data_channels["kmeans"])
                elif axis is not None:
                    axis_values.append(PlotData(axis, dataframe[axis].values))
                else:
                    axis_values.append(None)
            print("D")
            x, y, z, color, size = tuple(axis_values)

            crossplot = self.update_plot(
                100,
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
                clustering=True,
            )
            print("E")
            stats_table = self.make_stats_table(dataframe)
            print("F")
            matrix = self.make_heatmap(dataframe)
            print("G")
            histogram = self.make_hist_plot(
                dataframe, channel, lower_bounds, upper_bounds
            )
            boxplot = self.make_boxplot(
                n_clusters,
                dataframe,
                channel,
                color_pickers,
                np.array(kmeans),
                np.array(indices),
            )
            inertia = self.make_inertia_plot(n_clusters, clusters)
            return crossplot, stats_table, matrix, histogram, boxplot, inertia

        else:
            return go.Figure(), None, go.Figure(), go.Figure(), go.Figure(), go.Figure()

    @staticmethod
    def update_color_picker(select_cluster, color_pickers):
        return {"color_picker": dict(hex=color_pickers[select_cluster])}

    @staticmethod
    def update_colormap(n_clusters, color_pickers):
        """
        Change the colormap for clusters
        """
        cluster_map = {}
        for ii in range(n_clusters):
            colorpicker = color_pickers[ii]
            if "#" in colorpicker:
                color = colorpicker.lstrip("#")
                cluster_map[ii] = [
                    np.min([ii / (n_clusters - 1), 1]),
                    "rgb("
                    + ",".join([f"{int(color[i:i + 2], 16)}" for i in (0, 2, 4)])
                    + ")",
                ]
            else:
                cluster_map[ii] = [
                    np.min([ii / (n_clusters - 1), 1]),
                    colorpicker,
                ]

        return list(cluster_map.values())

    def update_clustering(
        self,
        channel,
        channels,
        full_scales,
        full_lower_bounds,
        full_upper_bounds,
        downsampling,
        n_clusters,
        kmeans,
        clusters,
        indices,
        workspace,
        update_all_clusters,
    ):
        # Update dataframe, data options for plots, and run clustering
        update_dict = self.update_channels(
            channel,
            channels,
            full_scales,
            full_lower_bounds,
            full_upper_bounds,
            kmeans,
            indices,
        )
        update_dict.update(
            ClusteringDriver.update_dataframe(downsampling, channels, workspace)
        )
        update_dict.update(
            ClusteringDriver.run_clustering(
                n_clusters,
                update_dict["dataframe"],
                update_dict["full_scales"],
                clusters,
                update_dict["mapping"],
                update_all_clusters,
            )
        )
        update_dict.update(
            self.update_channels(
                channel,
                channels,
                update_dict["full_scales"],
                update_dict["full_lower_bounds"],
                update_dict["full_upper_bounds"],
                update_dict["kmeans"],
                update_dict["indices"],
            )
        )
        return update_dict

    @staticmethod
    def make_inertia_plot(n_clusters, clusters):
        """
        Generate an inertia plot
        """
        if n_clusters in clusters.keys():
            ind = np.sort(list(clusters.keys()))
            inertias = [clusters[ii]["inertia"] for ii in ind]
            clusters = ind
            line = go.Scatter(x=clusters, y=inertias, mode="lines")
            point = go.Scatter(
                x=[n_clusters],
                y=[clusters[n_clusters].inertia_],
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

    @staticmethod
    def make_hist_plot(dataframe, channel, lower_bounds, upper_bounds):
        """
        Generate an histogram plot for the selected data channel.
        """
        if channel is not None:
            histogram = go.Figure(
                data=[
                    go.Histogram(
                        x=dataframe[channel].values,
                        histnorm="percent",
                        name=channel,
                    )
                ]
            )
            histogram.update_xaxes(range=[lower_bounds, upper_bounds])
            return histogram
        else:
            return None

    def make_boxplot(
        self, n_clusters, dataframe, channel, color_pickers, kmeans, indices
    ):
        """
        Generate a box plot for each cluster.
        """
        if (kmeans is not None) and (channel is not None):
            field = channel

            boxes = []
            for ii in range(n_clusters):

                cluster_ind = kmeans[indices] == ii
                x = np.ones(np.sum(cluster_ind)) * ii
                y = self.data_channels[field].values[indices][cluster_ind]

                boxes.append(
                    go.Box(
                        x=x,
                        y=y,
                        fillcolor=color_pickers[ii],
                        marker_color=color_pickers[ii],
                        line_color=color_pickers[ii],
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

    @staticmethod
    def make_stats_table(dataframe):
        """
        Generate a table of statistics using pandas
        """
        stats_df = dataframe.describe(percentiles=None, include=None, exclude=None)
        stats_df.insert(0, "", stats_df.index)
        return stats_df.to_dict("records")

    @staticmethod
    def make_heatmap(dataframe):
        """
        Generate a confusion matrix
        """
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

    @staticmethod
    def get_output_workspace(live_link, workpath: str = "./", name: str = "Temp.geoh5"):
        """
        Create an active workspace with check for GA monitoring directory
        """
        if not name.endswith(".geoh5"):
            name += ".geoh5"

        workspace = Workspace(path.join(workpath, name))
        workspace.close()
        new_live_link = False
        time.sleep(1)
        # Check if GA digested the file already
        if not path.exists(workspace.h5file):
            workpath = path.join(workpath, ".working")
            if not path.exists(workpath):
                makedirs(workpath)
            workspace = Workspace(path.join(workpath, name))
            workspace.close()
            new_live_link = True
            if not live_link:
                print(
                    "ANALYST Pro active live link found. Switching to monitoring directory..."
                )
        elif live_link:
            print(
                "ANALYST Pro 'monitoring directory' inactive. Reverting to standalone mode..."
            )

        workspace.open()
        # return new live link
        return workspace, new_live_link

    def export_clusters(self, n_clicks):
        """
        Write cluster groups to the target geoh5 object.
        """
        param_dict = self.params.to_dict()
        temp_geoh5 = f"Clustering_{time.time():.0f}.geoh5"

        # Get output path.
        if self.params.live_link:
            if self.params.monitoring_directory is not None and os.path.exists(
                os.path.abspath(self.params.monitoring_directory)
            ):
                output_path = self.params.monitoring_directory
            else:
                print("Invalid monitoring directory path")
                return []
        else:
            output_path = os.path.dirname(self.params.geoh5.h5file)

        # Get output workspace.
        ws, self.params.live_link = Clustering.get_output_workspace(
            self.params.live_link, output_path, temp_geoh5
        )
        with ws as workspace:
            # Put entities in output workspace.
            param_dict["geoh5"] = workspace
            for key, value in param_dict.items():
                if isinstance(value, ObjectBase):
                    param_dict[key] = value.copy(parent=workspace, copy_children=True)

            # Write output uijson.
            ifile = InputFile(
                ui_json=self.params.input_file.ui_json,
                validation_options={"disabled": True},
            )
            new_params = ClusteringParams(input_file=ifile, **param_dict)
            new_params.write_input_file(
                name=temp_geoh5.replace(".geoh5", ".ui.json"),
                path=output_path,
                validate=False,
            )
            # Run driver.
            driver = ClusteringDriver(new_params)
            driver.run()

        if self.params.live_link:
            print("Live link active. Check your ANALYST session for new mesh.")
            return ["Geoscience ANALYST Pro - Live link"]
        else:
            print("Saved to " + os.path.abspath(output_path))
            return []

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
    app = Clustering(ui_json=ifile)
    print("Loaded. Building the clustering app . . .")
    app.run()
    print("Done")
