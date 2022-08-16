#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

# pylint: disable=E0401

from __future__ import annotations

import ast
import os
import sys
import time
import webbrowser
from os import environ, makedirs, path

import dash_daq as daq
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import callback_context, dash_table, dcc, html, no_update
from dash.dependencies import Input, Output
from flask import Flask
from geoh5py.objects import ObjectBase
from geoh5py.ui_json import InputFile
from geoh5py.workspace import Workspace
from jupyter_dash import JupyterDash

from geoapps.clustering.constants import app_initializer
from geoapps.clustering.driver import ClusteringDriver
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
                        dcc.Markdown("Output path:"),
                        dcc.Input(
                            id="output_path",
                            style={"margin-bottom": "20px"},
                            value=self.defaults["output_path"],
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
                            options=np.arange(0, self.defaults["n_clusters"], 1),
                            value=0,
                            style={"margin-bottom": "20px"},
                        ),
                        daq.ColorPicker(  # pylint: disable=E1102
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
                # Creating stored variables that can be passed through callbacks.
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

        # Update cluster color picker options from n_clusters
        self.app.callback(
            Output(component_id="select_cluster", component_property="options"),
            Input(component_id="n_clusters", component_property="value"),
        )(Clustering.update_select_cluster_options)

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
            Output(component_id="output_path", component_property="value"),
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
            Input(component_id="output_path", component_property="value"),
            Input(component_id="live_link", component_property="value"),
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
        )(self.trigger_click)

    def get_cluster_defaults(self) -> dict:
        """
        Get initial values from self.params to initialize the dash components.
        :return defaults: Initial values for dash components.
        """
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
                    full_list = ast.literal_eval(value)
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
                    full_list = ast.literal_eval(value)
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
    def update_color_select(checkbox: list) -> dict:
        """
        Updating visibility for cluster color picker.
        :param checkbox: Checkbox to select cluster colors.
        :return style: Return style for color_select_div. If display is "none", the div isn't visible.
        """
        if not checkbox:
            return {"display": "none"}
        else:
            return {"width": "25%", "display": "inline-block", "vertical-align": "top"}

    @staticmethod
    def update_norm_tabs(checkbox: list) -> dict:
        """
        Updating visibility for normalization plots.
        :param checkbox: Checkbox to show normalization plots.
        :return style: Return style for norm_tabs. If display is "none", the div isn't visible.
        """
        if not checkbox:
            return {"display": "none"}
        else:
            return {"display": "block"}

    @staticmethod
    def update_select_cluster_options(n_clusters):
        return np.arange(0, n_clusters, 1)

    def get_data_channels(self, channels: list) -> dict:
        """
        Loop through channels and add them to the data channels dict with name and object of all the current data.
        :param channels: Subset of data used for clustering and available for plotting.
        :return data_channels: Dictionary of data names and the corresponding Data objects.
        """
        data_channels = {}
        for channel in channels:
            if channel not in data_channels:
                if channel == "None":
                    data_channels[channel] = None
                elif self.params.geoh5.get_entity(channel):
                    data_channels[channel] = self.params.geoh5.get_entity(channel)[0]

        return data_channels

    def update_channels(
        self,
        channel: str,
        channels: list,
        full_scales: dict,
        full_lower_bounds: dict,
        full_upper_bounds: dict,
        kmeans: np.ndarray,
        indices: np.ndarray,
    ) -> dict:
        """
        Update the data options for the scatter plot and histogram from the data subset.
        :param channel: Input data name for histogram, boxplot.
        :param channels: Subset of data used for clustering and available for plotting.
        :param full_scales: Dictionary of data names and the corresponding scales.
        :param full_lower_bounds: Dictionary of data names and the corresponding lower bounds.
        :param full_upper_bounds: Dictionary of data names and the corresponding upper bounds.
        :param kmeans: K-means for the selected cluster number.
        :param indices: Active indices from current downsampling.
        :return update_dict: Dictionary of new channels and other affected parameters that need to be updated.
        """
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
                properties_dict = self.update_properties(
                    chan, full_scales, full_lower_bounds, full_upper_bounds
                )
                full_scales = properties_dict["full_scales"]
                full_lower_bounds = properties_dict["full_lower_bounds"]
                full_upper_bounds = properties_dict["full_upper_bounds"]

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
        self,
        channel: str,
        full_scales: dict,
        full_lower_bounds: dict,
        full_upper_bounds: dict,
    ) -> dict:
        """
        Get stored scale and bounds for a given channel. If there's no stored value, set a default.
        :param channel: Input data name for histogram, boxplot.
        :param full_scales: Dictionary of data names and the corresponding scales.
        :param full_lower_bounds: Dictionary of data names and the corresponding lower bounds.
        :param full_upper_bounds: Dictionary of data names and the corresponding upper bounds.
        :return update_dict: Dictionary of new scale and bounds.
        """
        if channel is not None:
            if channel not in full_scales:
                full_scales[channel] = 1
            scale = full_scales[channel]

            if (channel not in full_lower_bounds) or (
                full_lower_bounds[channel] is None
            ):
                full_lower_bounds[channel] = np.nanmin(
                    self.data_channels[channel].values
                )
            lower_bounds = float(full_lower_bounds[channel])

            if (channel not in full_upper_bounds) or (
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
        filename: str,
        contents: str,
        objects: str,
        x: str,
        y: str,
        z: str,
        color: str,
        size: str,
        channel: str,
        channels: list,
        scale: int,
        lower_bounds: float,
        upper_bounds: float,
        downsampling: int,
        select_cluster: int,
        n_clusters: int,
        full_scales: dict,
        full_lower_bounds: dict,
        full_upper_bounds: dict,
        color_picker: dict,
        color_pickers: list,
        kmeans: list,
        indices: list,
        clusters: dict,
        output_path: str,
        live_link: list,
    ) -> tuple:
        """
        Update self.params and dash components from user input.
        :param filename: Input filename. Either workspace or ui_json.
        :param contents: Input file contents. Either workspace or ui_json.
        :param objects: Input object name.
        :param x: Input x data name.
        :param y: Input y data name.
        :param z: Input z data name.
        :param color: Input color data name.
        :param size: Input size data name.
        :param channel: Input channel data name. Data displayed on histogram and boxplot.
        :param channels: The subset of data that is clustered and able to be plotted.
        :param scale: Scale for channel data.
        :param lower_bounds: Lower bounds for channel data.
        :param upper_bounds: Upper bounds for channel data.
        :param downsampling: Percent downsampling.
        :param select_cluster: Selected cluster used for picking cluster color.
        :param n_clusters: Number of clusters to make when running clustering.
        :param full_scales: Dictionary of data names and the corresponding scales.
        :param full_lower_bounds: Dictionary of data names and the corresponding lower bounds.
        :param full_upper_bounds: Dictionary of data names and the corresponding upper bounds.
        :param color_picker: Current selected color from color picker.
        :param color_pickers: List of colors with index corresponding to cluster number.
        :param kmeans: K-means values for n_clusters.
        :param indices: Active indices for data, gotten from downsampling.
        :param clusters: K-means values for (2, 4, 8, 16, 32, n_clusters)
        :param output_path: Output path for where to export clusters.
        :param live_link: Checkbox to enable monitoring directory.
        :return outputs: Values to update all the dash components in the callback.
        """
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
            "output_path",
        ]
        # Trigger is which variable triggered the callback
        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]

        if full_scales is None:
            full_scales = {}
        if full_lower_bounds is None:
            full_lower_bounds = {}
        if full_upper_bounds is None:
            full_upper_bounds = {}

        # Read in dcc.Store variables
        if indices is not None:
            indices = np.array(indices)
        if kmeans is not None:
            kmeans = np.array(kmeans)
        clusters = {int(k): v for k, v in clusters.items()}

        update_dict = {}
        if trigger == "upload":
            if filename.endswith(".ui.json"):
                # Update params from uploaded uijson
                update_dict = self.update_from_uijson(contents)
                if "plot_kmeans" in update_dict:
                    plot_kmeans = update_dict["plot_kmeans"]
                    if type(plot_kmeans) != list:
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
                if (
                    "output_path" in update_dict
                    and update_dict["output_path"] is not None
                ):
                    update_dict.update({"output_path": update_dict["output_path"]})
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
                self.params.geoh5 = update_dict["geoh5"]
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
                data_update_dict = self.update_data_options(update_dict["objects_name"])
                update_dict.update(
                    {
                        "data_options": data_update_dict["data_options"],
                        "channel_options": data_update_dict["data_options"],
                        "channel": None,
                        "x_name": None,
                        "y_name": None,
                        "z_name": None,
                        "color_name": None,
                        "size_name": None,
                    }
                )
                update_dict.update(
                    {
                        "objects_name": update_dict["objects_name"],
                        "channels_options": data_update_dict["data_options"],
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
            data_update_dict = self.update_data_options(objects)
            update_dict.update(
                {
                    "data_options": data_update_dict["data_options"],
                    "channel_options": data_update_dict["data_options"],
                    "channel": None,
                    "x_name": None,
                    "y_name": None,
                    "z_name": None,
                    "color_name": None,
                    "size_name": None,
                }
            )
            update_dict.update(
                {
                    "objects_name": objects,
                    "channels_options": data_update_dict["data_options"],
                    "channels": None,
                }
            )
        elif trigger == "select_cluster":
            # Update color displayed by the dash colorpicker
            update_dict = {"color_picker": dict(hex=color_pickers[select_cluster])}
        elif trigger == "output_path":
            if output_path is not None:
                update_dict.update({"output_path": output_path})
        elif trigger == "live_link":
            if not live_link:
                self.params.live_link = False
            else:
                self.params.live_link = True
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
            if param in update_dict:
                outputs.append(update_dict[param])
            else:
                outputs.append(no_update)

        return tuple(outputs)

    def update_param_dict(self, update_dict: dict):
        """
        Update self.params from update_dict.
        :param update_dict: Dictionary of parameters to update and the new values to assign.
        """
        if "plot_kmeans" in update_dict:
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
                    if update_dict[key + "_name"] is None:
                        setattr(self.params, key, None)
                    elif update_dict[key + "_name"] in self.data_channels:
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
        n_clusters: int,
        dataframe_dict: list[dict],
        channel: str,
        lower_bounds: float,
        upper_bounds: float,
        x: str,
        x_log: list,
        x_thresh: float,
        x_min: float,
        x_max: float,
        y: str,
        y_log: list,
        y_thresh: float,
        y_min: float,
        y_max: float,
        z: str,
        z_log: list,
        z_thresh: float,
        z_min: float,
        z_max: float,
        color: str,
        color_log: list,
        color_thresh: float,
        color_min: float,
        color_max: float,
        color_maps: list,
        color_pickers: list,
        size: str,
        size_log: list,
        size_thresh: float,
        size_min: float,
        size_max: float,
        size_markers: int,
        kmeans: list,
        indices: list,
        clusters: dict,
    ) -> (go.Figure, list[dict], go.Figure, go.Figure, go.Figure, go.Figure):
        """
        Update plots.
        :param n_clusters: Number of clusters.
        :param dataframe_dict: Dataframe with all channels and their data values.
        :param channel: Name of data displayed on histogram, boxplot.
        :param lower_bounds: Lower bounds for channel data.
        :param upper_bounds: Upper bounds for channel data.
        :param x: Name of data for x-axis of scatter plot.
        :param x_log: Checkbox for plotting log for x-axis of scatter plot.
        :param x_thresh: Threshold for x-axis of scatter plot.
        :param x_min: Min for x-axis of scatter plot.
        :param x_max: Max for x-axis of scatter plot.
        :param y: Name of data for y-axis of scatter plot.
        :param y_log: Checkbox for plotting log for y-axis of scatter plot
        :param y_thresh: Threshold for y-axis of scatter plot.
        :param y_min: Min for y-axis of scatter plot.
        :param y_max: Max for y-axis of scatter plot.
        :param z: Name of data for z-axis of scatter plot.
        :param z_log: Checkbox for plotting log for z-axis of scatter plot.
        :param z_thresh: Threshold for z-axis of scatter plot.
        :param z_min: Min for z-axis of scatter plot.
        :param z_max: Max for z-axis of scatter plot.
        :param color: Name of data for color-axis of scatter plot.
        :param color_log: Checkbox for plotting log for color-axis of scatter plot.
        :param color_thresh: Threshold for color-axis of scatter plot.
        :param color_min: Min for color-axis of scatter plot.
        :param color_max: Max for color-axis of scatter plot.
        :param color_maps: Color map for scatter plot.
        :param color_pickers: List of colors with index corresponding to cluster number.
        :param size: Name of data for size-axis of scatter plot.
        :param size_log: Checkbox for plotting log for size-axis of scatter plot.
        :param size_thresh: Threshold for size-axis of scatter plot.
        :param size_min: Min for size-axis of scatter plot.
        :param size_max: Max for size-axis of scatter plot.
        :param size_markers: Size of markers for scatter plot.
        :param kmeans: K-means for n_clusters.
        :param indices: Active indices for data, determined by downsampling.
        :param clusters: K-means values for (2, 4, 8, 16, 32, n_clusters)
        :return crossplot: Scatter plot with axes x, y, z, color, size.
        :return stats_table: Table of statistics: count, mean, std, min, max, etc. for data subset.
        :return matrix: Confusion matrix for data subset.
        :return histogram: Histogram of channel data.
        :return boxplot: Boxplots for clusters for channel data.
        :return inertia: Plot of kmeans inertia against number of clusters.
        """
        # Read in stored dataframe.
        dataframe = pd.DataFrame(dataframe_dict)
        # Read in stored clusters. Convert keys from string back to int.
        clusters = {int(k): v for k, v in clusters.items()}
        if kmeans is not None:
            kmeans = np.array(kmeans)
        if indices is not None:
            indices = np.array(indices)

        if not dataframe.empty:
            if color_maps == "kmeans":
                # Update color_maps
                color_maps = Clustering.update_colormap(n_clusters, color_pickers)
            elif color_maps is None:
                color_maps = [[0.0, "rgb(0,0,0)"]]

            # Input downsampled data to scatterplot so it doesn't regenerate data every time a parameter changes.
            axis_values = []
            for axis in [x, y, z, color, size]:
                if axis == "kmeans":
                    axis_values.append(self.data_channels["kmeans"])
                elif axis is not None:
                    axis_values.append(PlotData(axis, dataframe[axis].values))
                else:
                    axis_values.append(None)

            x, y, z, color, size = tuple(axis_values)  # pylint: disable=W0632

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
            stats_table = self.make_stats_table(dataframe)
            matrix = self.make_heatmap(dataframe)
            histogram = self.make_hist_plot(
                dataframe, channel, lower_bounds, upper_bounds
            )
            boxplot = self.make_boxplot(
                n_clusters,
                channel,
                color_pickers,
                kmeans,
                indices,
            )
            inertia = self.make_inertia_plot(n_clusters, clusters)
            return crossplot, stats_table, matrix, histogram, boxplot, inertia

        else:
            return go.Figure(), None, go.Figure(), go.Figure(), go.Figure(), go.Figure()

    @staticmethod
    def update_colormap(n_clusters: int, color_pickers: list) -> list:
        """
        Change the colormap for clusters
        :param n_clusters: Number of clusters.
        :param color_pickers: List of colors with index corresponding to cluster number.
        :return color_map: Color map for plotting kmeans on scatter plot.
        """
        color_map = {}
        for ii in range(n_clusters):
            colorpicker = color_pickers[ii]
            if "#" in colorpicker:
                color = colorpicker.lstrip("#")
                color_map[ii] = [
                    np.min([ii / (n_clusters - 1), 1]),
                    "rgb("
                    + ",".join([f"{int(color[i:i + 2], 16)}" for i in (0, 2, 4)])
                    + ")",
                ]
            else:
                color_map[ii] = [
                    np.min([ii / (n_clusters - 1), 1]),
                    colorpicker,
                ]

        return list(color_map.values())

    def update_clustering(
        self,
        channel: str,
        channels: list,
        full_scales: dict,
        full_lower_bounds: dict,
        full_upper_bounds: dict,
        downsampling: int,
        n_clusters: int,
        kmeans: np.ndarray,
        clusters: dict,
        indices: np.ndarray,
        workspace: Workspace,
        update_all_clusters: bool,
    ) -> dict:
        """
        Update clustering and the dropdown data that depends on kmeans.
        :param channel: Name of data displayed on histogram, boxplot.
        :param channels: Subset of data used for clustering and plotting.
        :param full_scales: Dictionary of data names and the corresponding scales.
        :param full_lower_bounds: Dictionary of data names and the corresponding lower bounds.
        :param full_upper_bounds: Dictionary of data names and the corresponding upper bounds.
        :param downsampling: Percent downsampling of data.
        :param n_clusters: Number of clusters.
        :param kmeans: K-mean values for n_clusters.
        :param clusters: K-means values for (2, 4, 8, 16, 32, n_clusters)
        :param indices: Active indices for data, determined by downsampling.
        :param workspace: Workspace.
        :param update_all_clusters: Whether to update all clusters values, or just n_clusters.
        :return update_dict: Dictionary of parameters to update and their new values.
        """
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
            ClusteringDriver.update_dataframe(
                downsampling, channels, workspace, downsample_min=5000
            )
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
    def make_inertia_plot(n_clusters: int, clusters: dict) -> go.Figure:
        """
        Generate an inertia plot.
        :param n_clusters: Number of clusters.
        :param clusters: K-means values for (2, 4, 8, 16, 32, n_clusters)
        :return inertia_plot: Inertia figure.
        """
        if n_clusters in clusters:
            ind = np.sort(list(clusters.keys()))
            inertias = [clusters[ii]["inertia"] for ii in ind]
            line = go.Scatter(x=ind, y=inertias, mode="lines")
            point = go.Scatter(
                x=[n_clusters],
                y=[clusters[n_clusters]["inertia"]],
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
            return go.Figure()

    @staticmethod
    def make_hist_plot(
        dataframe: pd.DataFrame, channel: str, lower_bounds: float, upper_bounds: float
    ) -> go.Figure:
        """
        Generate an histogram plot for the selected data channel.
        :param dataframe: Data names and values for the selected data subset.
        :param channel: Name of data plotted on histogram, boxplot.
        :param lower_bounds: Lower bounds for channel data.
        :param upper_bounds: Upper bounds for channel data.
        :return histogram: Histogram figure.
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
            return go.Figure()

    def make_boxplot(
        self,
        n_clusters: int,
        channel: str,
        color_pickers: list,
        kmeans: np.ndarray,
        indices: np.ndarray,
    ) -> go.Figure:
        """
        Generate a box plot for each cluster.
        :param n_clusters: Number of clusters.
        :param channel: Name of data to be plotted on histogram, boxplot.
        :param color_pickers: List of colors with index corresponding to cluster number.
        :param kmeans: K-means values for n_clusters.
        :param indices: Active indices for data, determined by downsampling.
        :return boxplot: Boxplot figure.
        """
        if (kmeans is not None) and (channel is not None):
            boxes = []
            for ii in range(n_clusters):
                cluster_ind = kmeans[indices] == ii
                x = np.ones(np.sum(cluster_ind)) * ii
                y = self.data_channels[channel].values[indices][cluster_ind]

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

            boxplot = go.Figure()

            boxplot.data = []
            for box in boxes:
                boxplot.add_trace(box)

            boxplot.update_layout(
                {
                    "xaxis": {"title": "Cluster #"},
                    "yaxis": {"title": channel},
                    "height": 600,
                    "width": 600,
                }
            )
            return boxplot
        else:
            return go.Figure()

    @staticmethod
    def make_stats_table(dataframe: pd.DataFrame) -> list[dict]:
        """
        Generate a table of statistics using pandas
        :param dataframe: Data names and values for selected data subset.
        :return stats_table: Table of stats: count, mean, std, min, max, etc.
        """
        stats_df = dataframe.describe(percentiles=None, include=None, exclude=None)
        stats_df.insert(0, "", stats_df.index)
        return stats_df.to_dict("records")

    @staticmethod
    def make_heatmap(dataframe: pd.DataFrame):
        """
        Generate a confusion matrix.
        :param dataframe: Data names and values for selected data subset.
        :return matrix: Confusion matrix figure.
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

    def trigger_click(self, _):
        """
        Write cluster groups to the target geoh5 object.
        :return live_link: Checkbox value for dash live_link component.
        """
        param_dict = self.params.to_dict()
        temp_geoh5 = f"Clustering_{time.time():.0f}.geoh5"

        if self.params.output_path is not None and os.path.exists(
            os.path.abspath(self.params.output_path)
        ):
            output_path = os.path.abspath(self.params.output_path)
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
            new_params = ClusteringParams(**param_dict)
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
