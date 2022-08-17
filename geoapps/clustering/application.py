#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

# pylint: disable=W0613
# pylint: disable=E0401

from __future__ import annotations

import ast
import os
import sys
import time
import uuid

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import callback_context, no_update
from dash.dependencies import Input, Output, State
from flask import Flask
from geoh5py.objects import ObjectBase
from geoh5py.shared.utils import is_uuid
from geoh5py.ui_json import InputFile
from jupyter_dash import JupyterDash

from geoapps.base.application import BaseApplication
from geoapps.clustering.constants import app_initializer
from geoapps.clustering.driver import ClusteringDriver
from geoapps.clustering.layout import cluster_layout
from geoapps.clustering.params import ClusteringParams
from geoapps.clustering.plot_data import PlotData
from geoapps.scatter_plot.application import ScatterPlots
from geoapps.scatter_plot.driver import ScatterPlotDriver
from geoapps.shared_utils.utils import colors


class Clustering(ScatterPlots):
    _param_class = ClusteringParams
    _driver_class = ClusteringDriver

    def __init__(self, ui_json=None, **kwargs):
        app_initializer.update(kwargs)
        if ui_json is not None and os.path.exists(ui_json.path):
            self.params = self._param_class(ui_json)
        else:
            self.params = self._param_class(**app_initializer)

        super().__init__(**self.params.to_dict())

        # Params and driver used for updating scatter plot in make_scatter_plot function.
        self.scatter_params = self._param_class(**self.params.to_dict())
        self.scatter_driver = ScatterPlotDriver(self.scatter_params)

        external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
        server = Flask(__name__)
        self.app = JupyterDash(
            server=server,
            url_base_pathname=os.environ.get("JUPYTERHUB_SERVICE_PREFIX", "/"),
            external_stylesheets=external_stylesheets,
        )

        self.app.layout = cluster_layout

        # Callbacks relating to layout
        self.app.callback(
            Output(component_id="x_div", component_property="style"),
            Output(component_id="y_div", component_property="style"),
            Output(component_id="z_div", component_property="style"),
            Output(component_id="color_div", component_property="style"),
            Output(component_id="size_div", component_property="style"),
            Input(component_id="axes_panels", component_property="value"),
        )(self.update_visibility)
        self.app.callback(
            Output(component_id="color_select_div", component_property="style"),
            Input(component_id="show_color_picker", component_property="value"),
        )(Clustering.update_color_select)
        self.app.callback(
            Output(component_id="norm_tabs", component_property="style"),
            Input(component_id="show_norm_tabs", component_property="value"),
        )(Clustering.update_norm_tabs)

        # Callbacks to update params
        self.app.callback(
            Output(component_id="select_cluster", component_property="options"),
            Output(component_id="select_cluster", component_property="value"),
            Input(component_id="n_clusters", component_property="value"),
            Input(component_id="select_cluster", component_property="value"),
        )(Clustering.update_select_cluster_options)
        self.app.callback(
            Output(component_id="objects", component_property="options"),
            Output(component_id="objects", component_property="value"),
            Output(component_id="ui_json_data", component_property="data"),
            Output(component_id="upload", component_property="filename"),
            Output(component_id="upload", component_property="contents"),
            Input(component_id="upload", component_property="filename"),
            Input(component_id="upload", component_property="contents"),
        )(self.update_object_options)
        self.app.callback(
            Output(component_id="data_subset", component_property="options"),
            Output(component_id="data_subset", component_property="value"),
            Input(component_id="ui_json_data", component_property="data"),
            Input(component_id="objects", component_property="value"),
        )(self.update_data_subset)
        self.app.callback(
            Output(component_id="color_pickers", component_property="data"),
            Input(component_id="ui_json_data", component_property="data"),
            Input(component_id="color_pickers", component_property="data"),
            Input(component_id="color_picker", component_property="value"),
            Input(component_id="select_cluster", component_property="value"),
        )(Clustering.update_color_pickers)
        self.app.callback(
            Output(component_id="color_picker", component_property="value"),
            Input(component_id="color_pickers", component_property="data"),
            Input(component_id="select_cluster", component_property="value"),
        )(Clustering.update_color_picker)
        self.app.callback(
            Output(component_id="x", component_property="options"),
            Output(component_id="y", component_property="options"),
            Output(component_id="z", component_property="options"),
            Output(component_id="color", component_property="options"),
            Output(component_id="size", component_property="options"),
            Output(component_id="channel", component_property="options"),
            Output(component_id="color_maps", component_property="options"),
            Output(component_id="x", component_property="value"),
            Output(component_id="y", component_property="value"),
            Output(component_id="z", component_property="value"),
            Output(component_id="color", component_property="value"),
            Output(component_id="size", component_property="value"),
            Output(component_id="channel", component_property="value"),
            Input(component_id="ui_json_data", component_property="data"),
            Input(component_id="data_subset", component_property="value"),
            Input(component_id="data_subset", component_property="options"),
            Input(component_id="x", component_property="value"),
            Input(component_id="y", component_property="value"),
            Input(component_id="z", component_property="value"),
            Input(component_id="color", component_property="value"),
            Input(component_id="size", component_property="value"),
            Input(component_id="channel", component_property="value"),
        )(Clustering.update_data_from_data_subset)
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
            Input(component_id="ui_json_data", component_property="data"),
            Input(component_id="x", component_property="value"),
            Input(component_id="y", component_property="value"),
            Input(component_id="z", component_property="value"),
            Input(component_id="color", component_property="value"),
            Input(component_id="size", component_property="value"),
            Input(component_id="kmeans", component_property="data"),
        )(self.update_channel_bounds)
        self.app.callback(
            Output(component_id="scale", component_property="value"),
            Output(component_id="lower_bounds", component_property="value"),
            Output(component_id="upper_bounds", component_property="value"),
            Output(component_id="full_scales", component_property="data"),
            Output(component_id="full_lower_bounds", component_property="data"),
            Output(component_id="full_upper_bounds", component_property="data"),
            Input(component_id="ui_json_data", component_property="data"),
            Input(component_id="channel", component_property="value"),
            Input(component_id="scale", component_property="value"),
            Input(component_id="lower_bounds", component_property="value"),
            Input(component_id="upper_bounds", component_property="value"),
            Input(component_id="full_scales", component_property="data"),
            Input(component_id="full_lower_bounds", component_property="data"),
            Input(component_id="full_upper_bounds", component_property="data"),
        )(self.update_properties)
        self.app.callback(
            Output(component_id="downsampling", component_property="value"),
            Output(component_id="x_log", component_property="value"),
            Output(component_id="x_thresh", component_property="value"),
            Output(component_id="y_log", component_property="value"),
            Output(component_id="y_thresh", component_property="value"),
            Output(component_id="z_log", component_property="value"),
            Output(component_id="z_thresh", component_property="value"),
            Output(component_id="color_log", component_property="value"),
            Output(component_id="color_thresh", component_property="value"),
            Output(component_id="color_maps", component_property="value"),
            Output(component_id="size_log", component_property="value"),
            Output(component_id="size_thresh", component_property="value"),
            Output(component_id="size_markers", component_property="value"),
            Output(component_id="n_clusters", component_property="value"),
            Output(component_id="ga_group_name", component_property="value"),
            Output(component_id="monitoring_directory", component_property="value"),
            Input(component_id="ui_json_data", component_property="data"),
        )(self.update_remainder_from_ui_json)

        # Clustering callbacks
        self.app.callback(
            Output(component_id="dataframe", component_property="data"),
            Output(component_id="mapping", component_property="data"),
            Output(component_id="indices", component_property="data"),
            Input(component_id="downsampling", component_property="value"),
            Input(component_id="data_subset", component_property="value"),
        )(self.update_dataframe)
        self.app.callback(
            Output(component_id="kmeans", component_property="data"),
            Output(component_id="clusters", component_property="data"),
            Input(component_id="dataframe", component_property="data"),
            Input(component_id="n_clusters", component_property="value"),
            Input(component_id="full_scales", component_property="data"),
            Input(component_id="clusters", component_property="data"),
            Input(component_id="mapping", component_property="data"),
        )(Clustering.run_clustering)

        # Callbacks to update the plots
        self.app.callback(
            Output(component_id="crossplot", component_property="figure"),
            Input(component_id="n_clusters", component_property="value"),
            Input(component_id="dataframe", component_property="data"),
            Input(component_id="kmeans", component_property="data"),
            Input(component_id="indices", component_property="data"),
            Input(component_id="color_pickers", component_property="data"),
            Input(component_id="channel", component_property="options"),
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
        )(self.make_scatter_plot)
        self.app.callback(
            Output(component_id="stats_table", component_property="data"),
            Input(component_id="dataframe", component_property="data"),
        )(Clustering.make_stats_table)
        self.app.callback(
            Output(component_id="matrix", component_property="figure"),
            Input(component_id="dataframe", component_property="data"),
        )(Clustering.make_heatmap)
        self.app.callback(
            Output(component_id="histogram", component_property="figure"),
            Input(component_id="dataframe", component_property="data"),
            Input(component_id="channel", component_property="value"),
            Input(component_id="channel", component_property="options"),
            Input(component_id="lower_bounds", component_property="value"),
            Input(component_id="upper_bounds", component_property="value"),
        )(Clustering.make_hist_plot)
        self.app.callback(
            Output(component_id="boxplot", component_property="figure"),
            Input(component_id="n_clusters", component_property="value"),
            Input(component_id="channel", component_property="value"),
            Input(component_id="channel", component_property="options"),
            Input(component_id="color_pickers", component_property="data"),
            Input(component_id="kmeans", component_property="data"),
            Input(component_id="indices", component_property="data"),
        )(self.make_boxplot)
        self.app.callback(
            Output(component_id="inertia", component_property="figure"),
            Input(component_id="n_clusters", component_property="value"),
            Input(component_id="clusters", component_property="data"),
        )(Clustering.make_inertia_plot)

        # Callback to export the clusters as a geoh5 file
        self.app.callback(
            Output(component_id="live_link", component_property="value"),
            Input(component_id="export", component_property="n_clicks"),
            State(component_id="monitoring_directory", component_property="value"),
            State(component_id="live_link", component_property="value"),
            State(component_id="n_clusters", component_property="value"),
            State(component_id="objects", component_property="value"),
            State(component_id="data_subset", component_property="value"),
            State(component_id="color_pickers", component_property="data"),
            State(component_id="downsampling", component_property="value"),
            State(component_id="full_scales", component_property="data"),
            State(component_id="full_lower_bounds", component_property="data"),
            State(component_id="full_upper_bounds", component_property="data"),
            State(component_id="x", component_property="value"),
            State(component_id="x_log", component_property="value"),
            State(component_id="x_thresh", component_property="value"),
            State(component_id="x_min", component_property="value"),
            State(component_id="x_max", component_property="value"),
            State(component_id="y", component_property="value"),
            State(component_id="y_log", component_property="value"),
            State(component_id="y_thresh", component_property="value"),
            State(component_id="y_min", component_property="value"),
            State(component_id="y_max", component_property="value"),
            State(component_id="z", component_property="value"),
            State(component_id="z_log", component_property="value"),
            State(component_id="z_thresh", component_property="value"),
            State(component_id="z_min", component_property="value"),
            State(component_id="z_max", component_property="value"),
            State(component_id="color", component_property="value"),
            State(component_id="color_log", component_property="value"),
            State(component_id="color_thresh", component_property="value"),
            State(component_id="color_min", component_property="value"),
            State(component_id="color_max", component_property="value"),
            State(component_id="color_maps", component_property="value"),
            State(component_id="size", component_property="value"),
            State(component_id="size_log", component_property="value"),
            State(component_id="size_thresh", component_property="value"),
            State(component_id="size_min", component_property="value"),
            State(component_id="size_max", component_property="value"),
            State(component_id="size_markers", component_property="value"),
            State(component_id="channel", component_property="value"),
            State(component_id="ga_group_name", component_property="value"),
            prevent_initial_call=True,
        )(self.trigger_click)

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
    def get_name(channel: str, channel_options: list) -> str:
        """
        Get channel name from uid and full channel options.

        :param channel: Channel uid.
        :param channel_options: Full options for channel. List of {"label": channel_name, "value": channel_uid}.

        :return channel_name: Channel name.
        """
        channel_name = None
        if channel_options:
            for item in channel_options:
                if item["value"] == channel:
                    channel_name = item["label"]
        return channel_name

    @staticmethod
    def update_select_cluster_options(
        n_clusters: int, select_cluster: int
    ) -> (list, int):
        """
        Update select cluster dropdown options to have a max of n_clusters. Dropdown used for picking colors for
        clusters.

        :param n_clusters: Number of clusters.
        :param select_cluster: Current selected cluster in dropdown.

        :return options: List of options from 0 to n_clusters.
        :return select_cluster: Current selected cluster in dropdown.
        """
        options = np.arange(0, n_clusters, 1)
        if select_cluster is None and len(options) > 0:
            select_cluster = options[0]
        return options, select_cluster

    @staticmethod
    def update_color_pickers(
        ui_json_data: dict, color_pickers: list, color_picker: dict, select_cluster: int
    ) -> list:
        """
        Update list of colors corresponding to clusters.

        :param ui_json_data: Uploaded ui.json data.
        :param color_pickers: List of colors corresponding to clusters.
        :param color_picker: Color corresponding to select_cluster.
        :param select_cluster: Current selected cluster.

        :return color_pickers: List of colors corresponding to clusters.
        """
        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]
        if trigger == "ui_json_data":
            # Read in list of colors from ui.json.
            if type(ui_json_data["color_pickers"]) == list:
                full_list = ui_json_data["color_pickers"]
            else:
                # Convert string to list.
                full_list = ast.literal_eval(ui_json_data["color_pickers"])
            if (full_list is None) | (not full_list):
                # Default list of colors.
                color_pickers = colors
            else:
                color_pickers = full_list
        elif trigger == "color_picker":
            # Update list of colors from user clicking on color picker.
            color_pickers[select_cluster] = color_picker["hex"]
        return color_pickers

    @staticmethod
    def update_color_picker(color_pickers: list, select_cluster: int) -> dict:
        """
        Update the displayed color on color picker when select cluster is switched.

        :param color_pickers: Full list of colors corresponding to clusters.
        :param select_cluster: Current selected cluster from dropdown.

        :return color_picker: Displayed value on color picker.
        """
        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]
        if trigger == "select_cluster" and color_pickers:
            return dict(hex=color_pickers[select_cluster])
        else:
            return no_update

    def update_data_subset(self, ui_json_data: dict, object_uid: str) -> (list, str):
        """
        Update data subset options and values from selected object.

        :param ui_json_data: Uploaded ui.json data.
        :param object_uid: Selected object from dropdown.

        :return options: Options for data subset dropdown.
        :return value: Value for data subset dropdown.
        """
        triggers = [c["prop_id"].split(".")[0] for c in callback_context.triggered]

        if "ui_json_data" in triggers:
            value = ast.literal_eval(ui_json_data["data_subset"])
            options = self.get_data_options("ui_json_data", ui_json_data, object_uid)
        else:
            value = []
            options = self.get_data_options("objects", ui_json_data, object_uid)

        return options, value

    @staticmethod
    def update_data_from_data_subset(
        ui_json_data: dict,
        data_subset: list,
        full_options: list,
        x: str,
        y: str,
        z: str,
        color: str,
        size: str,
        channel: str,
    ) -> (list, list, list, list, list, list, list, str, str, str, str, str, str):
        """
        Update data options and values for scatter plot and histogram/boxplot.

        :param ui_json_data: Uploaded ui.json data.
        :param data_subset: Subset of data used for clustering and plotting.
        :param full_options: Full options for data subset.
        :param x: X-axis data uid.
        :param y: Y-axis data uid.
        :param z: Z-axis data uid.
        :param color: Color-axis data uid.
        :param size: Size-axis data uid.
        :param channel: Data uid for boxplot/histogram.

        :return axis_options: Dropdown options for x-axis of scatter plot.
        :return axis_options: Dropdown options for y-axis of scatter plot.
        :return axis_options: Dropdown options for z-axis of scatter plot.
        :return axis_options: Dropdown options for color-axis of scatter plot.
        :return axis_options: Dropdown options for size-axis of scatter plot.
        :return channel_options: Dropdown options for histogram/boxplot.
        :return color_maps_options: Dropdown options for color maps in scatter plot.
        :return x_axis_values: Value for x-axis dropdown.
        :return y_axis_values: Value for y-axis dropdown.
        :return z_axis_values: Value for z-axis dropdown.
        :return color_axis_values: Value for color-axis dropdown.
        :return size_axis_values: Value for size-axis dropdown.
        :return channel_axis_values: Value for histogram/boxplot dropdown.
        """
        data_subset_options = []
        for item in full_options:
            if item["value"] in data_subset:
                data_subset_options.append(item)
        channel_options = data_subset_options
        axis_options = data_subset_options.copy()
        color_maps_options = px.colors.named_colorscales()

        # Add kmeans to the dropdown options for scatterplot.
        axis_options.append({"label": "kmeans", "value": "kmeans"})
        color_maps_options.insert(0, "kmeans")

        triggers = [c["prop_id"].split(".")[0] for c in callback_context.triggered]
        if "ui_json_data" in triggers:
            axis_values = Clustering.update_data_values(ui_json_data)
        else:
            axis_values = [None, None, None, None, None, None]
            axes = [x, y, z, color, size, channel]
            for i in range(len(axes)):
                if axes[i] in data_subset or axes[i] == "kmeans":
                    axis_values[i] = no_update
            axis_values = tuple(axis_values)

        # replace empty lists with empty dictionary
        if not axis_options:
            axis_options = {}
        if not channel_options:
            channel_options = {}
        return (
            axis_options,
            axis_options,
            axis_options,
            axis_options,
            axis_options,
            channel_options,
            color_maps_options,
        ) + axis_values

    @staticmethod
    def update_data_values(ui_json_data: dict) -> (str, str, str, str, str, str):
        """
        Read in axes values from ui.json.

        :param ui_json_data: Uploaded ui.json data.

        :return x_axis_values: Value for x-axis dropdown.
        :return y_axis_values: Value for y-axis dropdown.
        :return z_axis_values: Value for z-axis dropdown.
        :return color_axis_values: Value for color-axis dropdown.
        :return size_axis_values: Value for size-axis dropdown.
        :return channel_axis_values: Value for histogram/boxplot dropdown.
        """
        # Read in plot_kmeans list from ui.json. Used to know to plot kmeans, since kmeans can't be saved as data in
        # the ui.json.
        plot_kmeans = ast.literal_eval(ui_json_data["plot_kmeans"])
        if not plot_kmeans:
            plot_kmeans = [False, False, False, False, False]
        output_axes = []
        axes_names = ["x", "y", "z", "color", "size"]
        for i in range(len(axes_names)):
            axis_val = ui_json_data[axes_names[i]]
            if is_uuid(axis_val):
                output_axes.append(str(axis_val))
            elif (axis_val == "" or axis_val is None) and plot_kmeans[i]:
                output_axes.append("kmeans")
            else:
                output_axes.append(None)
        if is_uuid(ui_json_data["channel"]):
            output_axes.append(str(ui_json_data["channel"]))
        else:
            output_axes.append(None)

        return tuple(output_axes)

    def update_properties(
        self,
        ui_json_data: dict,
        channel: str,
        scale: int,
        lower_bounds: float,
        upper_bounds: float,
        full_scales: dict,
        full_lower_bounds: dict,
        full_upper_bounds: dict,
    ) -> tuple:
        """
        Update selected scale, bounds with stored value. Or update stored values with new scale, bounds.

        :param ui_json_data: Uploaded ui.json data.
        :param channel: Input data uid for histogram, boxplot.
        :param scale: Scale value for selected channel.
        :param lower_bounds: Lower bounds for selected channel.
        :param upper_bounds: Upper bounds for selected channel.
        :param full_scales: Dictionary of data names and the corresponding scales.
        :param full_lower_bounds: Dictionary of data names and the corresponding lower bounds.
        :param full_upper_bounds: Dictionary of data names and the corresponding upper bounds.

        :return scale: Scale value for selected channel.
        :return lower_bounds: Lower bounds for selected channel.
        :return upper_bounds: Upper bounds for selected channel.
        :return full_scales: Dictionary of data names and the corresponding scales.
        :return full_lower_bounds: Dictionary of data names and the corresponding lower bounds.
        :return full_upper_bounds: Dictionary of data names and the corresponding upper bounds.
        """
        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]

        if trigger == "ui_json_data":
            # Reconstruct scaling and bounds dicts from ui.json input lists.
            data_subset = ast.literal_eval(ui_json_data["data_subset"])
            if len(data_subset) == 0:
                full_scales, full_lower_bounds, full_upper_bounds = {}, {}, {}
            else:
                full_dicts = [None, None, None]
                params = ["full_scales", "full_lower_bounds", "full_upper_bounds"]
                for i in range(len(params)):
                    out_dict = {}
                    full_list = ast.literal_eval(ui_json_data[params[i]])
                    for j in range(len(data_subset)):
                        if (full_list is None) | (not full_list):
                            if params[i] == "full_scales":
                                out_dict[data_subset[j]] = 1
                            else:
                                out_dict[data_subset[j]] = None
                        else:
                            out_dict[data_subset[j]] = full_list[j]
                    full_dicts[i] = out_dict
                full_scales, full_lower_bounds, full_upper_bounds = tuple(full_dicts)

        if channel is not None:
            if trigger == "ui_json_data" or trigger == "channel":
                # Update scale, bounds from full_scales, full_bounds
                if channel not in full_scales:
                    full_scales[channel] = 1
                scale = full_scales[channel]

                if (channel not in full_lower_bounds) or (
                    full_lower_bounds[channel] is None
                ):
                    full_lower_bounds[channel] = np.nanmin(
                        self.workspace.get_entity(uuid.UUID(channel))[0].values
                    )
                lower_bounds = float(full_lower_bounds[channel])

                if (channel not in full_upper_bounds) or (
                    full_upper_bounds[channel] is None
                ):
                    full_upper_bounds[channel] = np.nanmax(
                        self.workspace.get_entity(uuid.UUID(channel))[0].values
                    )
                upper_bounds = float(full_upper_bounds[channel])
            else:
                # Update full_scales, full_bounds from scale, bounds
                full_scales[channel] = scale
                full_lower_bounds[channel] = lower_bounds
                full_upper_bounds[channel] = upper_bounds
        else:
            scale, lower_bounds, upper_bounds = None, None, None

        return (
            scale,
            lower_bounds,
            upper_bounds,
            full_scales,
            full_lower_bounds,
            full_upper_bounds,
        )

    def update_dataframe(
        self, downsampling: int, data_subset: list
    ) -> (pd.DataFrame, np.ndarray, np.ndarray):
        """
        Update dataframe.

        :param downsampling: Percent downsampling.
        :param data_subset: Selected data to use for creating dataframe and clustering.

        :return dataframe: Dataframe with data_subset data values.
        :return mapping: Mapping between generated kmeans and data to plot.
        :return indices: Active indices for plotting data.
        """
        if data_subset:
            dataframe, mapping, indices = ClusteringDriver.update_dataframe(
                downsampling, data_subset, self.workspace, downsample_min=5000
            )
            return dataframe, mapping, indices
        else:
            return None, None, None

    @staticmethod
    def run_clustering(
        dataframe_dict: list[dict],
        n_clusters: int,
        full_scales: dict,
        clusters: dict,
        mapping: list,
    ) -> (np.ndarray, dict):
        """
        Run clustering.

        :param dataframe_dict: Dataframe of data subset values.
        :param n_clusters: Number of clusters.
        :param full_scales: dictionary of data subset names and corresponding scaling factors.
        :param clusters: Current cluster values.
        :param mapping: Mapping between generated kmeans and data to plot.

        :return kmeans: K-means for n_clusters.
        :return clusters: K-means values for (2, 4, 8, 16, 32, n_clusters)
        """
        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]
        # Boolean to update clusters for inertia plot or not.
        if trigger != "n_clusters" or clusters == {}:
            update_all_clusters = True
        else:
            update_all_clusters = False

        kmeans, clusters = ClusteringDriver.run_clustering(
            n_clusters,
            dataframe_dict,
            full_scales,
            clusters,
            np.array(mapping),
            update_all_clusters,
        )
        return kmeans, clusters

    @staticmethod
    def update_colormap(n_clusters: int, color_pickers: list) -> list:
        """
        Change the colormap for clusters

        :param n_clusters: Number of clusters.
        :param color_pickers: List of colors with index corresponding to cluster number.

        :return color_map: Color map for plotting kmeans on scatter plot.
        """
        if color_pickers:
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
        else:
            return None

    def make_scatter_plot(
        self,
        n_clusters: int,
        dataframe_dict: list[dict],
        kmeans: list,
        indices: list,
        color_pickers: list,
        channel_options: list,
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
        size: str,
        size_log: list,
        size_thresh: float,
        size_min: float,
        size_max: float,
        size_markers: int,
    ) -> go.Figure:
        """
        Update plots.

        :param n_clusters: Number of clusters.
        :param dataframe_dict: Dataframe with all channels and their data values.
        :param kmeans: K-means for n_clusters.
        :param indices: Active indices for data, determined by downsampling.
        :param color_pickers: List of colors with index corresponding to cluster number.
        :param channel_options: Full list of data options for channel.
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
        :param size: Name of data for size-axis of scatter plot.
        :param size_log: Checkbox for plotting log for size-axis of scatter plot.
        :param size_thresh: Threshold for size-axis of scatter plot.
        :param size_min: Min for size-axis of scatter plot.
        :param size_max: Max for size-axis of scatter plot.
        :param size_markers: Size of markers for scatter plot.

        :return crossplot: Scatter plot with axes x, y, z, color, size.
        """
        # Read in stored dataframe.
        dataframe = pd.DataFrame(dataframe_dict)

        if not dataframe.empty:
            if kmeans is not None:
                kmeans = np.array(kmeans)
            if indices is not None:
                indices = np.array(indices)

            if color_maps == "kmeans" and kmeans is not None and kmeans != []:
                # Update color_maps
                color_maps = Clustering.update_colormap(n_clusters, color_pickers)
            if color_maps is None:
                color_maps = [[0.0, "rgb(0,0,0)"]]

            # Input downsampled data to scatterplot so it doesn't regenerate data every time a parameter changes.
            axis_values = [None, None, None, None, None]
            axes = [x, y, z, color, size]

            for i in range(len(axes)):
                if axes[i] == "kmeans" and kmeans is not None and kmeans != []:
                    axis_values[i] = PlotData(axes[i], kmeans[indices].astype(float))
                elif axes[i] is not None:
                    axis_name = Clustering.get_name(axes[i], channel_options)
                    if axis_name is not None:
                        axis_values[i] = PlotData(
                            axis_name, dataframe[axis_name].values
                        )

            x, y, z, color, size = tuple(axis_values)

            update_dict = {}
            for item in callback_context.triggered:
                update_dict[item["prop_id"].split(".")[0]] = item["value"]

            params_dict = self.get_params_dict(update_dict)
            params_dict.update(
                {
                    "downsampling": 100,
                    "color_maps": color_maps,
                    "x": x,
                    "y": y,
                    "z": z,
                    "color": color,
                    "size": size,
                }
            )
            self.scatter_params.update(params_dict, validate=False)
            crossplot = go.Figure(self.scatter_driver.run())
            return crossplot
        else:

            return go.Figure()

    @staticmethod
    def make_inertia_plot(n_clusters: int, clusters: dict) -> go.Figure:
        """
        Generate an inertia plot.

        :param n_clusters: Number of clusters.
        :param clusters: K-means values for (2, 4, 8, 16, 32, n_clusters)

        :return inertia: Plot of kmeans inertia against number of clusters.
        """
        inertia_plot = go.Figure()

        if clusters is not None:
            # Read in stored clusters. Convert keys from string back to int.
            clusters = {int(k): v for k, v in clusters.items()}

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

    @staticmethod
    def make_hist_plot(
        dataframe_dict: dict,
        channel: str,
        channel_options: list,
        lower_bounds: float,
        upper_bounds: float,
    ) -> go.Figure:
        """
        Generate an histogram plot for the selected data channel.

        :param dataframe_dict: Data names and values for the selected data subset.
        :param channel: Name of data plotted on histogram, boxplot.
        :param channel_options: Full list of data options for channel.
        :param lower_bounds: Lower bounds for channel data.
        :param upper_bounds: Upper bounds for channel data.

        :return histogram: Histogram of channel data.
        """
        dataframe = pd.DataFrame(dataframe_dict)

        channel_name = Clustering.get_name(channel, channel_options)
        if channel_name is not None and channel_name in dataframe:
            histogram = go.Figure(
                data=[
                    go.Histogram(
                        x=dataframe[channel_name].values,
                        histnorm="percent",
                        name=channel_name,
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
        channel_options: list,
        color_pickers: list,
        kmeans: list,
        indices: list,
    ) -> go.Figure:
        """
        Generate a box plot for each cluster.

        :param n_clusters: Number of clusters.
        :param channel: Name of data to be plotted on histogram, boxplot.
        :param channel_options: Full list of data options for channel.
        :param color_pickers: List of colors with index corresponding to cluster number.
        :param kmeans: K-means values for n_clusters.
        :param indices: Active indices for data, determined by downsampling.

        :return boxplot: Boxplots for clusters for channel data.
        """
        channel_name = Clustering.get_name(channel, channel_options)
        if (
            (kmeans is not None)
            and (channel_name is not None)
            and (indices is not None)
            and (color_pickers is not None)
        ):
            kmeans = np.array(kmeans)
            indices = np.array(indices)

            boxes = []
            y_data = self.workspace.get_entity(uuid.UUID(channel))[0].values
            for ii in range(n_clusters):
                cluster_ind = kmeans[indices] == ii
                x = np.ones(np.sum(cluster_ind)) * ii
                y = y_data[indices][cluster_ind]

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
                    "yaxis": {"title": channel_name},
                    "height": 600,
                    "width": 600,
                }
            )
            return boxplot
        else:
            return go.Figure()

    @staticmethod
    def make_stats_table(dataframe_dict: dict) -> list[dict]:
        """
        Generate a table of statistics using pandas

        :param dataframe_dict: Data names and values for selected data subset.

        :return stats_table: Table of stats: count, mean, std, min, max, etc.
        """
        dataframe = pd.DataFrame(dataframe_dict)
        if not dataframe.empty:
            stats_df = dataframe.describe(percentiles=None, include=None, exclude=None)
            stats_df.insert(0, "", stats_df.index)
            return stats_df.to_dict("records")
        else:
            return None

    @staticmethod
    def make_heatmap(dataframe_dict: dict) -> go.Figure:
        """
        Generate a confusion matrix.

        :param dataframe_dict: Data names and values for selected data subset.

        :return matrix: Confusion matrix for data subset.
        """
        dataframe = pd.DataFrame(dataframe_dict)
        corrs = dataframe.corr()

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
    def get_full_lists(
        data_subset: list,
        full_scales: dict,
        full_lower_bounds: dict,
        full_upper_bounds: dict,
    ) -> dict:
        """
        Loop through scales and bounds and convert to list to save in self.params.

        :param data_subset: Subset of data.
        :param full_scales: Dictionary of data and corresponding scaling factors.
        :param full_lower_bounds: Dictionary of data and corresponding lower bounds.
        :param full_upper_bounds: Dictionary of data and corresponding upper bounds.

        :return update_dict:
        """
        full_scales_list = []
        full_lower_bounds_list = []
        full_upper_bounds_list = []
        for data in data_subset:
            if data in full_scales:
                full_scales_list.append(full_scales[data])
            else:
                full_scales_list.append(1)
            if data in full_lower_bounds:
                full_lower_bounds_list.append(full_lower_bounds[data])
            else:
                full_lower_bounds_list.append(None)
            if data in full_upper_bounds:
                full_upper_bounds_list.append(full_upper_bounds[data])
            else:
                full_upper_bounds_list.append(None)

        return {
            "data_subset": str(data_subset),
            "full_scales": str(full_scales_list),
            "full_lower_bounds": str(full_lower_bounds_list),
            "full_upper_bounds": str(full_upper_bounds_list),
        }

    def trigger_click(  # pylint: disable=W0221
        self,
        n_clicks: int,
        monitoring_directory: str,
        live_link: list,
        n_clusters: int,
        objects: str,
        data_subset: list,
        color_pickers: list,
        downsampling: int,
        full_scales: dict,
        full_lower_bounds: dict,
        full_upper_bounds: dict,
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
        color_maps: str,
        size: str,
        size_log: list,
        size_thresh: float,
        size_min: float,
        size_max: float,
        size_markers: int,
        channel: str,
        ga_group_name: str,
        trigger: str = None,
    ) -> list:
        """
        Write cluster groups to the target geoh5 object. Inputs are all params that are written to ui.json.

        :param n_clicks: Trigger for export button.
        :param live_link: Checkbox for live link.
        :param n_clusters: Number of clusters.
        :param objects: Selected object uid.
        :param data_subset: Subset of data used for clustering and plotting.
        :param color_pickers: List of colors corresponding to clusters.
        :param downsampling: Percent downsampling.
        :param full_scales: Dictionary of data subset names and corresponding scaling factors.
        :param full_lower_bounds: Dictionary of data subset names and corresponding lower bounds.
        :param full_upper_bounds: Dictionary of data subset names and corresponding upper bounds.
        :param x: Selected x-axis data uid.
        :param x_log: Checkbox for plotting log on x-axis.
        :param x_thresh: X-axis threshold.
        :param x_min: X-axis minimum.
        :param x_max: X-axis maximum.
        :param y: Selected y-axis data uid.
        :param y_log: Checkbox for plotting log on y-axis.
        :param y_thresh: Y-axis threshold.
        :param y_min: Y-axis minimum.
        :param y_max: Y-axis maximum.
        :param z: Selected z-axis data uid.
        :param z_log: Checkbox for plotting log on z-axis.
        :param z_thresh: Z-axis threshold.
        :param z_min: Z-axis minimum.
        :param z_max: Z-axis maximum.
        :param color: Selected color-axis data uid.
        :param color_log: Checkbox for plotting log on color-axis.
        :param color_thresh: Color-axis threshold.
        :param color_min: Color-axis minimum.
        :param color_max: Color-axis maximum.
        :param color_maps: Color-axis color map.
        :param size: Selected size-axis data uid.
        :param size_log: Checkbox for plotting log on size-axis.
        :param size_thresh: Size-axis threshold.
        :param size_min: Size-axis minimum.
        :param size_max: Size-axis maximum.
        :param size_markers: Max size for markers.
        :param channel: Selected data for histogram/boxplot.
        :param ga_group_name: GA group name.
        :param monitoring_directory: Output path.
        :param trigger: Callback trigger.

        :return live_link: Checkbox value for dash live_link component.
        """
        if trigger is None:
            trigger = callback_context.triggered[0]["prop_id"].split(".")[0]
        if trigger == "export":
            if not live_link:
                live_link = False
            else:
                live_link = True

            param_dict = self.get_params_dict(locals())

            # Convert dicts to lists to save in the ui.json.
            param_dict.update(
                Clustering.get_full_lists(
                    data_subset, full_scales, full_lower_bounds, full_upper_bounds
                )
            )

            # Save axes that are plotting kmeans.
            plot_kmeans = []
            for axis in ["x", "y", "z", "color", "size"]:
                if locals()[axis] == "kmeans":
                    param_dict[axis] = None
                    plot_kmeans.append(True)
                else:
                    plot_kmeans.append(False)
            param_dict["plot_kmeans"] = str(plot_kmeans)

            # Get output path
            if (
                monitoring_directory is not None
                and monitoring_directory != ""
                and os.path.exists(os.path.abspath(monitoring_directory))
            ):
                monitoring_directory = os.path.abspath(monitoring_directory)
            else:
                monitoring_directory = os.path.dirname(self.workspace.h5file)

            # Get output workspace.
            temp_geoh5 = f"Clustering_{time.time():.0f}.geoh5"
            ws, live_link = BaseApplication.get_output_workspace(
                live_link, monitoring_directory, temp_geoh5
            )
            if not live_link:
                param_dict["monitoring_directory"] = ""

            with ws as workspace:
                # Put entities in output workspace.
                param_dict["geoh5"] = workspace
                for key, value in param_dict.items():
                    if isinstance(value, ObjectBase):
                        param_dict[key] = value.copy(
                            parent=workspace, copy_children=True
                        )
                # Write output uijson.
                new_params = ClusteringParams(**param_dict)
                new_params.write_input_file(
                    name=temp_geoh5.replace(".geoh5", ".ui.json"),
                    path=monitoring_directory,
                    validate=False,
                )
                # Run driver.
                self.driver.params = new_params
                self.driver.run()

            if live_link:
                print("Live link active. Check your ANALYST session for new mesh.")
                return [True]
            else:
                print("Saved to " + os.path.abspath(monitoring_directory))
                return []
        else:
            return no_update


if __name__ == "__main__":
    print("Loading geoh5 file . . .")
    file = sys.argv[1]
    ifile = InputFile.read_ui_json(file)
    ifile.workspace.open("r")
    app = Clustering(ui_json=ifile)
    print("Loaded. Building the clustering app . . .")
    app.run()
    print("Done")
