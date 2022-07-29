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

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import callback_context, no_update
from dash.dependencies import Input, Output
from flask import Flask
from geoh5py.objects import ObjectBase
from geoh5py.ui_json import InputFile
from jupyter_dash import JupyterDash

from geoapps.base.dash_application import BaseDashApplication
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

    def __init__(self, ui_json=None, **kwargs):
        app_initializer.update(kwargs)
        if ui_json is not None and os.path.exists(ui_json.path):
            self.params = self._param_class(ui_json)
        else:
            self.params = self._param_class(**app_initializer)

        super().__init__(**self.params.to_dict())

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
            Output(component_id="ui_json", component_property="data"),
            Output(component_id="objects", component_property="options"),
            Output(component_id="upload", component_property="filename"),
            Output(component_id="upload", component_property="contents"),
            Input(component_id="upload", component_property="filename"),
            Input(component_id="upload", component_property="contents"),
        )(self.update_object_options)
        self.app.callback(
            Output(component_id="data_subset", component_property="options"),
            Input(component_id="ui_json", component_property="data"),
            Input(component_id="objects", component_property="value"),
        )(self.update_data_subset)
        self.app.callback(
            Output(component_id="x", component_property="options"),
            Output(component_id="y", component_property="options"),
            Output(component_id="z", component_property="options"),
            Output(component_id="color", component_property="options"),
            Output(component_id="size", component_property="options"),
            Output(component_id="channel", component_property="options"),
            Output(component_id="color_maps", component_property="options"),
            Input(component_id="data_subset", component_property="value"),
            Input(component_id="kmeans", component_property="data"),
        )(Clustering.update_data_options)
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
            Input(component_id="kmeans", component_property="data"),
        )(self.update_channel_bounds)
        self.app.callback(
            Output(component_id="color_pickers", component_property="data"),
            Input(component_id="ui_json", component_property="data"),
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
            Output(component_id="scale", component_property="value"),
            Output(component_id="lower_bounds", component_property="value"),
            Output(component_id="upper_bounds", component_property="value"),
            Output(component_id="full_scales", component_property="data"),
            Output(component_id="full_lower_bounds", component_property="data"),
            Output(component_id="full_upper_bounds", component_property="data"),
            Input(component_id="ui_json", component_property="data"),
            Input(component_id="channel", component_property="value"),
            Input(component_id="full_scales", component_property="data"),
            Input(component_id="full_lower_bounds", component_property="data"),
            Input(component_id="full_upper_bounds", component_property="data"),
        )(self.update_properties)
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
            Output(component_id="color_maps", component_property="value"),
            Output(component_id="size", component_property="value"),
            Output(component_id="size_log", component_property="value"),
            Output(component_id="size_thresh", component_property="value"),
            Output(component_id="size_markers", component_property="value"),
            Output(component_id="data_subset", component_property="value"),
            Output(component_id="channel", component_property="value"),
            Output(component_id="n_clusters", component_property="value"),
            Output(component_id="plot_kmeans", component_property="data"),
            Output(component_id="output_path", component_property="value"),
            Input(component_id="ui_json", component_property="data"),
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
        )(self.make_scatter_plot)
        self.app.callback(
            Output(component_id="stats_table", component_property="data"),
            Input(component_id="dataframe", component_property="data"),
        )(self.make_stats_table)
        self.app.callback(
            Output(component_id="matrix", component_property="figure"),
            Input(component_id="dataframe", component_property="data"),
        )(self.make_heatmap)
        self.app.callback(
            Output(component_id="histogram", component_property="figure"),
            Input(component_id="dataframe", component_property="data"),
            Input(component_id="channel", component_property="value"),
            Input(component_id="lower_bounds", component_property="value"),
            Input(component_id="upper_bounds", component_property="value"),
        )(self.make_hist_plot)
        self.app.callback(
            Output(component_id="boxplot", component_property="figure"),
            Input(component_id="n_clusters", component_property="value"),
            Input(component_id="channel", component_property="value"),
            Input(component_id="color_pickers", component_property="data"),
            Input(component_id="kmeans", component_property="data"),
            Input(component_id="indices", component_property="data"),
        )(self.make_boxplot)
        self.app.callback(
            Output(component_id="inertia", component_property="figure"),
            Input(component_id="n_clusters", component_property="value"),
            Input(component_id="clusters", component_property="data"),
        )(self.make_inertia_plot)
        # Callback to export the clusters as a geoh5 file
        self.app.callback(
            Output(component_id="live_link", component_property="value"),
            Input(component_id="export", component_property="n_clicks"),
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
    def update_select_cluster_options(n_clusters, select_cluster):
        options = np.arange(0, n_clusters, 1)
        if select_cluster is None and len(options) > 0:
            select_cluster = options[0]
        return options, select_cluster

    @staticmethod
    def update_color_pickers(ui_json, color_pickers, color_picker, select_cluster):
        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]
        if trigger == "ui_json":
            if type(ui_json["color_pickers"]) == list:
                full_list = ui_json["color_pickers"]
            else:
                full_list = ast.literal_eval(ui_json["color_pickers"])
            if (full_list is None) | (not full_list):
                color_pickers = colors
            else:
                color_pickers = full_list
        elif trigger == "color_picker":
            color_pickers[select_cluster] = color_picker["hex"]
        return color_pickers

    @staticmethod
    def update_color_picker(color_pickers, select_cluster):
        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]
        if trigger == "select_cluster" and color_pickers:
            return dict(hex=color_pickers[select_cluster])
        else:
            return no_update

    def update_data_subset(self, ui_json, object_name):
        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]
        options = self.get_data_options(trigger, ui_json, object_name)

        return options

    @staticmethod
    def update_data_options(data_subset, kmeans):
        channel_options = data_subset

        if kmeans is not None:
            axis_options = data_subset + ["kmeans"]
            color_maps_options = px.colors.named_colorscales() + ["kmeans"]
        else:
            axis_options = data_subset
            color_maps_options = px.colors.named_colorscales()

        return (
            axis_options,
            axis_options,
            axis_options,
            axis_options,
            axis_options,
            channel_options,
            color_maps_options,
        )

    def update_properties(
        self,
        ui_json: dict,
        channel: str,
        full_scales: dict,
        full_lower_bounds: dict,
        full_upper_bounds: dict,
    ) -> tuple:
        """
        Get stored scale and bounds for a given channel. If there's no stored value, set a default.
        :param channel: Input data name for histogram, boxplot.
        :param full_scales: Dictionary of data names and the corresponding scales.
        :param full_lower_bounds: Dictionary of data names and the corresponding lower bounds.
        :param full_upper_bounds: Dictionary of data names and the corresponding upper bounds.
        :return update_dict: Dictionary of new scale and bounds.
        """

        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]
        if trigger == "ui_json":
            # Reconstruct scaling and bounds dicts from uijson input lists.
            data_subset = ast.literal_eval(ui_json["data_subset"])
            if len(data_subset) == 0:
                full_scales, full_lower_bounds, full_upper_bounds = {}, {}, {}
            else:
                full_dicts = []
                for key in ["full_scales", "full_lower_bounds", "full_upper_bounds"]:
                    out_dict = {}
                    full_list = ast.literal_eval(ui_json[key])
                    for i in range(len(data_subset)):
                        if (full_list is None) | (not full_list):
                            if key == "full_scales":
                                out_dict[data_subset[i]] = 1
                            else:
                                out_dict[data_subset[i]] = None
                        else:
                            out_dict[data_subset[i]] = full_list[i]
                    full_dicts.append(out_dict)
                full_scales, full_lower_bounds, full_upper_bounds = full_list
        if channel is not None:
            if channel not in full_scales:
                full_scales[channel] = 1
            scale = full_scales[channel]

            if (channel not in full_lower_bounds) or (
                full_lower_bounds[channel] is None
            ):
                full_lower_bounds[channel] = np.nanmin(
                    self.params.geoh5.get_entity(channel)[0].values
                )
            lower_bounds = float(full_lower_bounds[channel])

            if (channel not in full_upper_bounds) or (
                full_upper_bounds[channel] is None
            ):
                full_upper_bounds[channel] = np.nanmax(
                    self.params.geoh5.get_entity(channel)[0].values
                )
            upper_bounds = float(full_upper_bounds[channel])
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

    def update_remainder_from_ui_json(self, ui_json: dict) -> tuple:
        """
        Update parameters from uploaded ui_json that aren't involved in another callback.

        :param ui_json: Uploaded ui_json.

        :return outputs: List of outputs corresponding to the callback expected outputs.
        """

        # List of outputs for the callback
        output_ids = [
            item["id"] + "_" + item["property"]
            for item in callback_context.outputs_list
        ]

        update_dict = self.update_param_list_from_ui_json(ui_json, output_ids)

        # Add parameters that are specific to clustering.
        # Get initial data subset values. If the subset is empty, set it from the selected axis data.
        data_subset = ast.literal_eval(ui_json["data_subset"])
        if not data_subset:
            plot_data = [
                update_dict["x_value"],
                update_dict["y_value"],
                update_dict["z_value"],
                update_dict["color_value"],
                update_dict["size_value"],
            ]
            data_subset = list(filter(None, plot_data))
        update_dict.update({"data_subset_value": data_subset})
        update_dict.update({"channel_value": update_dict["x_value"]})

        outputs = BaseDashApplication.get_outputs(output_ids, update_dict)

        return outputs

    def update_dataframe(self, downsampling, data_subset):
        dataframe, mapping, indices = ClusteringDriver.update_dataframe(
            downsampling, data_subset, self.params.geoh5, downsample_min=5000
        )
        return dataframe, mapping, indices

    @staticmethod
    def run_clustering(dataframe_dict, n_clusters, full_scales, clusters, mapping):
        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]
        if trigger != "n_clusters" or clusters == {}:
            update_all_clusters = True
        else:
            update_all_clusters = False

        kmeans, clusters = ClusteringDriver.run_clustering(
            n_clusters,
            dataframe_dict,
            full_scales,
            clusters,
            mapping,
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
        color_maps: list,
        color_pickers: list,
        size_name: str,
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

        if not dataframe.empty:
            self.update_params(locals())

            if kmeans is not None:
                kmeans = np.array(kmeans)
            if indices is not None:
                indices = np.array(indices)

            if color_maps == "kmeans":
                # Update color_maps
                color_maps = Clustering.update_colormap(n_clusters, color_pickers)
            if color_maps is None:
                color_maps = [[0.0, "rgb(0,0,0)"]]

            # Input downsampled data to scatterplot so it doesn't regenerate data every time a parameter changes.
            axis_values = []
            for axis in [x_name, y_name, z_name, color_name, size_name]:
                if axis == "kmeans":
                    axis_values.append(PlotData(axis, kmeans[indices].astype(float)))
                elif axis is not None:
                    axis_values.append(PlotData(axis, dataframe[axis].values))
                else:
                    axis_values.append(None)

            x, y, z, color, size = tuple(axis_values)

            new_params = ClusteringParams(**self.params.to_dict())
            new_params.update(
                {
                    "downsampling": 100,
                    "color_maps": color_maps,
                    "x": x,
                    "y": y,
                    "z": z,
                    "color": color,
                    "size": size,
                },
                validate=False,
            )
            driver = ScatterPlotDriver(new_params)
            crossplot = go.Figure(driver.run())

            return crossplot
        else:

            return go.Figure()

    @staticmethod
    def make_inertia_plot(n_clusters: int, clusters: dict) -> go.Figure:
        """
        Generate an inertia plot.
        :param n_clusters: Number of clusters.
        :param clusters: K-means values for (2, 4, 8, 16, 32, n_clusters)
        :return inertia_plot: Inertia figure.
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
        dataframe_dict: dict, channel: str, lower_bounds: float, upper_bounds: float
    ) -> go.Figure:
        """
        Generate an histogram plot for the selected data channel.
        :param dataframe: Data names and values for the selected data subset.
        :param channel: Name of data plotted on histogram, boxplot.
        :param lower_bounds: Lower bounds for channel data.
        :param upper_bounds: Upper bounds for channel data.
        :return histogram: Histogram figure.
        """
        dataframe = pd.DataFrame(dataframe_dict)

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
        kmeans: list,
        indices: list,
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

        if (
            (kmeans is not None)
            and (channel is not None)
            and (indices is not None)
            and (color_pickers is not None)
        ):
            kmeans = np.array(kmeans)
            indices = np.array(indices)
            boxes = []
            for ii in range(n_clusters):
                cluster_ind = kmeans[indices] == ii
                x = np.ones(np.sum(cluster_ind)) * ii
                y = self.params.geoh5.get_entity(channel)[0].values[indices][
                    cluster_ind
                ]

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
    def make_stats_table(dataframe_dict: dict) -> list[dict]:
        """
        Generate a table of statistics using pandas
        :param dataframe: Data names and values for selected data subset.
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
    def make_heatmap(dataframe_dict: dict):
        """
        Generate a confusion matrix.
        :param dataframe: Data names and values for selected data subset.
        :return matrix: Confusion matrix figure.
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


if __name__ == "__main__":
    print("Loading geoh5 file . . .")
    file = sys.argv[1]
    ifile = InputFile.read_ui_json(file)
    app = Clustering(ui_json=ifile)
    print("Loaded. Building the clustering app . . .")
    app.run()
    print("Done")
