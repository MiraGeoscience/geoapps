#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import os
import sys
import webbrowser
from os import environ

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dash_table, dcc, html
from dash.dependencies import Input, Output
from flask import Flask
from geoh5py.ui_json import InputFile
from jupyter_dash import JupyterDash
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

from geoapps.clustering.constants import app_initializer
from geoapps.clustering.params import ClusteringParams
from geoapps.scatter_plot.application import ScatterPlots
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

        self.tabs_layout = html.Div(
            [
                dcc.Tabs(
                    [
                        dcc.Tab(
                            label="Crossplot",
                            children=[html.Div([self.axis_layout, self.plot_layout])],
                        ),
                        dcc.Tab(
                            label="Statistics",
                            children=[
                                html.Div([dash_table.DataTable(id="stats_table")])
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
                                    ]
                                )
                            ],
                        ),
                        dcc.Tab(
                            label="Histogram",
                            children=[
                                html.Div(
                                    [
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
                                        dcc.Markdown("Scale"),
                                        dcc.Slider(
                                            id="scale",
                                            min=1,
                                            max=10,
                                            step=1,
                                            value=self.defaults["scale"],
                                        ),
                                        dcc.Markdown("Lower bound"),
                                        dcc.Input(
                                            id="lower_bounds",
                                            value=self.defaults["lower_bounds"],
                                        ),
                                        dcc.Markdown("Upper bound"),
                                        dcc.Input(
                                            id="upper_bounds",
                                            value=self.defaults["upper_bounds"],
                                        ),
                                        dcc.Graph(
                                            id="histogram",
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
                                    ]
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
                                    ]
                                )
                            ],
                        ),
                    ]
                )
            ]
        )

        self.app.layout = html.Div(
            [
                self.workspace_layout,
                html.Div(
                    [
                        dcc.Markdown("Data: "),
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
                    style={"width": "50%"},
                ),
                html.Div(
                    [
                        dcc.Markdown("Number of clusters"),
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
                        dcc.Markdown("Cluster"),
                        dcc.Dropdown(id="select_cluster", options=np.arange(0, 101, 1)),
                        dcc.Markdown("Color"),
                    ],
                    style={"width": "50%"},
                ),
                self.tabs_layout,
                dcc.Store(id="dataframe", data={}),
            ]
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
            Output(component_id="x", component_property="options"),
            Output(component_id="y", component_property="options"),
            Output(component_id="z", component_property="options"),
            Output(component_id="color", component_property="options"),
            Output(component_id="size", component_property="options"),
            Output(component_id="channel", component_property="options"),
            Input(component_id="channels", component_property="value"),
        )(self.update_channels)
        self.app.callback(
            Output(component_id="dataframe", component_property="data"),
            Input(component_id="downsampling", component_property="value"),
            Input(component_id="channels", component_property="value"),
            Input(component_id="scale", component_property="value"),
            Input(component_id="lower_bounds", component_property="value"),
            Input(component_id="upper_bounds", component_property="value"),
        )(self.update_dataframe)
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
            Input(component_id="size", component_property="value"),
            Input(component_id="size_log", component_property="value"),
            Input(component_id="size_thresh", component_property="value"),
            Input(component_id="size_min", component_property="value"),
            Input(component_id="size_max", component_property="value"),
            Input(component_id="size_markers", component_property="value"),
        )(self.update_plots)
        """self.app.callback(
            Output(component_id="scale", component_property="value"),
            Output(component_id="lower_bounds", component_property="value"),
            Output(component_id="upper_bounds", component_property="value"),
            Input(component_id="channel", component_property="value"),
        )(self.update_properties)"""

        """self.app.callback(
            Output(component_id="download", component_property="href"),
            Input(component_id="plot", component_property="figure"),
        )(self.save_figure)"""

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
        return defaults

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
        if channel in self.scalings.keys():
            scale = self.scalings[channel]
        else:
            scale = 1

        if channel in self.lower_bounds.keys():
            lower_bounds = self.lower_bounds[channel]
        else:
            lower_bounds = None  # get default value

        if channel in self.upper_bounds.keys():
            upper_bounds = self.upper_bounds[channel]
        else:
            upper_bounds = None  # get default value

        return scale, lower_bounds, upper_bounds

    def update_plots(
        self,
        n_clusters,
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
        size,
        size_log,
        size_thresh,
        size_min,
        size_max,
        size_markers,
    ):
        dataframe = pd.DataFrame(dataframe_dict["dataframe"])
        self.run_clustering(n_clusters, dataframe)

        color_maps = None

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
            if val not in self.clusters.keys():
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

            """if getattr(self, "inertia_plot", None) is None:
                inertia_plot = go.Figure([line, point])
            else:
                inertia_plot.data = []
                inertia_plot.add_trace(line)
                inertia_plot.add_trace(point)"""
            inertia_plot.update_layout(
                {
                    "height": 300,
                    "width": 400,
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
                        fillcolor="blue",  # self.color_pickers[ii].value,
                        marker_color="blue",  # self.color_pickers[ii].value,
                        line_color="blue",  # self.color_pickers[ii].value,
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
            return dataframe.describe(percentiles=None, include=None, exclude=None)
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
        self, downsampling, channels, scale, lower_bounds, upper_bounds
    ):
        """
        Normalize the the selected data and perform the kmeans clustering.
        """
        self.kmeans = None
        # convert downsampling from percent to number***
        fields = channels
        if len(fields) > 0:
            values = []
            n_values = 0
            for field in fields:
                vals = self.data_channels[field].values.copy()
                n_values = len(vals)
                # nns = ~np.isnan(vals)
                if field not in self.scalings.keys():
                    self.scalings[field] = scale
                if field not in self.lower_bounds.keys():
                    self.lower_bounds[field] = lower_bounds
                if field not in self.upper_bounds.keys():
                    self.upper_bounds[field] = upper_bounds

                vals[
                    (vals < self.lower_bounds[field])
                    | (vals > self.upper_bounds[field])
                ] = np.nan
                values += [vals]

            values = np.vstack(values).T
            active_set = np.where(np.all(~np.isnan(values), axis=1))[0]

            if len(active_set) == 0:
                print("No rows were found without no-data-values. Check input field")
                return None

            samples = random_sampling(
                values[active_set, :],
                np.min([downsampling, len(active_set)]),
                bandwidth=2.0,
                rtol=1e0,
                method="histogram",
            )
            self.indices = active_set[samples]
            dataframe = pd.DataFrame(
                values[self.indices, :],
                columns=fields,
            )

            tree = cKDTree(dataframe.values)
            inactive_set = np.ones(n_values, dtype="bool")
            inactive_set[self.indices] = False
            out_values = values[inactive_set, :]
            for ii in range(values.shape[1]):
                out_values[np.isnan(out_values[:, ii]), ii] = np.mean(
                    values[self.indices, ii]
                )

            _, ind_out = tree.query(out_values)
            del tree

            self.mapping = np.empty(n_values, dtype="int")
            self.mapping[inactive_set] = ind_out
            self.mapping[self.indices] = np.arange(self.indices.shape[0])
            # self._inactive_set = np.where(np.all(np.isnan(values), axis=1))[0]
            # options = [[self.data.uid_name_map[key], key] for key in fields]
            # self.channels_plot_options.options = options
            return {"dataframe": dataframe.to_dict("records")}

        else:
            """
            self.dataframe = None
            self.dataframe_scaled = None
            self._mapping = None
            self._inactive_set = None
            """
            return None

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
