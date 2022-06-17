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
import plotly.express as px
import plotly.graph_objects as go
from dash import callback_context, dash_table, dcc, html, no_update
from dash.dependencies import Input, Output
from flask import Flask
from geoh5py.objects.object_base import ObjectBase
from geoh5py.ui_json import InputFile
from geoh5py.workspace import Workspace
from jupyter_dash import JupyterDash
from scipy.spatial import cKDTree

from geoapps.clustering.constants import app_initializer
from geoapps.clustering.driver import ClusteringDriver
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

        self.data_channels = {}
        self.scalings = {}
        self.lower_bounds = {}
        self.upper_bounds = {}
        self.indices = []
        # Initial values for the dash components
        defaults = self.get_defaults()
        super().__init__(**defaults)

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
                    id="tabs",
                    value="crossplot",
                    children=[
                        dcc.Tab(label="Crossplot", value="crossplot_tab"),
                        dcc.Tab(label="Statistics", value="statistics_tab"),
                        dcc.Tab(label="Confusion Matrix", value="matrix_tab"),
                        dcc.Tab(label="Histogram", value="histogram_tab"),
                        dcc.Tab(label="Boxplot", value="boxplot_tab"),
                        dcc.Tab(label="Inertia", value="inertia_tab"),
                    ],
                ),
                html.Div(id="tabs-content"),
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
                            multi=True,
                        ),
                    ]
                ),
                html.Div(
                    [
                        dcc.Markdown("Number of clusters"),
                        dcc.Slider(
                            id="clusters",
                            min=2,
                            max=100,
                            step=1,
                            value=8,
                        ),
                        dcc.Markdown("Cluster"),
                        dcc.Dropdown(id="select_cluster", options=np.arange(0, 101, 1)),
                        dcc.Markdown("Color"),
                    ]
                ),
                self.tabs_layout,
                dcc.Store(id="dataframe"),
            ]
        )

        # Callbacks relating to layout
        self.app.callback(
            Output(component_id="tabs-content", component_property="children"),
            Input(component_id="tabs", component_property="value"),
        )(self.render_content)
        self.app.callback(
            Output(component_id="x_div", component_property="style"),
            Output(component_id="y_div", component_property="style"),
            Output(component_id="z_div", component_property="style"),
            Output(component_id="color_div", component_property="style"),
            Output(component_id="size_div", component_property="style"),
            Input(component_id="axes_pannels", component_property="value"),
        )(self.update_visibility)
        #
        self.app.callback(
            Output(component_id="dataframe", component_property="value"),
            Input(component_id="downsampling", component_property="value"),
            # Input(component_id='x', component_property='value'),
            # Input(component_id='y', component_property='value'),
            # Input(component_id='z', component_property='value'),
            # Input(component_id='color', component_property='value'),
            # Input(component_id='size', component_property='value'),
            Input(component_id="channel", component_property="value"),
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
            Input(component_id="clusters", component_property="value"),
            Input(component_id="dataframe", component_property="value"),
        )(self.update_plots)
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
            Output(component_id="channel", component_property="options"),
            Output(component_id="channels", component_property="value"),
            Input(component_id="objects", component_property="value"),
        )(self.update_data_options)
        self.app.callback(
            Output(component_id="scale", component_property="value"),
            Output(component_id="lower_bounds", component_property="value"),
            Output(component_id="upper_bounds", component_property="value"),
            Input(component_id="channel", component_property="value"),
        )(self.update_properties)

    def render_content(self, tab):
        if tab == "crossplot_tab":
            return self.plot_layout
        elif tab == "statistics_tab":
            return html.Div([dash_table.DataTable(id="stats_table")])
        elif tab == "matrix_tab":
            return html.Div(
                [
                    dcc.Dropdown(id="surface_type", options=["Heatmap", "3D Surface"]),
                    dcc.Dropdown(
                        id="colormap",
                        options=["Viridis", "Rainbow", "Cividis", "Blues", "Greens"],
                    ),
                    dcc.Graph(
                        id="matrix",
                    ),
                ]
            )
        elif tab == "histogram_tab":
            return html.Div(
                [
                    dcc.Dropdown(
                        id="channel",
                    ),
                    dcc.Slider(
                        id="scale",
                        min=1,
                        max=10,
                        step=1,
                    ),
                    dcc.Markdown("Lower bound"),
                    dcc.Input(id="lower_bounds"),
                    dcc.Markdown("Upper bound"),
                    dcc.Input(id="upper_bounds"),
                    dcc.Graph(
                        id="histogram",
                    ),
                ]
            )
        elif tab == "boxplot_tab":
            dcc.Dropdown(
                id="channel",
            ),
            return html.Div(
                [
                    dcc.Graph(
                        id="boxplot",
                    )
                ]
            )
        elif tab == "inertia_tab":
            return html.Div(
                [
                    dcc.Graph(
                        id="inertia",
                    )
                ]
            )

    def update_channels(self, channels):
        self.data_channels = {}
        for channel in channels:
            self.get_channel(channel)

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

    def update_plots(self, clusters, dataframe):
        pass

    def update_dataframe(
        self, downsampling, channel, channels, scale, lower_bounds, upper_bounds
    ):
        """
        Normalize the the selected data and perform the kmeans clustering.
        """
        # convert downsampling from percent to number***
        fields = channels
        if len(fields) > 0:
            values = []
            for field in fields:
                vals = self.data_channels[field].copy()
                nns = ~np.isnan(vals)
                if field not in self.scalings.keys():
                    self.scalings[field] = scale
                if field not in self.lower_bounds.keys():
                    self.lower_bounds[field] = scale
                if field not in self.upper_bounds.keys():
                    self.upper_bounds[field] = scale

                vals[
                    (vals < self.lower_bounds[field].value)
                    | (vals > self.upper_bounds[field].value)
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
                columns=[self.data.uid_name_map[field] for field in fields],
            )
            tree = cKDTree(dataframe.values)
            inactive_set = np.ones(self.n_values, dtype="bool")
            inactive_set[self.indices] = False
            out_values = values[inactive_set, :]
            for ii in range(values.shape[1]):
                out_values[np.isnan(out_values[:, ii]), ii] = np.mean(
                    values[self.indices, ii]
                )

            _, ind_out = tree.query(out_values)
            del tree

            # self._mapping = np.empty(self.n_values, dtype="int")
            # self._mapping[inactive_set] = ind_out
            # self._mapping[self.indices] = np.arange(self.indices.shape[0])
            # self._inactive_set = np.where(np.all(np.isnan(values), axis=1))[0]
            # options = [[self.data.uid_name_map[key], key] for key in fields]
            # self.channels_plot_options.options = options

            return dataframe

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


if __name__ == "__main__":
    print("Loading geoh5 file . . .")
    file = sys.argv[1]
    ifile = InputFile.read_ui_json(file)
    app = Clustering(ui_json=ifile)
    print("Loaded. Building the clustering app . . .")
    app.run()
    print("Done")
