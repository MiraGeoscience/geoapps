#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

# pylint: disable=W0613

from __future__ import annotations

import os
import sys
import uuid
from time import time

import numpy as np
import plotly.graph_objects as go
from dash import callback_context, no_update
from dash.dependencies import Input, Output
from flask import Flask
from geoh5py.objects import ObjectBase
from geoh5py.ui_json import InputFile
from jupyter_dash import JupyterDash

from geoapps.base.application import BaseApplication
from geoapps.base.dash_application import BaseDashApplication
from geoapps.scatter_plot.constants import app_initializer
from geoapps.scatter_plot.driver import ScatterPlotDriver
from geoapps.scatter_plot.layout import scatter_layout
from geoapps.scatter_plot.params import ScatterPlotParams


class ScatterPlots(BaseDashApplication):
    """
    Dash app to make a scatter plot.
    """

    _param_class = ScatterPlotParams
    _driver_class = ScatterPlotDriver

    def __init__(self, ui_json=None, **kwargs):
        app_initializer.update(kwargs)
        if ui_json is not None and os.path.exists(ui_json.path):
            self.params = self._param_class(ui_json)
        else:
            self.params = self._param_class(**app_initializer)

        super().__init__()

        external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
        server = Flask(__name__)
        self.app = JupyterDash(
            server=server,
            url_base_pathname=os.environ.get("JUPYTERHUB_SERVICE_PREFIX", "/"),
            external_stylesheets=external_stylesheets,
        )

        self.app.layout = scatter_layout

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
            Output(component_id="objects", component_property="options"),
            Output(component_id="objects", component_property="value"),
            Output(component_id="ui_json_data", component_property="data"),
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
            Output(component_id="x", component_property="value"),
            Output(component_id="y", component_property="value"),
            Output(component_id="z", component_property="value"),
            Output(component_id="color", component_property="value"),
            Output(component_id="size", component_property="value"),
            Input(component_id="ui_json_data", component_property="data"),
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
            Input(component_id="ui_json_data", component_property="data"),
            Input(component_id="x", component_property="value"),
            Input(component_id="y", component_property="value"),
            Input(component_id="z", component_property="value"),
            Input(component_id="color", component_property="value"),
            Input(component_id="size", component_property="value"),
        )(self.update_channel_bounds)
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
            Output(component_id="monitoring_directory", component_property="value"),
            Input(component_id="ui_json_data", component_property="data"),
        )(self.update_remainder_from_ui_json)
        self.app.callback(
            Output(component_id="crossplot", component_property="figure"),
            Input(component_id="downsampling", component_property="value"),
            Input(component_id="objects", component_property="value"),
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
            Input(component_id="monitoring_directory", component_property="value"),
            Input(component_id="crossplot", component_property="figure"),
        )(self.trigger_click)

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

    def update_data_options(self, ui_json_data: dict, object_uid: str):
        """
        Get data dropdown options from a given object.

        :param ui_json_data: Uploaded ui.json data to read object from.
        :param object_uid: Selected object in object dropdown.

        :return options: Data dropdown options for x-axis of scatter plot.
        :return options: Data dropdown options for y-axis of scatter plot.
        :return options: Data dropdown options for z-axis of scatter plot.
        :return options: Data dropdown options for color-axis of scatter plot.
        :return options: Data dropdown options for size-axis of scatter plot.
        :return x_value: Data dropdown options for x-axis of scatter plot.
        :return y_value: Data dropdown options for y-axis of scatter plot.
        :return z_value: Data dropdown options for z-axis of scatter plot.
        :return color_value: Data dropdown options for color-axis of scatter plot.
        :return size_value: Data dropdown options for size-axis of scatter plot.
        """
        triggers = [c["prop_id"].split(".")[0] for c in callback_context.triggered]

        if "ui_json_data" in triggers:
            x_value = ui_json_data.get("x", None)
            y_value = ui_json_data.get("y", None)
            z_value = ui_json_data.get("z", None)
            color_value = ui_json_data.get("color", None)
            size_value = ui_json_data.get("size", None)
            trigger = "ui_json"
        else:
            x_value, y_value, z_value, color_value, size_value = (
                None,
                None,
                None,
                None,
                None,
            )
            trigger = "objects"

        options = self.get_data_options(trigger, ui_json_data, object_uid)

        return (
            options,
            options,
            options,
            options,
            options,
            x_value,
            y_value,
            z_value,
            color_value,
            size_value,
        )

    def get_channel_bounds(self, channel: str, kmeans: list = None) -> (float, float):
        """
        Set the min and max values for the given axis channel.

        :param channel: Name of channel to find data for.
        :param kmeans: Optional data to use instead of channel name.

        :return cmin: Minimum value for input channel.
        :return cmax: Maximum value for input channel.
        """
        data, cmin, cmax = None, None, None

        if channel == "kmeans" and kmeans is not None:
            data = kmeans
        elif (
            channel is not None
            and self.workspace.get_entity(uuid.UUID(channel))[0] is not None
        ):
            data = self.workspace.get_entity(uuid.UUID(channel))[0].values

        if data is not None:
            cmin = float(f"{np.nanmin(data):.2e}")
            cmax = float(f"{np.nanmax(data):.2e}")

        return cmin, cmax

    def update_channel_bounds(
        self,
        ui_json_data: dict,
        x: str,
        y: str,
        z: str,
        color: str,
        size: str,
        kmeans: list = None,
    ):
        """
        Update min and max for all channels, either from uploaded ui.json or from change of data.

        :param ui_json_data: Uploaded ui.json data.
        :param x: Name of selected x data.
        :param y: Name of selected y data.
        :param z: Name of selected z data.
        :param color: Name of selected color data.
        :param size: Name of selected size data.
        :param kmeans: Optional data to use instead of channel name.

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
        (
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
        ) = (
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
        )

        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]
        if trigger == "ui_json_data":
            x_min, x_max = ui_json_data.get("x_min", None), ui_json_data.get(
                "x_max", None
            )
            y_min, y_max = ui_json_data.get("y_min", None), ui_json_data.get(
                "y_max", None
            )
            z_min, z_max = ui_json_data.get("z_min", None), ui_json_data.get(
                "z_max", None
            )
            color_min, color_max = ui_json_data.get(
                "color_min", None
            ), ui_json_data.get("color_max", None)
            size_min, size_max = ui_json_data.get("size_min", None), ui_json_data.get(
                "size_max", None
            )

        elif trigger == "x":
            x_min, x_max = self.get_channel_bounds(x, kmeans)
        elif trigger == "y":
            y_min, y_max = self.get_channel_bounds(y, kmeans)
        elif trigger == "z":
            z_min, z_max = self.get_channel_bounds(z, kmeans)
        elif trigger == "color":
            color_min, color_max = self.get_channel_bounds(color, kmeans)
        elif trigger == "size":
            size_min, size_max = self.get_channel_bounds(size, kmeans)

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

    def update_plot(
        self,
        downsampling: int,
        objects: str,
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
    ) -> go.Figure:
        """
        Run scatter plot driver, and if export was clicked save the figure as html.

        :param downsampling: Percent of total values to plot.
        :param objects: UUID of selected object.
        :param x: UUID of selected x data.
        :param x_log: Whether or not to plot the log of x data.
        :param x_thresh: X threshold.
        :param x_min: Minimum value for x data.
        :param x_max: Maximum value for x data.
        :param y: UUID of selected y data.
        :param y_log: Whether or not to plot the log of y data.
        :param y_thresh: Y threshold.
        :param y_min: Minimum value for y data.
        :param y_max: Maximum value for y data.
        :param z: UUID of selected z data.
        :param z_log: Whether or not to plot the log of z data.
        :param z_thresh: Z threshold.
        :param z_min: Minimum value for z data.
        :param z_max: Maximum value for x data.
        :param color: UUID of selected color data.
        :param color_log: Whether or not to plot the log of color data.
        :param color_thresh: Color threshold.
        :param color_min: Minimum value for color data.
        :param color_max: Maximum value for color data.
        :param color_maps: Color map.
        :param size: UUID of selected size data.
        :param size_log: Whether or not to plot the log of size data.
        :param size_thresh: Size threshold.
        :param size_min: Minimum value for size data.
        :param size_max: Maximum value for size data.
        :param size_markers: Max size for markers.

        :return figure: Scatter plot.
        """
        update_dict = {}
        # Get list of parameters to update in self.params, from callback trigger.
        for item in callback_context.triggered:
            update_dict[item["prop_id"].split(".")[0]] = item["value"]

        # Don't update plot if objects triggered the callback, but use objects to update self.params.
        if "objects" in update_dict and len(update_dict) == 1:
            return go.Figure()
        elif set(update_dict.keys()).intersection({"x", "y", "z", "color", "size"}):
            update_dict.update({"objects": objects})

        # Update self.params
        param_dict = self.get_params_dict(update_dict)
        self.params.update(param_dict)
        # Run driver to get updated scatter plot.
        figure = go.Figure(self.driver.run())

        return figure

    def trigger_click(
        self, n_clicks: int, monitoring_directory: str, figure: go.Figure = None
    ):
        """
        Save the plot as html, write out ui.json.

        :param n_clicks: Trigger export from button.
        :param monitoring_directory: Output path.
        :param figure: Figure created by update_plots.
        """

        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]
        if trigger == "export":
            param_dict = self.params.to_dict()

            # Get output path.
            if (
                (monitoring_directory is not None)
                and (monitoring_directory != "")
                and (os.path.exists(os.path.abspath(monitoring_directory)))
            ):
                param_dict["monitoring_directory"] = os.path.abspath(
                    monitoring_directory
                )
                temp_geoh5 = f"Scatterplot_{time():.0f}.geoh5"

                # Get output workspace.
                ws, _ = BaseApplication.get_output_workspace(
                    False, param_dict["monitoring_directory"], temp_geoh5
                )

                with ws as new_workspace:
                    # Put entities in output workspace.
                    param_dict["geoh5"] = new_workspace
                    for key, value in param_dict.items():
                        if isinstance(value, ObjectBase):
                            param_dict[key] = value.copy(
                                parent=new_workspace, copy_children=True
                            )

                    # Write output uijson.
                    new_params = ScatterPlotParams(**param_dict)
                    new_params.write_input_file(
                        name=temp_geoh5.replace(".geoh5", ".ui.json"),
                        path=param_dict["monitoring_directory"],
                        validate=False,
                    )

                    go.Figure(figure).write_html(
                        os.path.join(
                            param_dict["monitoring_directory"],
                            temp_geoh5.replace(".geoh5", ".html"),
                        )
                    )
                print("Saved to " + param_dict["monitoring_directory"])
            else:
                print("Invalid output path.")

        return no_update


if __name__ == "__main__":
    print("Loading geoh5 file . . .")
    file = sys.argv[1]
    ifile = InputFile.read_ui_json(file)
    ifile.workspace.open("r")
    app = ScatterPlots(ui_json=ifile)
    print("Loaded. Building the plotly scatterplot . . .")
    app.run()
    print("Done")
