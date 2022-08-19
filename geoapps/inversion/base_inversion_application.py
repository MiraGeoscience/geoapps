#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


from __future__ import annotations

import os
import uuid
import warnings
import webbrowser

import matplotlib
import numpy as np
import scipy
import skimage
from dash import callback_context, no_update
from geoh5py.objects import Curve, Grid2D, Points, Surface
from geoh5py.shared.utils import is_uuid
from geoh5py.ui_json import InputFile
from matplotlib import colors
from matplotlib import pyplot as plt
from notebook import notebookapp
from plotly import graph_objects as go

from geoapps.base.application import BaseApplication
from geoapps.base.dash_application import BaseDashApplication
from geoapps.base.selection import TopographyOptions
from geoapps.inversion import InversionBaseParams
from geoapps.inversion.driver import InversionDriver
from geoapps.inversion.potential_fields.gravity.params import GravityParams
from geoapps.inversion.potential_fields.magnetic_scalar.params import (
    MagneticScalarParams,
)
from geoapps.inversion.potential_fields.magnetic_vector.constants import app_initializer
from geoapps.inversion.potential_fields.magnetic_vector.params import (
    MagneticVectorParams,
)
from geoapps.shared_utils.utils import downsample_grid, filter_xy


class InversionApp(BaseDashApplication):
    """
    Application for the inversion of potential field data using SimPEG
    """

    _param_class = InversionBaseParams
    # _driver_class = InversionDriver
    _inversion_type = None
    _inversion_params = None
    _run_params = None

    def __init__(self, ui_json=None, **kwargs):
        app_initializer.update(kwargs)
        if ui_json is not None and os.path.exists(ui_json.path):
            self.params = self._param_class(ui_json)
        else:
            self.params = self._param_class(**app_initializer)

        super().__init__()

    @staticmethod
    def update_uncertainty_visibility(selection):
        if selection == "Floor":
            return (
                {"display": "block"},
                {"display": "none"},
            )
        elif selection == "Channel":
            return (
                {"display": "none"},
                {"display": "block"},
            )

    @staticmethod
    def update_topography_visibility(selection):

        if selection == "None":
            return (
                {"display": "block"},
                {"display": "none"},
                {"display": "none"},
            )
        elif selection == "Object":
            return (
                {"display": "none"},
                {"display": "block"},
                {"display": "none"},
            )
        elif selection == "Constant":
            return (
                {"display": "none"},
                {"display": "none"},
                {"display": "block"},
            )
        else:
            return (no_update, no_update, no_update)

    def open_mesh_app(self, _):
        nb_port = None
        servers = list(notebookapp.list_running_servers())
        for s in servers:
            if s["notebook_dir"] == os.path.abspath("../../../"):
                nb_port = s["port"]
                break

        if nb_port is not None:
            # The reloader has not yet run - open the browser
            if not os.environ.get("WERKZEUG_RUN_MAIN"):
                webbrowser.open_new(
                    "http://localhost:"
                    + str(nb_port)
                    + "/notebooks/octree_creation/notebook.ipynb"
                )
        return 0

    @staticmethod
    def update_model_visibility(selection):
        if selection == "Constant":
            return {"display": "block"}, {"display": "none"}
        elif selection == "Model":
            return {"display": "none"}, {"display": "block"}
        elif selection == "None":
            return {"display": "none"}, {"display": "none"}

    @staticmethod
    def update_visibility_from_checkbox(selection):
        if selection:
            return {"display": "block"}
        else:
            return {"display": "none"}

    @staticmethod
    def unpack_val(val):
        if is_uuid(val):
            options = "Model"
            data = str(val)
            const = None
        elif (type(val) == float) or (type(val) == int):
            options = "Constant"
            data = None
            const = val
        else:
            options = "None"
            data = None
            const = None
        return options, data, const

    def update_inversion_params_from_ui_json(self, ui_json_data):
        options, const, obj, data = no_update, no_update, no_update, no_update

        prefix, param = tuple(
            callback_context.outputs_list[0]["id"]
            .removesuffix("_options")
            .split("_", 1)
        )

        if param in self._inversion_params:
            if prefix + "_" + param in ui_json_data:
                obj = str(ui_json_data[prefix + "_" + param + "_object"])
                val = ui_json_data[prefix + "_" + param]
            elif prefix + "_model" in ui_json_data:
                obj = str(ui_json_data[prefix + "_model_object"])
                val = ui_json_data[prefix + "_model"]
            else:
                val = None

            options, data, const = InversionApp.unpack_val(val)

        return options, const, obj, data

    @staticmethod
    def update_general_param_from_ui_json(ui_json_data):
        param = callback_context.outputs_list[0]["id"].removesuffix("_options")
        obj = str(ui_json_data[param + "_object"])
        val = ui_json_data[param]

        options, data, const = InversionApp.unpack_val(val)
        return options, const, obj, data

    # Update object dropdowns
    @staticmethod
    def update_remaining_object_options(obj_options):
        return obj_options

    # Update input data dropdown options

    def update_channel_options(
        self, ui_json_data: dict, object_uid: str
    ) -> (list, str):
        """
        Update data subset options and values from selected object.

        :param ui_json_data: Uploaded ui.json data.
        :param object_uid: Selected object from dropdown.

        :return options: Options for data subset dropdown.
        :return value: Value for data subset dropdown.
        """
        if object_uid is None or object_uid == "None":
            return no_update
        triggers = [c["prop_id"].split(".")[0] for c in callback_context.triggered]

        if "ui_json_data" in triggers:
            # value = ui_json["channel"]["value"]
            value = None
            options = self.get_data_options("ui_json_data", ui_json_data, object_uid)
        else:
            value = None
            options = self.get_data_options("data_object", ui_json_data, object_uid)

        return options

    def update_data_options(self, ui_json_data: dict, object_uid: str) -> (list, str):
        """
        Update data subset options and values from selected object.

        :param ui_json_data: Uploaded ui.json data.
        :param object_uid: Selected object from dropdown.

        :return options: Options for data subset dropdown.
        :return value: Value for data subset dropdown.
        """
        triggers = [c["prop_id"].split(".")[0] for c in callback_context.triggered]

        if "ui_json_data" in triggers:
            # value = ui_json["channel"]["value"]
            value = None
            options = self.get_data_options("ui_json_data", ui_json_data, object_uid)
        else:
            value = None
            options = self.get_data_options("data_object", ui_json_data, object_uid)

        return options, value

    @staticmethod
    def update_full_components(
        ui_json_data,
        full_components,
        channel_bool,
        channel,
        uncertainty_type,
        uncertainty_floor,
        uncertainty_channel,
        component,
        component_options,
    ):
        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]
        if trigger == "ui_json_data":
            full_components = {}
            for comp in component_options:
                # Get channel value
                if is_uuid(ui_json_data[comp + "_channel"]):
                    channel = str(ui_json_data[comp + "_channel"])
                else:
                    channel = None
                # Get channel_bool value
                if ui_json_data[comp + "_channel_bool"]:
                    channel_bool = [True]
                else:
                    channel_bool = []
                # Get uncertainty value
                if (type(ui_json_data[comp + "_uncertainty"]) == float) or (
                    type(ui_json_data[comp + "_uncertainty"]) == int
                ):
                    uncertainty_type = "Floor"
                    uncertainty_floor = ui_json_data[comp + "_uncertainty"]
                    uncertainty_channel = None
                elif is_uuid(ui_json_data[comp + "_uncertainty"]):
                    uncertainty_type = "Channel"
                    uncertainty_floor = None
                    uncertainty_channel = str(ui_json_data[comp + "_uncertainty"])

                full_components[comp] = {
                    "channel_bool": channel_bool,
                    "channel": channel,
                    "uncertainty_type": uncertainty_type,
                    "uncertainty_floor": uncertainty_floor,
                    "uncertainty_channel": uncertainty_channel,
                }
        else:
            full_components[component] = {
                "channel_bool": channel_bool,
                "channel": channel,
                "uncertainty_type": uncertainty_type,
                "uncertainty_floor": uncertainty_floor,
                "uncertainty_channel": uncertainty_channel,
            }
        return full_components

    @staticmethod
    def update_input_channel(component, full_components):
        if full_components and component is not None:
            channel_bool = full_components[component]["channel_bool"]
            channel = full_components[component]["channel"]
            uncertainty_type = full_components[component]["uncertainty_type"]
            uncertainty_floor = full_components[component]["uncertainty_floor"]
            uncertainty_channel = full_components[component]["uncertainty_channel"]
            return (
                channel_bool,
                channel,
                uncertainty_type,
                uncertainty_floor,
                uncertainty_channel,
            )
        else:
            return no_update, no_update, no_update, no_update, no_update

    @staticmethod
    def plot_plan_data_selection(entity, data, **kwargs):
        """
        Plot data values in 2D with contours

        :param entity: `geoh5py.objects`
            Input object with either `vertices` or `centroids` property.
        :param data: `geoh5py.data`
            Input data with `values` property.

        :return ax:
        :return out:
        :return indices:
        :return line_selection:
        :return contour_set:
        """
        indices = None
        values = None
        figure = None

        if isinstance(entity, (Grid2D, Points, Curve, Surface)):
            if "figure" not in kwargs.keys():
                figure = go.Figure()
            else:
                figure = kwargs["figure"]
        else:
            return figure, indices

        if getattr(entity, "vertices", None) is not None:
            locations = entity.vertices
        else:
            locations = entity.centroids

        if "resolution" not in kwargs.keys():
            resolution = 0
        else:
            resolution = kwargs["resolution"]

        if "indices" in kwargs.keys():
            indices = kwargs["indices"]
            if isinstance(indices, np.ndarray) and np.all(indices == False):
                indices = None

        if isinstance(getattr(data, "values", None), np.ndarray) and not isinstance(
            data.values[0], str
        ):
            values = np.asarray(data.values, dtype=float).copy()
            values[values == -99999] = np.nan
        elif isinstance(data, str) and (data in "XYZ"):
            values = locations[:, "XYZ".index(data)]

        if values is not None and (values.shape[0] != locations.shape[0]):
            values = None

        window = None
        if "window" in kwargs.keys():
            window = kwargs["window"]

        if (
            data is not None
            and getattr(data, "entity_type", None) is not None
            and getattr(data.entity_type, "color_map", None) is not None
        ):
            new_cmap = data.entity_type.color_map._values
            map_vals = new_cmap["Value"].copy()
            cmap = colors.ListedColormap(
                np.c_[
                    new_cmap["Red"] / 255,
                    new_cmap["Green"] / 255,
                    new_cmap["Blue"] / 255,
                ]
            )
            color_norm = colors.BoundaryNorm(map_vals, cmap.N)
        else:
            cmap = "Spectral_r"

        if isinstance(entity, Grid2D):
            x = entity.centroids[:, 0].reshape(entity.shape, order="F")
            y = entity.centroids[:, 1].reshape(entity.shape, order="F")

            rot = entity.rotation[0] + window["azimuth"]

            x_min = x.min()
            x_max = x.max()
            y_min = y.min()
            y_max = y.max()
            width = x_max - x_min
            height = y_max - y_min
            indices = filter_xy(x, y, resolution, window=window)

            """
            {
                "center": [x_min + (width/2), y_min + (height/2)],
                "size": [width, height],
                "azimuth": window["azimuth"]
            }
            """

            ind_x, ind_y = (
                np.any(indices, axis=1),
                np.any(indices, axis=0),
            )

            if values is not None:
                # values[np.isnan(values)] = None
                values = np.asarray(
                    values.reshape(entity.shape, order="F"), dtype=float
                )
                # values[indices == False] = np.nan
                # values = values[ind_x, :][:, ind_y]

                # """
                new_values = scipy.ndimage.rotate(values, rot, cval=np.nan)
                rot_x = np.linspace(x.min(), x.max(), new_values.shape[0])
                rot_y = np.linspace(y.min(), y.max(), new_values.shape[1])

                # downsampled_index, down_x, down_y = downsample_grid(rot_x, rot_y, resolution)"""

                downsampled = skimage.measure.block_reduce(
                    new_values, (resolution, resolution)
                )

            if np.any(values):
                """
                figure["data"][0]["x"] = np.linspace(
                    x.min(), x.max(), values.shape[0]
                )
                figure["data"][0]["y"] = np.linspace(
                    y.min(), y.max(), values.shape[1]
                )"""
                figure["data"][0]["z"] = values.T
                # figure["data"][0]["x"] = down_x
                # figure["data"][0]["y"] = down_y
                # figure["data"][0]["z"] = new_values.T[downsampled_index]  # new_values
                figure["data"][0]["x"] = rot_x
                figure["data"][0]["y"] = rot_y
                figure["data"][0]["z"] = downsampled.T  # new_values

        else:
            x, y = entity.vertices[:, 0], entity.vertices[:, 1]
            if indices is None:
                indices = filter_xy(
                    x,
                    y,
                    resolution,
                    window=window,
                )
            X, Y = x[indices], y[indices]

            if values is not None:
                values = values[indices]

            if "marker_size" not in kwargs.keys():
                marker_size = 50
            else:
                marker_size = kwargs["marker_size"]

            out = axis.scatter(X, Y, marker_size, values, cmap=cmap, norm=color_norm)

            if (
                "contours" in kwargs.keys()
                and kwargs["contours"] is not None
                and np.any(values)
            ):
                ind = ~np.isnan(values)
                contour_set = axis.tricontour(
                    X[ind],
                    Y[ind],
                    values[ind],
                    levels=kwargs["contours"],
                    colors="k",
                    linewidths=1.0,
                )

        if np.any(x) and np.any(y):
            figure.update_layout(plot_bgcolor="rgba(0,0,0,0)")
            figure.update_layout(xaxis_title="Easting (m)", yaxis_title="Northing (m)")
            # figure.update_layout(
            #    xaxis_range=[window["center"][0] - (window["size"][0]/2), window["center"][0] + (window["size"][0]/2)],
            #    yaxis_range=[window["center"][1] - (window["size"][1]/2), window["center"][1] + (window["size"][1]/2)]
            # )

        if "fix_aspect_ratio" in kwargs.keys():
            if kwargs["fix_aspect_ratio"]:
                figure.update_layout(yaxis=dict(scaleanchor="x"))
            else:
                figure.update_layout(yaxis=dict(scaleanchor=None))

        if "colorbar" in kwargs.keys():
            if kwargs["colorbar"]:
                figure.update_traces(showscale=True)
            else:
                figure.update_traces(showscale=False)

        return figure, indices

    def plot_selection(
        self,
        ui_json_data,
        figure,
        object,
        data,
        resolution,
        azimuth,
        colorbar,
        fix_aspect_ratio,
    ):
        triggers = [c["prop_id"].split(".")[0] for c in callback_context.triggered]

        figure = go.Figure(figure)
        data_count = "Data Count: "

        """
        if "ui_json_data" in triggers:
            center_x = ui_json_data["window_center_x"]
            center_y = ui_json_data["window_center_y"]
            width = ui_json_data["window_width"]
            height = ui_json_data["window_height"]
            figure.update_layout(
                xaxis_range=[center_x - (width / 2), center_x + (width / 2)],
                yaxis_range=[center_y - (height / 2), center_y + (height / 2)],
            )
        else:
            x_range = figure["layout"]["xaxis"]["range"]
            width = x_range[1] - x_range[0]
            center_x = x_range[0] + (width / 2)
            y_range = figure["layout"]["yaxis"]["range"]
            height = y_range[1] - y_range[0]
            center_y = y_range[0] + (height / 2)
        """

        if object is not None and data is not None:
            obj = self.workspace.get_entity(uuid.UUID(object))[0]

            data_obj = self.workspace.get_entity(uuid.UUID(data))[0]

            lim_x = [1e8, -1e8]
            lim_y = [1e8, -1e8]

            if isinstance(obj, Grid2D):
                lim_x[0], lim_x[1] = (
                    obj.centroids[:, 0].min(),
                    obj.centroids[:, 0].max(),
                )
                lim_y[0], lim_y[1] = (
                    obj.centroids[:, 1].min(),
                    obj.centroids[:, 1].max(),
                )
            elif isinstance(obj, (Points, Curve, Surface)):
                lim_x[0], lim_x[1] = obj.vertices[:, 0].min(), obj.vertices[:, 0].max()
                lim_y[0], lim_y[1] = obj.vertices[:, 1].min(), obj.vertices[:, 1].max()
            else:
                return

            width = lim_x[1] - lim_x[0]
            height = lim_y[1] - lim_y[0]
            center_x = np.mean(lim_x)
            center_y = np.mean(lim_y)

            if isinstance(obj, (Grid2D, Surface, Points, Curve)):

                figure, ind_filter = InversionApp.plot_plan_data_selection(
                    obj,
                    data_obj,
                    **{
                        "figure": figure,
                        "resolution": resolution,
                        "window": {
                            "center": [center_x, center_y],
                            "size": [width, height],
                            "azimuth": azimuth,
                        },
                        # "resize": True,
                        "colorbar": colorbar,
                        "fix_aspect_ratio": fix_aspect_ratio,
                    },
                )

                if "ui_json_data" in triggers:
                    center_x = ui_json_data["window_center_x"]
                    center_y = ui_json_data["window_center_y"]
                    width = ui_json_data["window_width"]
                    height = ui_json_data["window_height"]
                    figure.update_layout(
                        xaxis_range=[center_x - (width / 2), center_x + (width / 2)],
                        yaxis_range=[center_y - (height / 2), center_y + (height / 2)],
                    )

                # axes range is wrong
                x = np.array(figure["data"][0]["x"])
                x_range = figure["layout"]["xaxis"]["range"]
                x_indices = (x_range[0] < x) & (x < x_range[1])

                y = np.array(figure["data"][0]["y"])
                y_range = figure["layout"]["yaxis"]["range"]
                y_indices = (y_range[0] < y) & (y < y_range[1])

                z = np.array(figure["data"][0]["z"])
                # z = z[y_indices, x_indices]
                # z_points = np.sum(~np.isnan(z))

                # indices = filter_xy(x, y, resolution, window=window)

                # data_count += f"{np.sum(ind_filter)}"
                data_count += f"{0}"

        return (
            figure,
            data_count,
        )

    def trigger_click(self, _):
        """"""
        if self._run_params is None:
            warnings.warn("Input file must be written before running.")
            return

        self.run_inversion(self._run_params)

    @staticmethod
    def run_inversion(params):
        """
        Trigger the inversion.
        """
        os.system(
            "start cmd.exe @cmd /k "
            + f"python -m {params.run_command} "
            + f'"{params.input_file.path_name}"'
        )
