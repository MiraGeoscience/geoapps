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
from dash import callback_context, no_update
from geoh5py.objects import Curve, Grid2D, Points, Surface
from geoh5py.shared.utils import is_uuid
from geoh5py.ui_json import InputFile
from matplotlib import colors
from notebook import notebookapp
from plotly import graph_objects as go

from geoapps.base.application import BaseApplication
from geoapps.base.dash_application import BaseDashApplication
from geoapps.base.selection import TopographyOptions
from geoapps.inversion.potential_fields.gravity.params import GravityParams
from geoapps.inversion.potential_fields.magnetic_scalar.params import (
    MagneticScalarParams,
)
from geoapps.inversion.potential_fields.magnetic_vector.constants import app_initializer
from geoapps.inversion.potential_fields.magnetic_vector.params import (
    MagneticVectorParams,
)
from geoapps.shared_utils.utils import filter_xy


class InversionApp(BaseDashApplication):
    """
    Application for the inversion of potential field data using SimPEG
    """

    _param_class = None
    _inversion_type = None
    _inversion_params = None
    _run_params = None

    def __init__(self, ui_json=None, **kwargs):
        app_initializer.update(kwargs)
        if ui_json is not None and os.path.exists(ui_json.path):
            self.params = self._param_class(ui_json)
        else:
            self.params = self._param_class(**app_initializer)

        super().__init__(**self.params.to_dict())

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

    def update_inversion_params_from_ui_json(self, ui_json):
        options, const, obj, data = no_update, no_update, no_update, no_update

        prefix, param = tuple(
            callback_context.outputs_list[0]["id"]
            .removesuffix("_options")
            .split("_", 1)
        )

        if param in self._inversion_params:
            if prefix + "_" + param in ui_json:
                obj = str(ui_json[prefix + "_" + param + "_object"]["value"])
                val = ui_json[prefix + "_" + param]["value"]
            elif prefix + "_model" in ui_json:
                obj = str(ui_json[prefix + "_model_object"]["value"])
                val = ui_json[prefix + "_model"]["value"]
            else:
                val = None

            options, data, const = InversionApp.unpack_val(val)

        return options, const, obj, data

    @staticmethod
    def update_general_param_from_ui_json(ui_json):
        param = callback_context.outputs_list[0]["id"].removesuffix("_options")
        obj = str(ui_json[param + "_object"]["value"])
        val = ui_json[param]["value"]

        options, data, const = InversionApp.unpack_val(val)
        return options, const, obj, data

    # Update object dropdowns
    @staticmethod
    def update_remaining_object_options(obj_options):
        return obj_options

    # Update input data dropdown options

    def update_channel_options(self, ui_json: dict, object_uid: str) -> (list, str):
        """
        Update data subset options and values from selected object.

        :param ui_json: Uploaded ui.json.
        :param object_uid: Selected object from dropdown.

        :return options: Options for data subset dropdown.
        :return value: Value for data subset dropdown.
        """
        if object_uid is None or object_uid == "None":
            return no_update
        triggers = [c["prop_id"].split(".")[0] for c in callback_context.triggered]

        if "ui_json" in triggers:
            # value = ui_json["channel"]["value"]
            value = None
            options = self.get_data_options("ui_json", ui_json, object_uid)
        else:
            value = None
            options = self.get_data_options("data_object", ui_json, object_uid)

        return options

    def update_data_options(self, ui_json: dict, object_uid: str) -> (list, str):
        """
        Update data subset options and values from selected object.

        :param ui_json: Uploaded ui.json.
        :param object_uid: Selected object from dropdown.

        :return options: Options for data subset dropdown.
        :return value: Value for data subset dropdown.
        """
        triggers = [c["prop_id"].split(".")[0] for c in callback_context.triggered]

        if "ui_json" in triggers:
            # value = ui_json["channel"]["value"]
            value = None
            options = self.get_data_options("ui_json", ui_json, object_uid)
        else:
            value = None
            options = self.get_data_options("data_object", ui_json, object_uid)

        return options, value

    @staticmethod
    def update_full_components(
        ui_json,
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
        if trigger == "ui_json":
            full_components = {}
            for comp in component_options:
                # Get channel value
                if is_uuid(ui_json[comp + "_channel"]["value"]):
                    channel = str(ui_json[comp + "_channel"]["value"])
                else:
                    channel = None
                # Get channel_bool value
                if ui_json[comp + "_channel_bool"]:
                    channel_bool = [True]
                else:
                    channel_bool = []
                # Get uncertainty value
                if (type(ui_json[comp + "_uncertainty"]["value"]) == float) or (
                    type(ui_json[comp + "_uncertainty"]["value"]) == int
                ):
                    uncertainty_type = "Floor"
                    uncertainty_floor = ui_json[comp + "_uncertainty"]["value"]
                    uncertainty_channel = None
                elif is_uuid(ui_json[comp + "_uncertainty"]["value"]):
                    uncertainty_type = "Channel"
                    uncertainty_floor = None
                    uncertainty_channel = str(ui_json[comp + "_uncertainty"]["value"])

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
        line_selection = None
        contour_set = None
        values = None
        figure = None
        out = None

        if isinstance(entity, (Grid2D, Points, Curve, Surface)):
            if "figure" not in kwargs.keys():
                figure = go.Figure()
            else:
                figure = kwargs["figure"]
        else:
            return figure, out, indices, line_selection, contour_set

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

        color_norm = None
        if "color_norm" in kwargs.keys():
            color_norm = kwargs["color_norm"]

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
            indices = filter_xy(x, y, resolution, window=window)

            ind_x, ind_y = (
                np.any(indices, axis=1),
                np.any(indices, axis=0),
            )

            """
            if window is not None:
                x_min = window["center_x"] - (window["width"]/2)
                x_max = window["center_x"] + (window["width"]/2)
                y_min = window["center_y"] - (window["height"]/2)
                y_max = window["center_y"] + (window["height"]/2)
            else:
                x_min = x.min()
                x_max = x.max()
                y_min = y.min()
                y_max = y.max()

            plot_x = np.arange(x_min, x_max + resolution, resolution)
            plot_y = np.arange(y_min, y_max + resolution, resolution)
            """

            if values is not None:
                values = np.asarray(
                    values.reshape(entity.shape, order="F"), dtype=float
                )
                values[indices == False] = np.nan
                values = values[ind_x, :][:, ind_y]

            if np.any(values):
                figure.add_trace(
                    go.Heatmap(
                        x=x[ind_x, :][0],
                        y=y[:, ind_y][1],
                        z=values.T,
                    )
                )

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

        if "fix_aspect_ratio" in kwargs.keys():
            if kwargs["fix_aspect_ratio"]:
                # figure.update_layout(scene=dict(aspectmode="data"))
                figure.update_layout(yaxis=dict(scaleanchor="x"))
            else:
                # figure.update_layout(scene=dict(aspectmode=None))
                figure.update_layout(yaxis=dict(scaleanchor=None))

        if "colorbar" in kwargs.keys():
            if kwargs["colorbar"]:
                figure.update_traces(showscale=True)
            else:
                figure.update_traces(showscale=False)

        return figure, out, indices, line_selection, contour_set

    def update_window_params(self, ui_json, figure, channel):
        print("test")
        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]
        print(trigger)

        if trigger == "plot":
            if channel is None:
                return (
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                )
            else:
                x_min = figure["layout"]["xaxis"]["range"][0]
                x_max = figure["layout"]["xaxis"]["range"][1]
                y_min = figure["layout"]["yaxis"]["range"][0]
                y_max = figure["layout"]["yaxis"]["range"][1]

                width = x_max - x_min
                height = y_max - y_min
                center_x = x_min + (width / 2)
                center_y = y_min + (height / 2)
        else:
            center_x = ui_json["window_center_x"]["value"]
            center_y = ui_json["window_center_y"]["value"]
            width = ui_json["window_width"]["value"]
            height = ui_json["window_height"]["value"]
            print(center_x)

        return (
            center_x,
            center_y,
            width,
            height,
        )

    def plot_selection(
        self,
        object,
        data,
        center_x,
        center_y,
        width,
        height,
        resolution,
        colorbar,
        fix_aspect_ratio,
    ):
        figure = go.Figure()
        data_count = "Data Count: "

        if object is not None and data is not None:
            obj = self.workspace.get_entity(uuid.UUID(object))[0]

            data_obj = self.workspace.get_entity(uuid.UUID(data))[0]

            if isinstance(obj, (Grid2D, Surface, Points, Curve)):
                figure, _, ind_filter, _, _ = InversionApp.plot_plan_data_selection(
                    obj,
                    data_obj,
                    **{
                        "figure": figure,
                        "window": {
                            "center": [center_x, center_y],
                            "size": [width, height],
                            # "azimuth": azimuth,
                        },
                        "resolution": resolution,
                        "resize": True,
                        "colorbar": colorbar,
                        "fix_aspect_ratio": fix_aspect_ratio,
                    },
                )
                data_count += f"{ind_filter.sum()}"

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
