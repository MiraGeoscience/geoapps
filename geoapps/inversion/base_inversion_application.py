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

# pylint: disable=W0613

from __future__ import annotations

import os
import uuid
import warnings
import webbrowser
from time import time

import numpy as np
import scipy
import skimage
from dash import callback_context, no_update
from geoh5py.objects import Curve, Grid2D, ObjectBase, Points, Surface
from geoh5py.shared import entity
from geoh5py.shared.utils import is_uuid
from matplotlib import colors
from notebook import notebookapp
from plotly import graph_objects as go

from geoapps.base.application import BaseApplication
from geoapps.base.dash_application import BaseDashApplication
from geoapps.inversion import InversionBaseParams
from geoapps.inversion.potential_fields.magnetic_vector.constants import app_initializer
from geoapps.shared_utils.utils import downsample_grid, downsample_xy, filter_xy


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
                {"display": "none"},
                {"display": "none"},
            )
        elif selection == "Data":
            return (
                {"display": "block"},
                {"display": "none"},
            )
        elif selection == "Constant":
            return (
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
    def unpack_val(val, topography=False):
        if is_uuid(val):
            if topography:
                options = "Data"
            else:
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
    def update_topography_from_ui_json(ui_json_data):
        param = callback_context.outputs_list[0]["id"].removesuffix("_options")
        obj = str(ui_json_data[param + "_object"])
        val = ui_json_data[param]

        options, data, const = InversionApp.unpack_val(val, topography=True)
        return options, const, obj, data

    @staticmethod
    def update_bounds_from_ui_json(ui_json_data):
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
    def update_radar_options(self, ui_json_data: dict, object_uid: str) -> (list, str):
        """
        Update data subset options and values from selected object.

        :param ui_json_data: Uploaded ui.json data.
        :param object_uid: Selected object from dropdown.

        :return options: Options for data subset dropdown.
        :return value: Value for data subset dropdown.
        """
        if object_uid is None or object_uid == "None":
            return no_update, no_update
        triggers = [c["prop_id"].split(".")[0] for c in callback_context.triggered]

        if "ui_json_data" in triggers:
            value = ui_json_data["receivers_radar_drape"]
            options = self.get_data_options("ui_json_data", ui_json_data, object_uid)
        else:
            value = None
            options = self.get_data_options("data_object", ui_json_data, object_uid)

        return options, value

    def update_data_options(self, ui_json_data: dict, object_uid: str) -> (list, str):
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

    def update_full_components(
        self,
        ui_json_data,
        full_components,
        data_object,
        channel_bool,
        channel,
        uncertainty_type,
        uncertainty_floor,
        uncertainty_channel,
        component,
        component_options,
    ):
        # full_components = no_update
        # channel_bool = no_update
        # channel = no_update
        # uncertainty_type = no_update
        # uncertainty_floor = no_update
        # uncertainty_channel = no_update
        # component = no_update

        triggers = [c["prop_id"].split(".")[0] for c in callback_context.triggered]

        if "ui_json_data" in triggers:
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
                else:
                    uncertainty_type = "Floor"
                    uncertainty_floor = None
                    uncertainty_channel = None

                full_components[comp] = {
                    "channel_bool": channel_bool,
                    "channel": channel,
                    "uncertainty_type": uncertainty_type,
                    "uncertainty_floor": uncertainty_floor,
                    "uncertainty_channel": uncertainty_channel,
                }
            # Get starting component
            for comp, value in full_components.items():
                if value["channel_bool"]:
                    component = comp
                    channel_bool = value["channel_bool"]
                    channel = value["channel"]
                    uncertainty_type = value["uncertainty_type"]
                    uncertainty_floor = value["uncertainty_floor"]
                    uncertainty_channel = value["uncertainty_channel"]
                    break
        elif "component" in triggers:
            if full_components and component is not None:
                channel_bool = full_components[component]["channel_bool"]
                channel = full_components[component]["channel"]
                uncertainty_type = full_components[component]["uncertainty_type"]
                uncertainty_floor = full_components[component]["uncertainty_floor"]
                uncertainty_channel = full_components[component]["uncertainty_channel"]
        elif "data_object" in triggers:
            for comp in full_components:
                full_components[comp]["channel_bool"] = []
                full_components[comp]["channel"] = None
                full_components[comp]["uncertainty_type"] = "Floor"
                full_components[comp]["uncertainty_floor"] = None
                full_components[comp]["uncertainty_channel"] = None
            channel_bool = []
            channel = None
            uncertainty_type = "Floor"
            uncertainty_floor = None
            uncertainty_channel = None
            component = no_update
        else:
            if "channel" in triggers:
                if channel is None:
                    channel_bool = []
                    uncertainty_floor = 1.0
                else:
                    channel_bool = [True]
                    values = self.workspace.get_entity(uuid.UUID(channel))[0].values
                    if values is not None and values.dtype in [
                        np.float32,
                        np.float64,
                        np.int32,
                    ]:
                        uncertainty_floor = np.round(
                            np.percentile(np.abs(values[~np.isnan(values)]), 5), 5
                        )
            full_components[component] = {
                "channel_bool": channel_bool,
                "channel": channel,
                "uncertainty_type": uncertainty_type,
                "uncertainty_floor": uncertainty_floor,
                "uncertainty_channel": uncertainty_channel,
            }
        return (
            full_components,
            channel_bool,
            channel,
            uncertainty_type,
            uncertainty_floor,
            uncertainty_channel,
            component,
        )

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

        if isinstance(entity, Grid2D):
            # Plot heatmap
            grid = True
            x = entity.centroids[:, 0].reshape(entity.shape, order="F")
            y = entity.centroids[:, 1].reshape(entity.shape, order="F")

            rot = entity.rotation[0]

            if values is not None:
                values = np.asarray(
                    values.reshape(entity.shape, order="F"), dtype=float
                )

                new_values = scipy.ndimage.rotate(values, rot, cval=np.nan)
                rot_x = np.linspace(x.min(), x.max(), new_values.shape[0])
                rot_y = np.linspace(y.min(), y.max(), new_values.shape[1])

                X, Y = np.meshgrid(rot_x, rot_y)

                downsampled_index, down_x, down_y = downsample_grid(X, Y, resolution)

            if np.any(values):
                figure["data"][0]["x"] = down_x
                figure["data"][0]["y"] = down_y
                figure["data"][0]["z"] = new_values.T[downsampled_index]  # new_values
        else:
            # Plot scatter plot
            grid = False
            x, y = entity.vertices[:, 0], entity.vertices[:, 1]

            if values is not None:
                downsampled_index, down_x, down_y = downsample_xy(
                    x, y, resolution, mask=~np.isnan(values)
                )
                figure["data"][0]["x"] = down_x
                figure["data"][0]["y"] = down_y
                figure["data"][0]["marker"]["color"] = values[downsampled_index]
            else:
                downsampled_index, down_x, down_y = downsample_xy(x, y, resolution)
                figure["data"][0]["x"] = down_x
                figure["data"][0]["y"] = down_y

        # Add colorbar
        if "colorbar" in kwargs.keys():
            if kwargs["colorbar"]:
                if grid:
                    figure.update_traces(showscale=True)
                else:
                    figure.update_traces(marker_showscale=True)
            else:
                if grid:
                    figure.update_traces(showscale=False)
                else:
                    figure.update_traces(marker_showscale=False)

        # Fix aspect ratio
        if "fix_aspect_ratio" in kwargs.keys():
            if kwargs["fix_aspect_ratio"]:
                figure.update_layout(yaxis_scaleanchor="x")
            else:
                figure.update_layout(yaxis_scaleanchor=None)

        return figure, indices

    def plot_selection(
        self,
        ui_json_data,
        figure,
        object,
        channel,
        resolution,
        colorbar,
        fix_aspect_ratio,
    ):
        triggers = [c["prop_id"].split(".")[0] for c in callback_context.triggered]
        data_count = "Data Count: "

        if object is None:
            return go.Figure(), data_count

        obj = self.workspace.get_entity(uuid.UUID(object))[0]
        if "data_object" in triggers or channel is None:
            if isinstance(obj, Grid2D):
                figure = go.Figure(go.Heatmap(colorscale="rainbow"))
            else:
                figure = go.Figure(
                    go.Scatter(mode="markers", marker={"colorscale": "rainbow"})
                )
            figure.update_layout(plot_bgcolor="rgba(0,0,0,0)")
            figure.update_layout(xaxis_title="Easting (m)", yaxis_title="Northing (m)")

            if "ui_json_data" not in triggers:
                return figure, data_count
        else:
            figure = go.Figure(figure)

        if channel is not None:
            data_obj = self.workspace.get_entity(uuid.UUID(channel))[0]

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

            if isinstance(obj, (Grid2D, Surface, Points, Curve)):
                figure, ind_filter = InversionApp.plot_plan_data_selection(
                    obj,
                    data_obj,
                    **{
                        "figure": figure,
                        "resolution": resolution,
                        # "window": {
                        #    "center": [center_x, center_y],
                        #    "size": [width, height],
                        # },
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
                """
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
                """
                data_count += f"{0}"

        return (
            figure,
            data_count,
        )

    @staticmethod
    def get_window_params(plot):
        x_range = plot["layout"]["xaxis"]["range"]
        y_range = plot["layout"]["yaxis"]["range"]
        width = x_range[1] - x_range[0]
        height = y_range[1] - y_range[0]

        return {
            "window_width": width,
            "window_height": height,
            "window_center_x": x_range[0] + (width / 2),
            "window_center_y": y_range[0] + (height / 2),
            "window_azimuth": 0.0,
        }

    def get_topography_params(self, topography):
        param_dict = {
            "topography_object": self.workspace.get_entity(
                uuid.UUID(topography["object"])
            )[0]
        }

        if topography["options"] == "Object":
            param_dict["topography"] = self.workspace.get_entity(
                uuid.UUID(topography["data"])
            )[0]
        elif topography["options"] == "Constant":
            param_dict["topography"] = topography["const"]
        else:
            param_dict["topography"] = None

        return param_dict

    def get_bound_params(self, bounds):
        param_dict = {}
        for key, value in bounds.items():
            param_dict[key + "_object"] = None
            param_dict[key] = None
            if value["options"] == "Object":
                param_dict[key + "_object"] = self.workspace.get_entity(
                    value["object"]
                )[0]
                param_dict[key] = self.workspace.get_entity(value["data"])[0]
            elif value["options"] == "Constant":
                param_dict[key] = value["const"]
        return param_dict

    def get_model_params(self, inversion_type, update_dict):
        models = {
            "starting_model": {
                "options": update_dict["starting_model_options"],
                "object": update_dict["starting_model_object"],
                "data": update_dict["starting_model_data"],
                "const": update_dict["starting_model_const"],
            },
            "reference_model": {
                "options": update_dict["reference_model_options"],
                "object": update_dict["reference_model_object"],
                "data": update_dict["reference_model_data"],
                "const": update_dict["reference_model_const"],
            },
        }
        if inversion_type == "magnetic_vector":
            models.update(
                {
                    "starting_inclination": {
                        "options": update_dict["starting_inclination_options"],
                        "object": update_dict["starting_inclination_object"],
                        "data": update_dict["starting_inclination_data"],
                        "const": update_dict["starting_inclination_const"],
                    },
                    "reference_inclination": {
                        "options": update_dict["reference_inclination_options"],
                        "object": update_dict["reference_inclination_object"],
                        "data": update_dict["reference_inclination_data"],
                        "const": update_dict["reference_inclination_const"],
                    },
                    "starting_declination": {
                        "options": update_dict["starting_declination_options"],
                        "object": update_dict["starting_declination_object"],
                        "data": update_dict["starting_declination_data"],
                        "const": update_dict["starting_declination_const"],
                    },
                    "reference_declination": {
                        "options": update_dict["reference_declination_options"],
                        "object": update_dict["reference_declination_object"],
                        "data": update_dict["reference_declination_data"],
                        "const": update_dict["reference_declination_const"],
                    },
                }
            )

        param_dict = {}

        for key, value in models.items():
            param_dict[key + "_object"] = None
            param_dict[key] = None
            if value["options"] == "Object":
                param_dict[key + "_object"] = self.workspace.get_entity(
                    value["object"]
                )[0]
                param_dict[key + "_data"] = self.workspace.get_entity(value["data"])[0]

            elif value["options"] == "Constant":
                param_dict[key] = value["const"]

        return param_dict

    def get_full_component_params(self, full_components, forward_only):
        param_dict = {}
        for comp, value in full_components.items():
            if value["channel_bool"] and not forward_only:
                param_dict[comp + "_channel_bool"] = True
                param_dict[comp + "_channel"] = self.workspace.get_entity(
                    uuid.UUID(value["channel"])
                )[0]
            else:
                param_dict[comp + "_channel_bool"] = False

            if value["uncertainty_type"] == "Floor":
                param_dict[comp + "_uncertainty"] = value["uncertainty_floor"]
            elif value["uncertainty_type"] == "Channel":
                # param_dict[comp + "_uncertainty"] = value["uncertainty_channel"]
                if self.workspace.get_entity(value["uncertainty_channel"])[0] is None:
                    param_dict[comp + "_uncertainty"] = self.workspace.get_entity(
                        uuid.UUID(value["uncertainty_channel"])
                    )[0]
            else:
                param_dict[comp + "_uncertainty"] = None

        return param_dict

    def get_inversion_params_dict(self, inversion_type, update_dict):
        param_dict = {}
        param_dict.update(InversionApp.get_window_params(update_dict["plot"]))
        param_dict.update(
            self.get_topography_params(
                {
                    "options": update_dict["topography_options"],
                    "object": update_dict["topography_object"],
                    "data": update_dict["topography_data"],
                    "const": update_dict["topography_const"],
                }
            )
        )
        param_dict.update(
            self.get_bound_params(
                {
                    "lower_bound": {
                        "options": update_dict["topography_options"],
                        "object": update_dict["topography_object"],
                        "data": update_dict["topography_data"],
                        "const": update_dict["topography_const"],
                    },
                    "upper_bound": {
                        "options": update_dict["topography_options"],
                        "object": update_dict["topography_object"],
                        "data": update_dict["topography_data"],
                        "const": update_dict["topography_const"],
                    },
                }
            )
        )
        param_dict.update(self.get_model_params(inversion_type, update_dict))
        param_dict.update(
            self.get_full_component_params(
                update_dict["full_components"], update_dict["forward_only"]
            )
        )
        return param_dict

    def write_trigger(
        self,
        n_clicks,
        live_link,
        data_object,
        full_components,
        resolution,
        plot,
        fix_aspect_ratio,
        topography_options,
        topography_object,
        topography_data,
        topography_const,
        z_from_topo,
        receivers_offset_x,
        receivers_offset_y,
        receivers_offset_z,
        receivers_radar_drape,
        forward_only,
        starting_model_options,
        starting_model_object,
        starting_model_data,
        starting_model_const,
        mesh,
        reference_model_options,
        reference_model_object,
        reference_model_data,
        reference_model_const,
        alpha_s,
        alpha_x,
        alpha_y,
        alpha_z,
        s_norm,
        x_norm,
        y_norm,
        z_norm,
        lower_bound_options,
        lower_bound_object,
        lower_bound_data,
        lower_bound_const,
        upper_bound_options,
        upper_bound_object,
        upper_bound_data,
        upper_bound_const,
        detrend_type,
        detrend_order,
        ignore_values,
        max_iterations,
        chi_factor,
        initial_beta_ratio,
        max_cg_iterations,
        tol_cg,
        n_cpu,
        tile_spatial,
        ga_group_name,
        monitoring_directory,
        starting_inclination_options=None,
        starting_inclination_object=None,
        starting_inclination_data=None,
        starting_inclination_const=None,
        reference_inclination_options=None,
        reference_inclination_object=None,
        reference_inclination_data=None,
        reference_inclination_const=None,
        starting_declination_options=None,
        starting_declination_object=None,
        starting_declination_data=None,
        starting_declination_const=None,
        reference_declination_options=None,
        reference_declination_object=None,
        reference_declination_data=None,
        reference_declination_const=None,
    ):
        # Get dict of params from base dash application
        param_dict = self.get_params_dict(locals())
        # Add inversion specific params to param_dict
        param_dict.update(
            self.get_inversion_params_dict(self._inversion_type, locals())
        )

        if not live_link:
            live_link = False
        else:
            live_link = True

        # Get output path
        if (
            monitoring_directory is not None
            and monitoring_directory != ""
            and os.path.exists(os.path.abspath(monitoring_directory))
        ):
            monitoring_directory = os.path.abspath(monitoring_directory)
        else:
            monitoring_directory = os.path.dirname(self.workspace.h5file)

        # Create a new workspace and copy objects into it
        temp_geoh5 = f"{ga_group_name}_{time():.0f}.geoh5"
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
                    param_dict[key] = value.copy(parent=workspace, copy_children=True)

            # param_dict["resolution"] = None  # No downsampling for dcip
            self._run_params = self.params.__class__(**param_dict)
            self._run_params.write_input_file(
                name=temp_geoh5.replace(".geoh5", ".ui.json"),
                path=monitoring_directory,
            )

        if live_link:
            print("Live link active. Check your ANALYST session for new mesh.")
            return [True]
        else:
            print("Saved to " + os.path.abspath(monitoring_directory))
            return []

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
