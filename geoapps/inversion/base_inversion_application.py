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
from dash import callback_context, no_update
from geoh5py.data import Data
from geoh5py.objects import Curve, Grid2D, ObjectBase, Points, Surface
from geoh5py.shared.utils import is_uuid
from notebook import notebookapp
from plotly import graph_objects as go

from geoapps.base.application import BaseApplication
from geoapps.base.dash_application import BaseDashApplication
from geoapps.inversion import InversionBaseParams
from geoapps.inversion.potential_fields.magnetic_vector.constants import app_initializer
from geoapps.shared_utils.utils import downsample_grid, downsample_xy


class InversionApp(BaseDashApplication):
    """
    Application for the inversion of potential field data using SimPEG
    """

    _param_class = InversionBaseParams
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
    def update_uncertainty_visibility(selection: str) -> (dict, dict):
        """
        Update visibility of channel and floor input data uncertainty from radio buttons.

        :param selection: Radio button selection.

        :return floor_style: Visibility for floor uncertainty input box.
        :return channel_style: Visibility for channel uncertainty dropdown.
        """
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
    def update_topography_visibility(selection: str) -> (dict, dict):
        """
        Update visibility of topography data and constant input boxes from radio buttons.

        :param selection: Radio button selection.

        :return data_style: Visibility for topography data dropdown.
        :return constant_style: Visibility for topography constant input box.
        """
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

    @staticmethod
    def update_model_visibility(selection: str) -> (dict, dict):
        """
        Update visibility of starting and reference model data and constant input boxes from radio buttons.

        :param selection: Radio button selection.

        :return constant_style: Visibility for model constant input box.
        :return model_style: Visibility for model object and data dropdowns.
        """
        if selection == "Constant":
            return {"display": "block"}, {"display": "none"}
        elif selection == "Model":
            return {"display": "none"}, {"display": "block"}
        elif selection == "None":
            return {"display": "none"}, {"display": "none"}

    @staticmethod
    def update_visibility_from_checkbox(selection: str) -> dict:
        """
        Update visibility of a given component from a checkbox.

        :param selection: Checkbox value.

        :return style: Visibility for the component.
        """
        if selection:
            return {"display": "block"}
        else:
            return {"display": "none"}

    @staticmethod
    def update_reference_model_options(forward_only: list) -> list:
        if forward_only:
            options = [
                {"label": "Constant", "value": "Constant", "disabled": True},
                {"label": "Model", "value": "Model", "disabled": True},
                {"label": "None", "value": "None", "disabled": False},
            ]
        else:
            options = ["Constant", "Model", "None"]
        return options

    @staticmethod
    def update_forward_only_layout(forward_only: list):
        if forward_only:
            style = {"display": "none"}
            options = [
                {"label": "Advanced parameters", "value": True, "disabled": True}
            ]
        else:
            style = {"display": "block"}
            options = [{"label": "Advanced parameters", "value": True}]
        return style, options

    @staticmethod
    def open_mesh_app(_: int) -> int:
        """
        Triggered on open mesh app button click. Opens mesh creator notebook in a new window.

        :param _: Triggers function when button is clicked.

        :return n_clicks: Placeholder return since dash requires callbacks to have output.
        """
        # Get a notebook port that is running from the index page.
        nb_port = None
        servers = list(notebookapp.list_running_servers())
        for s in servers:
            if s["notebook_dir"] == os.path.abspath("../../../"):
                nb_port = s["port"]
                break

        # Open the octree creation notebook in a new window using the notebook port in the url.
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
    def unpack_val(
        val: float | int | str, topography: bool = False
    ) -> (str, str, float | int):
        """
        Determine if input value is a constant or data, and determine the corresponding radio button value.

        :param val: Input value. Either constant, data, or None.
        :param topography: Whether the parameter is topography, since the topography radio buttons are slightly
        different.

        :return options: Radio button selection.
        :return data: Data value.
        :return const: Constant value.
        """
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

    def update_models_from_ui_json(
        self,
        ui_json_data: dict,
        data_object_options: list,
        object_uid: str,
        forward_only: list,
    ) -> (str, float | int, str, list, str, list):
        """
        Update starting and reference models from ui.json data. Update dropdown options and values.

        :param ui_json_data: Uploaded ui.json data.
        :param data_object_options: List of dropdown options for main input object.
        :param object_uid: Selected object for the model.

        :return options: Selected option for radio button.
        :return const: Value of constant for model.
        :return obj: Value of object for model.
        :return obj_options: Dropdown options for model object. Same as data_object_options.
        :return data: Value of data for model.
        :return data_options: Dropdown options for model data.
        """
        options, const, obj, obj_options, data, data_options = (
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
        )

        # Get param name based on callback input.
        prefix, param = tuple(
            callback_context.outputs_list[0]["id"]
            .removesuffix("_options")
            .split("_", 1)
        )
        # Get callback triggers
        triggers = [c["prop_id"].split(".")[0] for c in callback_context.triggered]

        if "ui_json_data" in triggers:
            if param in self._inversion_params:
                # Read in from ui.json using dict of inversion params.
                if prefix + "_" + param in ui_json_data:
                    obj = str(ui_json_data[prefix + "_" + param + "_object"])
                    val = ui_json_data[prefix + "_" + param]
                elif prefix + "_model" in ui_json_data:
                    obj = str(ui_json_data[prefix + "_model_object"])
                    val = ui_json_data[prefix + "_model"]
                else:
                    obj = None
                    val = None

                options, data, const = InversionApp.unpack_val(val)
                data_options = self.get_data_options(ui_json_data, obj)
                obj_options = data_object_options
        elif "data_object" in triggers:
            # Clear object value and data dropdown on workspace change.
            obj_options = data_object_options
            obj = None
            data_options = []
            data = None
        elif "forward_only" in triggers and forward_only:
            if prefix == "reference":
                options = "None"
        else:
            # Update data options and clear data value on object change.
            data = None
            data_options = self.get_data_options(ui_json_data, object_uid)

        return options, const, obj, obj_options, data, data_options

    def update_general_inversion_params_from_ui_json(
        self, ui_json_data, data_object_options, param_object_uid
    ) -> (str, float | int, str, list, str, list):
        """
        Update topography and bounds from ui.json data. Update dropdown options and values.

        :param ui_json_data: Uploaded ui.json data.
        :param data_object_options: List of dropdown options for main input object.
        :param param_object_uid: Selected object for the param.

        :return options: Selected option for radio button.
        :return const: Value of constant for param.
        :return obj: Value of object for param.
        :return obj_options: Dropdown options for param object. Same as data_object_options.
        :return data: Value of data for param.
        :return data_options: Dropdown options for param data.
        """
        options, const, obj, obj_options, data, data_options = (
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
        )
        triggers = [c["prop_id"].split(".")[0] for c in callback_context.triggered]

        if "ui_json_data" in triggers:
            param = callback_context.outputs_list[0]["id"].removesuffix("_options")
            obj = str(ui_json_data[param + "_object"])
            val = ui_json_data[param]

            if param == "topography":
                options, data, const = InversionApp.unpack_val(val, topography=True)
            else:
                options, data, const = InversionApp.unpack_val(val)

            data_options = self.get_data_options(ui_json_data, obj)
            obj_options = data_object_options
        elif "data_object" in triggers:
            obj_options = data_object_options
            obj = None
            data_options = []
            data = None
        else:
            data = None
            data_options = self.get_data_options(ui_json_data, param_object_uid)

        return options, const, obj, obj_options, data, data_options

    @staticmethod
    def update_mesh_options(obj_options: list) -> list:
        """
        Update mesh dropdown options from the main input object options.

        :param obj_options: Main input object options.

        :return obj_options: Mesh dropdown options.
        """
        return obj_options

    def update_radar_options(self, ui_json_data: dict, object_uid: str) -> (list, str):
        """
        Update data dropdown options for receivers radar drape.

        :param ui_json_data: Uploaded ui.json data.
        :param object_uid: Selected object from dropdown.

        :return options: Options for radar dropdown.
        :return value: Value for radar dropdown.
        """
        if object_uid is None or object_uid == "None":
            return no_update, no_update
        triggers = [c["prop_id"].split(".")[0] for c in callback_context.triggered]

        if "ui_json_data" in triggers:
            value = ui_json_data["receivers_radar_drape"]
            options = self.get_data_options(
                ui_json_data,
                object_uid,
                trigger="ui_json_data",
                object_name="data_object",
            )
        else:
            value = None
            options = self.get_data_options(ui_json_data, object_uid)

        return options, value

    def update_full_components(
        self,
        ui_json_data: dict,
        full_components: dict,
        data_object: str,
        channel_bool: list,
        channel: str,
        uncertainty_type: str,
        uncertainty_floor: float | int,
        uncertainty_channel: str,
        component: str,
        component_options: list,
    ) -> (dict, list, str, list, str, float | int, str, list, str):
        """
        Update components in relating to input data. Update the dictionary storing these values for each component.

        :param ui_json_data: Uploaded ui.json data.
        :param full_components: Dictionary with keys of component_options, and with values channel_bool, channel,
        uncertainty_type, uncertainty_floor, uncertainty_channel for each key.
        :param data_object: Input data object.
        :param channel_bool: Checkbox for whether the channel is active.
        :param channel: Input data.
        :param uncertainty_type: Type of uncertainty. Floor or channel.
        :param uncertainty_floor: Uncertainty floor.
        :param uncertainty_channel: Uncertainty data.
        :param component: Component that data corresponds to.
        :param component_options: List of component options.

        :return full_components: Dictionary with keys of component_options, and with values channel_bool, channel,
        uncertainty_type, uncertainty_floor, uncertainty_channel for each key.
        :return channel_bool: Checkbox for whether the channel is active.
        :return channel: Input data.
        :return dropdown_options: Dropdown options for channel.
        :return uncertainty_type: Type of uncertainty. Floor or channel.
        :return uncertainty_floor: Uncertainty floor.
        :return uncertainty_channel: Uncertainty data.
        :return dropdown_options: Dropdown options for uncertainty channel.
        :return component: Component that data corresponds to.
        """
        dropdown_options = no_update
        triggers = [c["prop_id"].split(".")[0] for c in callback_context.triggered]

        if "ui_json_data" in triggers:
            # Fill in full_components dict from ui.json.
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
                    # Default uncertainty value
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

            # Get component to initialize app. First active component.
            for comp, value in full_components.items():
                if value["channel_bool"]:
                    component = comp
                    channel_bool = value["channel_bool"]
                    channel = value["channel"]
                    uncertainty_type = value["uncertainty_type"]
                    uncertainty_floor = value["uncertainty_floor"]
                    uncertainty_channel = value["uncertainty_channel"]
                    break
            # Update channel data dropdown options.
            dropdown_options = self.get_data_options(
                ui_json_data,
                data_object,
                trigger="ui_json_data",
                object_name="data_object",
            )
        elif "component" in triggers:
            # On component change, read in new values to display from full_components dict.
            if full_components and component is not None:
                channel_bool = full_components[component]["channel_bool"]
                channel = full_components[component]["channel"]
                uncertainty_type = full_components[component]["uncertainty_type"]
                uncertainty_floor = full_components[component]["uncertainty_floor"]
                uncertainty_channel = full_components[component]["uncertainty_channel"]
        elif "data_object" in triggers:
            # On object change, clear full_components and update data dropdown options.
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
            dropdown_options = self.get_data_options(ui_json_data, data_object)
        else:
            # If the channel, channel_bool, or uncertainties are changed, we need to update the full_components dict.
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
            dropdown_options,
            uncertainty_type,
            uncertainty_floor,
            uncertainty_channel,
            dropdown_options,
            component,
        )

    @staticmethod
    def plot_plan_data_selection(entity: ObjectBase, data: Data, **kwargs) -> go.Figure:
        """
        A simplified version of the plot_plan_data_selection function in utils/plotting, except for dash.

        :param entity: Input object with either `vertices` or `centroids` property.
        :param data: Input data with `values` property.

        :return figure: Figure with updated data
        """
        values = None
        figure = None

        if isinstance(entity, (Grid2D, Points, Curve, Surface)):
            if "figure" not in kwargs.keys():
                figure = go.Figure()
            else:
                figure = kwargs["figure"]
        else:
            return figure

        if getattr(entity, "vertices", None) is not None:
            locations = entity.vertices
        else:
            locations = entity.centroids

        if "resolution" not in kwargs.keys():
            resolution = 0
        else:
            resolution = kwargs["resolution"]

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

                # Rotate plot to match object rotation.
                new_values = scipy.ndimage.rotate(values, rot, cval=np.nan)
                rot_x = np.linspace(x.min(), x.max(), new_values.shape[0])
                rot_y = np.linspace(y.min(), y.max(), new_values.shape[1])

                X, Y = np.meshgrid(rot_x, rot_y)

                # Downsample grid
                downsampled_index, down_x, down_y = downsample_grid(X, Y, resolution)

            if np.any(values):
                # Update figure data.
                figure["data"][0]["x"] = down_x
                figure["data"][0]["y"] = down_y
                figure["data"][0]["z"] = new_values.T[downsampled_index]
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

        return figure

    def plot_selection(
        self,
        ui_json_data: dict,
        figure: dict,
        object_uid: str,
        channel: str,
        resolution: float | int,
        colorbar: list,
        fix_aspect_ratio: list,
    ) -> (go.Figure, str):
        """
        Dash version of the plot_selection function in base/plot.

        :param ui_json_data: Uploaded ui.json data.
        :param figure: Current displayed figure.
        :param object_uid: Input object.
        :param channel: Input data.
        :param resolution: Resolution distance.
        :param colorbar: Checkbox value for whether to display colorbar.
        :param fix_aspect_ratio: Checkbox value for whether to fix aspect ratio.

        :return figure: Updated figure.
        :return data_count: Displayed data count value.
        """
        triggers = [c["prop_id"].split(".")[0] for c in callback_context.triggered]
        data_count = "Data Count: "

        if object_uid is None:
            # If object is None, figure is empty.
            return go.Figure(), data_count

        obj = self.workspace.get_entity(uuid.UUID(object_uid))[0]
        if "data_object" in triggers or channel is None:
            # If object changes, update plot type based on object type.
            if isinstance(obj, Grid2D):
                figure = go.Figure(go.Heatmap(colorscale="rainbow"))
            else:
                figure = go.Figure(
                    go.Scatter(mode="markers", marker={"colorscale": "rainbow"})
                )
            figure.update_layout(plot_bgcolor="rgba(0,0,0,0)")
            figure.update_layout(xaxis_title="Easting (m)", yaxis_title="Northing (m)")

            if "ui_json_data" not in triggers:
                # If we aren't reading in a ui.json, return the empty plot.
                return figure, data_count
        else:
            # Construct figure from existing figure to keep bounds and plot layout.
            figure = go.Figure(figure)

        if channel is not None:
            data_obj = self.workspace.get_entity(uuid.UUID(channel))[0]

            # Update plot data.
            if isinstance(obj, (Grid2D, Surface, Points, Curve)):
                figure = InversionApp.plot_plan_data_selection(
                    obj,
                    data_obj,
                    **{
                        "figure": figure,
                        "resolution": resolution,
                        "colorbar": colorbar,
                        "fix_aspect_ratio": fix_aspect_ratio,
                    },
                )
                # Update plot bounds from ui.json.
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
    def get_window_params(plot: go.Figure) -> dict:
        """
        Get window params from plot to update self.params.

        :param plot: Plot of data.

        :return dict: Dictionary with window params.
        """
        # Get plot bounds
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

    def get_topography_params(self, topography: dict) -> dict:
        """
        Get topography params to update self.params.

        :param topography: Dict with topography type, data, constant, and object.

        :return param_dict: Dictionary with topography_object and topography.
        """
        param_dict = {
            "topography_object": self.workspace.get_entity(
                uuid.UUID(topography["object"])
            )[0]
        }

        # Determine whether to save data or constant from topography options value.
        if topography["options"] == "Data" and is_uuid(topography["data"]):
            param_dict["topography"] = self.workspace.get_entity(
                uuid.UUID(topography["data"])
            )[0]
        elif topography["options"] == "Constant":
            param_dict["topography"] = topography["const"]
        else:
            param_dict["topography"] = None

        return param_dict

    def get_bound_params(self, bounds: dict) -> dict:
        """
        Get lower and upper bounds to add to self.params.

        :param bounds: Dictionary of lower, upper bounds with radio button selection, data, constant, and object for
        each.

        :return param_dict: Dictionary with lower_bound_object, lower_bound, upper_bound_object, upper_bound.
        """
        param_dict = {}
        for key, value in bounds.items():
            param_dict[key + "_object"] = None
            param_dict[key] = None
            if value["options"] == "Model":
                if is_uuid(value["object"]):
                    param_dict[key + "_object"] = self.workspace.get_entity(
                        uuid.UUID(value["object"])
                    )[0]
                if is_uuid(value["data"]):
                    param_dict[key] = self.workspace.get_entity(
                        uuid.UUID(value["data"])
                    )[0]
            elif value["options"] == "Constant":
                param_dict[key] = value["const"]

        return param_dict

    def get_model_params(self, update_dict: dict) -> dict:
        """
        Get dictionary of params to update self.params with starting and reference model values.

        :param update_dict: Dictionary of input values from dash callback.

        :return param_dict: Dictionary with model values to save.
        """
        # Base dictionary of models.
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
        # Add additional parameters for magnetic vector inversion.
        if self._inversion_type == "magnetic_vector":
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
        # Loop through dict of models and determine whether to save data or constant.
        for key, value in models.items():
            param_dict[key + "_object"] = None
            param_dict[key] = None
            if value["options"] == "Model":
                if is_uuid(value["object"]):
                    param_dict[key + "_object"] = self.workspace.get_entity(
                        uuid.UUID(value["object"])
                    )[0]
                if is_uuid(value["data"]):
                    param_dict[key] = self.workspace.get_entity(
                        uuid.UUID(value["data"])
                    )[0]

            elif value["options"] == "Constant":
                param_dict[key] = value["const"]

        return param_dict

    def get_full_component_params(
        self, full_components: dict, forward_only: list
    ) -> dict:
        """
        Get param_dict of values to add to self.params from full_components.

        :param full_components: Dictionary with keys of component_options, and with values channel_bool, channel,
        uncertainty_type, uncertainty_floor, uncertainty_channel for each key.
        :param forward_only: Checkbox of whether to perform only forward inversion.

        :return param_dict: Dictionary of values to update self.params.
        """
        param_dict = {}
        # Loop through
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

    def get_inversion_params_dict(self, update_dict):
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
                        "options": update_dict["lower_bound_options"],
                        "object": update_dict["lower_bound_object"],
                        "data": update_dict["lower_bound_data"],
                        "const": update_dict["lower_bound_const"],
                    },
                    "upper_bound": {
                        "options": update_dict["upper_bound_options"],
                        "object": update_dict["upper_bound_object"],
                        "data": update_dict["upper_bound_data"],
                        "const": update_dict["upper_bound_const"],
                    },
                }
            )
        )
        param_dict.update(self.get_model_params(update_dict))
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
        out_group,
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
        param_dict.update(self.get_inversion_params_dict(locals()))

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
        temp_geoh5 = f"{'GravityInversion'}_{time():.0f}.geoh5"
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
