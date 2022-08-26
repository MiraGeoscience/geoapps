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
from dash import Input, Output, State, callback_context, no_update
from flask import Flask
from geoh5py.data import Data
from geoh5py.objects import Curve, Grid2D, ObjectBase, Points, Surface
from geoh5py.shared.utils import is_uuid
from geoh5py.ui_json import monitored_directory_copy
from geoh5py.workspace import Workspace
from jupyter_dash import JupyterDash
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
    _inversion_params = {}
    _run_params = None
    _layout = None

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

        self.app.layout = self._layout

        # Callbacks relating to layout
        self.app.callback(
            Output(component_id="uncertainty_floor", component_property="style"),
            Output(component_id="uncertainty_channel", component_property="style"),
            Input(component_id="uncertainty_options", component_property="value"),
        )(InversionApp.update_uncertainty_visibility)
        self.app.callback(
            Output(component_id="topography_object_div", component_property="style"),
            Output(component_id="topography_constant_div", component_property="style"),
            Input(component_id="topography_options", component_property="value"),
        )(InversionApp.update_topography_visibility)
        for model_type in ["starting", "reference"]:
            for param in self._inversion_params:
                self.app.callback(
                    Output(
                        component_id=model_type + "_" + param + "_const_div",
                        component_property="style",
                    ),
                    Output(
                        component_id=model_type + "_" + param + "_mod_div",
                        component_property="style",
                    ),
                    Input(
                        component_id=model_type + "_" + param + "_options",
                        component_property="value",
                    ),
                )(InversionApp.update_model_visibility)
        self.app.callback(
            Output(component_id="lower_bound_const_div", component_property="style"),
            Output(component_id="lower_bound_mod_div", component_property="style"),
            Input(component_id="lower_bound_options", component_property="value"),
        )(InversionApp.update_model_visibility)
        self.app.callback(
            Output(component_id="upper_bound_const_div", component_property="style"),
            Output(component_id="upper_bound_mod_div", component_property="style"),
            Input(component_id="upper_bound_options", component_property="value"),
        )(InversionApp.update_model_visibility)
        self.app.callback(
            Output(component_id="core_params_div", component_property="style"),
            Input(component_id="core_params", component_property="value"),
        )(InversionApp.update_visibility_from_checkbox)
        self.app.callback(
            Output(component_id="advanced_params_div", component_property="style"),
            Input(component_id="advanced_params", component_property="value"),
        )(InversionApp.update_visibility_from_checkbox)
        # Update components from forward only checkbox
        for param in self._inversion_params:
            self.app.callback(
                Output(
                    component_id="reference_" + param + "_options",
                    component_property="options",
                ),
                Input(
                    component_id="forward_only",
                    component_property="value",
                ),
            )(InversionApp.update_reference_model_options)
        self.app.callback(
            Output(component_id="forward_only_div", component_property="style"),
            Output(component_id="advanced_params", component_property="options"),
            Input(component_id="forward_only", component_property="value"),
        )(InversionApp.update_forward_only_layout)

        # Update object and data dropdowns
        self.app.callback(
            Output(component_id="data_object", component_property="options"),
            Output(component_id="data_object", component_property="value"),
            Output(component_id="ui_json_data", component_property="data"),
            Output(component_id="upload", component_property="filename"),
            Output(component_id="upload", component_property="contents"),
            Input(component_id="upload", component_property="filename"),
            Input(component_id="upload", component_property="contents"),
        )(self.update_object_options)

        # Update mesh object dropdown options
        self.app.callback(
            Output(component_id="mesh", component_property="options"),
            Input(component_id="data_object", component_property="options"),
        )(InversionApp.update_mesh_options)

        # Update radar data options
        self.app.callback(
            Output(component_id="receivers_radar_drape", component_property="options"),
            Output(component_id="receivers_radar_drape", component_property="value"),
            Input(component_id="ui_json_data", component_property="data"),
            Input(component_id="data_object", component_property="value"),
        )(self.update_radar_options)

        # Update input data channel and uncertainties from component
        self.app.callback(
            Output(component_id="full_components", component_property="data"),
            Output(component_id="channel_bool", component_property="value"),
            Output(component_id="channel", component_property="value"),
            Output(component_id="channel", component_property="options"),
            Output(component_id="uncertainty_options", component_property="value"),
            Output(component_id="uncertainty_floor", component_property="value"),
            Output(component_id="uncertainty_channel", component_property="value"),
            Output(component_id="uncertainty_channel", component_property="options"),
            Output(component_id="component", component_property="value"),
            Input(component_id="ui_json_data", component_property="data"),
            Input(component_id="full_components", component_property="data"),
            Input(component_id="data_object", component_property="value"),
            Input(component_id="channel_bool", component_property="value"),
            Input(component_id="channel", component_property="value"),
            Input(component_id="uncertainty_options", component_property="value"),
            Input(component_id="uncertainty_floor", component_property="value"),
            Input(component_id="uncertainty_channel", component_property="value"),
            Input(component_id="component", component_property="value"),
            State(component_id="component", component_property="options"),
        )(self.update_full_components)

        # Update model dropdown options and values
        for model_type in ["starting", "reference"]:
            for param in self._inversion_params:
                self.app.callback(
                    Output(
                        component_id=model_type + "_" + param + "_options",
                        component_property="value",
                    ),
                    Output(
                        component_id=model_type + "_" + param + "_const",
                        component_property="value",
                    ),
                    Output(
                        component_id=model_type + "_" + param + "_object",
                        component_property="value",
                    ),
                    Output(
                        component_id=model_type + "_" + param + "_object",
                        component_property="options",
                    ),
                    Output(
                        component_id=model_type + "_" + param + "_data",
                        component_property="value",
                    ),
                    Output(
                        component_id=model_type + "_" + param + "_data",
                        component_property="options",
                    ),
                    Input(component_id="ui_json_data", component_property="data"),
                    Input(component_id="data_object", component_property="options"),
                    Input(
                        component_id=model_type + "_" + param + "_object",
                        component_property="value",
                    ),
                    Input(component_id="forward_only", component_property="value"),
                )(self.update_models_from_ui_json)

        # Update bounds dropdown options and values
        for param in ["lower_bound", "upper_bound", "topography"]:
            self.app.callback(
                Output(component_id=param + "_options", component_property="value"),
                Output(component_id=param + "_const", component_property="value"),
                Output(component_id=param + "_object", component_property="value"),
                Output(component_id=param + "_object", component_property="options"),
                Output(component_id=param + "_data", component_property="value"),
                Output(component_id=param + "_data", component_property="options"),
                Input(component_id="ui_json_data", component_property="data"),
                Input(component_id="data_object", component_property="options"),
                Input(component_id=param + "_object", component_property="value"),
            )(self.update_general_inversion_params_from_ui_json)

        # Update from ui.json
        self.app.callback(
            # Input Data
            Output(component_id="resolution", component_property="value"),
            # Topography
            Output(component_id="z_from_topo", component_property="value"),
            Output(component_id="receivers_offset_x", component_property="value"),
            Output(component_id="receivers_offset_y", component_property="value"),
            Output(component_id="receivers_offset_z", component_property="value"),
            # Inversion - mesh
            Output(component_id="mesh", component_property="value"),
            # Inversion - regularization
            Output(component_id="alpha_s", component_property="value"),
            Output(component_id="alpha_x", component_property="value"),
            Output(component_id="alpha_y", component_property="value"),
            Output(component_id="alpha_z", component_property="value"),
            Output(component_id="s_norm", component_property="value"),
            Output(component_id="x_norm", component_property="value"),
            Output(component_id="y_norm", component_property="value"),
            Output(component_id="z_norm", component_property="value"),
            # Inversion - detrend
            Output(component_id="detrend_type", component_property="value"),
            Output(component_id="detrend_order", component_property="value"),
            # Inversion - ignore values
            Output(component_id="ignore_values", component_property="value"),
            # Inversion - optimization
            Output(component_id="max_iterations", component_property="value"),
            Output(component_id="chi_factor", component_property="value"),
            Output(component_id="initial_beta_ratio", component_property="value"),
            Output(component_id="max_cg_iterations", component_property="value"),
            Output(component_id="tol_cg", component_property="value"),
            Output(component_id="n_cpu", component_property="value"),
            Output(component_id="tile_spatial", component_property="value"),
            # Output
            Output(component_id="out_group", component_property="value"),
            Output(component_id="monitoring_directory", component_property="value"),
            Output(component_id="forward_only", component_property="value"),
            Input(component_id="ui_json_data", component_property="data"),
        )(self.update_remainder_from_ui_json)

        # Plot callbacks
        # Update slider bounds
        self.app.callback(
            Output(component_id="window_center_x", component_property="min"),
            Output(component_id="window_center_x", component_property="max"),
            Output(component_id="window_center_y", component_property="min"),
            Output(component_id="window_center_y", component_property="max"),
            Output(component_id="window_width", component_property="max"),
            Output(component_id="window_height", component_property="max"),
            Input(component_id="data_object", component_property="value"),
        )(self.set_bounding_box)
        # Update plot
        self.app.callback(
            Output(component_id="plot", component_property="figure"),
            Output(component_id="data_count", component_property="children"),
            Output(component_id="window_center_x", component_property="value"),
            Output(component_id="window_center_y", component_property="value"),
            Output(component_id="window_width", component_property="value"),
            Output(component_id="window_height", component_property="value"),
            Output(component_id="fix_aspect_ratio", component_property="value"),
            Input(component_id="ui_json_data", component_property="data"),
            Input(component_id="plot", component_property="figure"),
            Input(component_id="plot", component_property="relayoutData"),
            Input(component_id="data_object", component_property="value"),
            Input(component_id="channel", component_property="value"),
            Input(component_id="window_center_x", component_property="value"),
            Input(component_id="window_center_y", component_property="value"),
            Input(component_id="window_width", component_property="value"),
            Input(component_id="window_height", component_property="value"),
            Input(component_id="resolution", component_property="value"),
            Input(component_id="colorbar", component_property="value"),
            Input(component_id="fix_aspect_ratio", component_property="value"),
        )(self.plot_selection)

        # Button callbacks
        self.app.callback(
            Output(component_id="compute", component_property="n_clicks"),
            Input(component_id="compute", component_property="n_clicks"),
            prevent_initial_call=True,
        )(self.trigger_click)
        self.app.callback(
            Output(component_id="open_mesh", component_property="n_clicks"),
            Input(component_id="open_mesh", component_property="n_clicks"),
            prevent_initial_call=True,
        )(InversionApp.open_mesh_app)

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
        """
        Disable constant and model options for reference model when forward only is selected.

        :param forward_only: Checkbox for whether to perform forward inversion.

        :return options: Reference model radio button options.
        """
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
    def update_forward_only_layout(forward_only: list) -> (dict, list):
        """
        Update layout from forward_only checkbox.

        :param forward_only: Checkbox for whether to perform forward inversion.

        :return style: Style for div containing channel, and uncertainties.
        :return options: Options for advanced parameters checkbox.
        """
        if forward_only:
            style = {"display": "none"}
            # Disable advanced parameters if forward only inversion.
            options = [
                {"label": "Advanced parameters", "value": True, "disabled": True}
            ]
        else:
            style = {"display": "block"}
            options = [{"label": "Advanced parameters", "value": True}]
        return style, options

    @staticmethod
    def open_mesh_app(_) -> None:
        """
        Triggered on open mesh app button click. Opens mesh creator notebook in a new window.

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
        return None

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
        :param forward_only: Checkbox for performing forward inversion.

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
            if param in self._inversion_params.keys():
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
            # Set the reference model to none for forward inversion.
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
            if param in ui_json_data:
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

        options.insert(0, {"label": "", "value": ""})
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
                # Get channel_bool value
                if ui_json_data[comp + "_channel_bool"]:
                    channel_bool = [True]
                else:
                    channel_bool = []
                # Get channel value
                if comp + "_channel" in ui_json_data and is_uuid(
                    ui_json_data[comp + "_channel"]
                ):
                    channel = str(ui_json_data[comp + "_channel"])
                else:
                    channel = None
                # Get uncertainty value
                if comp + "_uncertainty" in ui_json_data and (
                    (type(ui_json_data[comp + "_uncertainty"]) == float)
                    or (type(ui_json_data[comp + "_uncertainty"]) == int)
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
                    # Set default floor on channel change
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

    def set_bounding_box(
        self, object_uid: str
    ) -> (float, float, float, float, float, float):
        """
        Get slider min and max values after object change.

        :param object_uid: Input object uuid.

        :return window_center_x_min: Center x slider min.
        :return window_center_x_max: Center x slider max.
        :return window_center_y_min: Center y slider min.
        :return window_center_y_max: Center y slider max.
        :return window_width_max: Width slider max.
        :return window_height_max: Height slider max.
        """
        # Fetch vertices in the project
        lim_x = [1e8, -1e8]
        lim_y = [1e8, -1e8]

        if is_uuid(object_uid):
            obj = self.workspace.get_entity(uuid.UUID(object_uid))[0]
        else:
            return no_update, no_update, no_update, no_update, no_update, no_update
        if isinstance(obj, Grid2D):
            lim_x[0], lim_x[1] = obj.centroids[:, 0].min(), obj.centroids[:, 0].max()
            lim_y[0], lim_y[1] = obj.centroids[:, 1].min(), obj.centroids[:, 1].max()
        elif isinstance(obj, (Points, Curve, Surface)):
            lim_x[0], lim_x[1] = obj.vertices[:, 0].min(), obj.vertices[:, 0].max()
            lim_y[0], lim_y[1] = obj.vertices[:, 1].min(), obj.vertices[:, 1].max()
        else:
            return no_update, no_update, no_update, no_update, no_update, no_update

        width = lim_x[1] - lim_x[0]
        height = lim_y[1] - lim_y[0]

        window_center_x_max = lim_x[1] + width * 0.1
        window_center_x_min = lim_x[0] - width * 0.1

        window_center_y_max = lim_y[1] + height * 0.1
        window_center_y_min = lim_y[0] - height * 0.1

        window_width_max = width * 1.2
        window_height_max = height * 1.2

        return (
            window_center_x_min,
            window_center_x_max,
            window_center_y_min,
            window_center_y_max,
            window_width_max,
            window_height_max,
        )

    @staticmethod
    def plot_plan_data_selection(
        entity: ObjectBase, data: Data, **kwargs
    ) -> (go.Figure, int):
        """
        A simplified version of the plot_plan_data_selection function in utils/plotting, except for dash.

        :param entity: Input object with either `vertices` or `centroids` property.
        :param data: Input data with `values` property.

        :return figure: Figure with updated data
        :return data_count: Count of data in the window, to be exported.
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

        if "window" not in kwargs.keys():
            window = None
        else:
            window = kwargs["window"]

            # Figure layout changes
            # Fix aspect ratio
            if "fix_aspect_ratio" in kwargs.keys():
                if kwargs["fix_aspect_ratio"]:
                    figure.update_layout(yaxis_scaleanchor="x")
                else:
                    figure.update_layout(yaxis_scaleanchor=None)

            # Set plot axes limits
            if window:
                figure.update_layout(
                    xaxis_autorange=False,
                    xaxis_range=[
                        window["center"][0] - (window["size"][0] / 2),
                        window["center"][0] + (window["size"][0] / 2),
                    ],
                    yaxis_autorange=False,
                    yaxis_range=[
                        window["center"][1] - (window["size"][1] / 2),
                        window["center"][1] + (window["size"][1] / 2),
                    ],
                )
        # Add data to figure
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

                z = new_values.T[downsampled_index]

            if np.any(values):
                # Update figure data.
                figure["data"][0]["x"] = down_x
                figure["data"][0]["y"] = down_y
                figure["data"][0]["z"] = z

            # Get data count
            # Downsample grid
            downsampled_index, down_x, down_y = downsample_grid(x, y, resolution)
            z = values[downsampled_index]

            if figure["layout"]["xaxis"]["autorange"]:
                z_count = z
            else:
                x_range = figure["layout"]["xaxis"]["range"]
                x_indices = (x_range[0] <= down_x) & (down_x <= x_range[1])
                y_range = figure["layout"]["yaxis"]["range"]
                y_indices = (y_range[0] <= down_y) & (down_y <= y_range[1])
                z_count = z[np.logical_and(x_indices, y_indices)]

            data_count = np.sum(~np.isnan(z_count))

            # Add colorbar
            if "colorbar" in kwargs.keys():
                if kwargs["colorbar"]:
                    figure.update_traces(showscale=True)
                else:
                    figure.update_traces(showscale=False)

        else:
            # Plot scatter plot
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

            if figure["layout"]["xaxis"]["autorange"]:
                count = np.logical_and(down_x, down_y)
            else:
                x_range = figure["layout"]["xaxis"]["range"]
                x_indices = (x_range[0] <= down_x) & (down_x <= x_range[1])
                y_range = figure["layout"]["yaxis"]["range"]
                y_indices = (y_range[0] <= down_y) & (down_y <= y_range[1])
                count = np.logical_and(x_indices, y_indices)

            data_count = np.sum(count)

            # Add colorbar
            if "colorbar" in kwargs.keys():
                if kwargs["colorbar"]:
                    figure.update_traces(marker_showscale=True)
                else:
                    figure.update_traces(marker_showscale=False)

        return figure, data_count

    def plot_selection(
        self,
        ui_json_data: dict,
        figure: dict,
        figure_zoom_trigger: dict,
        object_uid: str,
        channel: str,
        center_x: float | int,
        center_y: float | int,
        width: float | int,
        height: float | int,
        resolution: float | int,
        colorbar: list,
        fix_aspect_ratio: list,
    ) -> (go.Figure, str, float, float, float, float, list):
        """
        Dash version of the plot_selection function in base/plot.

        :param ui_json_data: Uploaded ui.json data.
        :param figure: Current displayed figure.
        :param figure_zoom_trigger: Trigger for when zoom on figure.
        :param object_uid: Input object.
        :param channel: Input data.
        :param center_x: Window center x.
        :param center_y: Window center y.
        :param width: Window width.
        :param height: Window height
        :param resolution: Resolution distance.
        :param colorbar: Checkbox value for whether to display colorbar.
        :param fix_aspect_ratio: Checkbox value for whether to fix aspect ratio.

        :return figure: Updated figure.
        :return data_count: Displayed data count value.
        :param center_x: Window center x.
        :param center_y: Window center y.
        :param width: Window width.
        :param height: Window height
        :return fix_aspect_ratio: Whether to fix aspect ratio. Turned off when width/height sliders adjusted.
        """
        triggers = [c["prop_id"].split(".")[0] for c in callback_context.triggered]
        data_count = "Data Count: "

        if object_uid is None:
            # If object is None, figure is empty.
            return (
                go.Figure(),
                data_count,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
            )

        obj = self.workspace.get_entity(uuid.UUID(object_uid))[0]
        if "data_object" in triggers or channel is None:
            # If object changes, update plot type based on object type.
            if isinstance(obj, Grid2D):
                figure = go.Figure(go.Heatmap(colorscale="rainbow"))
            else:
                figure = go.Figure(
                    go.Scatter(mode="markers", marker={"colorscale": "rainbow"})
                )
            figure.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Easting (m)",
                yaxis_title="Northing (m)",
                margin=dict(l=20, r=20, t=20, b=20),
            )

            if "ui_json_data" not in triggers:
                # If we aren't reading in a ui.json, return the empty plot.
                return (
                    figure,
                    data_count,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                )
        else:
            # Construct figure from existing figure to keep bounds and plot layout.
            figure = go.Figure(figure)

        if channel is not None:
            data_obj = self.workspace.get_entity(uuid.UUID(channel))[0]

            window = None
            # Update plot bounds from ui.json.
            if "ui_json_data" in triggers:
                if ui_json_data["fix_aspect_ratio"]:
                    fix_aspect_ratio = [True]
                else:
                    fix_aspect_ratio = []

                center_x = ui_json_data["window_center_x"]
                center_y = ui_json_data["window_center_y"]
                width = ui_json_data["window_width"]
                height = ui_json_data["window_height"]

                window = {
                    "center": [
                        center_x,
                        center_y,
                    ],
                    "size": [
                        width,
                        height,
                    ],
                }
            elif any(
                elem
                in [
                    "window_center_x",
                    "window_center_y",
                    "window_width",
                    "window_height",
                ]
                for elem in triggers
            ):
                if "window_width" in triggers or "window_height" in triggers:
                    fix_aspect_ratio = []
                window = {
                    "center": [
                        center_x,
                        center_y,
                    ],
                    "size": [
                        width,
                        height,
                    ],
                }
            # Update plot data.
            if isinstance(obj, (Grid2D, Surface, Points, Curve)):
                figure, count = InversionApp.plot_plan_data_selection(
                    obj,
                    data_obj,
                    **{
                        "figure": figure,
                        "resolution": resolution,
                        "window": window,
                        "colorbar": colorbar,
                        "fix_aspect_ratio": fix_aspect_ratio,
                    },
                )

                data_count += f"{count}"

            if (
                "plot" in triggers
                or center_x is None
                or center_y is None
                or width is None
                or height is None
            ):
                if figure["layout"]["xaxis"]["autorange"]:
                    x = np.array(figure["data"][0]["x"])
                    x_range = [np.amin(x), np.amax(x)]
                else:
                    x_range = figure["layout"]["xaxis"]["range"]
                width = x_range[1] - x_range[0]
                center_x = x_range[0] + (width / 2)
                if figure["layout"]["yaxis"]["autorange"]:
                    y = np.array(figure["data"][0]["y"])
                    y_range = [np.amin(y), np.amax(y)]
                else:
                    y_range = figure["layout"]["yaxis"]["range"]
                height = y_range[1] - y_range[0]
                center_y = y_range[0] + (height / 2)

        return figure, data_count, center_x, center_y, width, height, fix_aspect_ratio

    def get_general_inversion_params(
        self, new_workspace: Workspace, inversion_params_dict: dict
    ) -> dict:
        """
        Get topography, bounds, and models to add to self.params.

        :param new_workspace: New workspace to copy objects and data to.
        :param inversion_params_dict: Dictionary of params with radio button selection, data, constant, and object for
        each.

        :return param_dict: Dictionary with params to update self.params.
        """
        param_dict = {}
        for key, value in inversion_params_dict.items():
            param_dict[key + "_object"] = None
            param_dict[key] = None
            if value["options"] == "Model" or key == "topography":
                # Topography always saves an object and only saves data if the radio button is selected.
                if is_uuid(value["object"]):
                    obj = self.workspace.get_entity(uuid.UUID(value["object"]))[0]
                    param_dict[key + "_object"] = new_workspace.get_entity(obj.uid)[0]
                    if param_dict[key + "_object"] is None:
                        param_dict[key + "_object"] = obj.copy(
                            parent=new_workspace, copy_children=False
                        )
            if value["options"] in ["Model", "Data"]:
                if is_uuid(value["data"]) and param_dict[key + "_object"]:
                    param_dict[key] = self.workspace.get_entity(
                        uuid.UUID(value["data"])
                    )[0]
                    if (
                        param_dict[key] is not None
                        and new_workspace.get_entity(param_dict[key].uid)[0] is None
                    ):
                        param_dict[key].copy(parent=param_dict[key + "_object"])
            elif value["options"] == "Constant":
                param_dict[key] = value["const"]

        return param_dict

    def get_full_component_params(
        self,
        new_workspace: Workspace,
        data_object: ObjectBase,
        full_components: dict,
        forward_only: list,
    ) -> dict:
        """
        Get param_dict of values to add to self.params from full_components.

        :param new_workspace: New workspace to copy channel data to.
        :param data_object: Parent object for channel data.
        :param full_components: Dictionary with keys of component_options, and with values channel_bool, channel,
        uncertainty_type, uncertainty_floor, uncertainty_channel for each key.
        :param forward_only: Checkbox of whether to perform only forward inversion.

        :return param_dict: Dictionary of values to update self.params.
        """
        param_dict = {}
        for comp, value in full_components.items():
            # Only save channel_bool if forward_only.
            if value["channel_bool"]:
                param_dict[comp + "_channel_bool"] = True
                if not forward_only:
                    if is_uuid(value["channel"]):
                        param_dict[comp + "_channel"] = self.workspace.get_entity(
                            uuid.UUID(value["channel"])
                        )[0]
                        if (
                            param_dict[comp + "_channel"] is not None
                            and new_workspace.get_entity(
                                param_dict[comp + "_channel"].uid
                            )[0]
                            is None
                        ):
                            param_dict[comp + "_channel"].copy(parent=data_object)

                    # Determine whether to save uncertainty as floor or channel
                    param_dict[comp + "_uncertainty"] = 1.0
                    if value["uncertainty_type"] == "Floor":
                        param_dict[comp + "_uncertainty"] = value["uncertainty_floor"]
                    elif value["uncertainty_type"] == "Channel":
                        if is_uuid(value["uncertainty_channel"]):
                            param_dict[
                                comp + "_uncertainty"
                            ] = self.workspace.get_entity(
                                uuid.UUID(value["uncertainty_channel"])
                            )[
                                0
                            ]
                            if (
                                param_dict[comp + "_uncertainty"] is not None
                                and new_workspace.get_entity(
                                    param_dict[comp + "_uncertainty"].uid
                                )[0]
                                is None
                            ):
                                param_dict[comp + "_uncertainty"].copy(
                                    parent=data_object
                                )
            else:
                param_dict[comp + "_channel_bool"] = False

        return param_dict

    def get_inversion_params_dict(
        self, new_workspace: Workspace, update_dict: dict, data_object: ObjectBase
    ) -> dict:
        """
        Get parameters that are specific to inversion, that will be used to update self.params.

        :param new_workspace: New workspace to copy objects and data to.
        :param update_dict: Dictionary of new parameters and values from dash callback.
        :param data_object: Parent for channel data when copying to new workspace.

        :return param_dict: Dictionary of parameters ready to update self.params.
        """

        # Put together dict of needed params for the get_general_inversion_params function
        input_param_dict = {}
        for param in [
            "topography",
            "lower_bound",
            "upper_bound",
            "starting_model",
            "reference_model",
            "starting_inclination"
            "reference_inclination"
            "starting_declination"
            "reference_declination",
        ]:
            if param + "_options" in update_dict:
                input_param_dict[param] = {
                    "options": update_dict[param + "_options"],
                    "object": update_dict[param + "_object"],
                    "data": update_dict[param + "_data"],
                    "const": update_dict[param + "_const"],
                }

        param_dict = {}
        # Update topography, bounds, models
        param_dict.update(
            self.get_general_inversion_params(new_workspace, input_param_dict)
        )
        # Update channel params
        param_dict.update(
            self.get_full_component_params(
                new_workspace,
                data_object,
                update_dict["full_components"],
                update_dict["forward_only"],
            )
        )
        param_dict["azimuth"] = 0.0
        return param_dict

    def write_trigger(
        self,
        n_clicks: int,
        live_link: list,
        data_object: str,
        full_components: dict,
        resolution: float,
        window_center_x: float,
        window_center_y: float,
        window_width: float,
        window_height: float,
        fix_aspect_ratio: list,
        topography_options: str,
        topography_object: str,
        topography_data: str,
        topography_const: float,
        z_from_topo: list,
        receivers_offset_x: float,
        receivers_offset_y: float,
        receivers_offset_z: float,
        receivers_radar_drape: str,
        forward_only: list,
        starting_model_options: list,
        starting_model_object: str,
        starting_model_data: str,
        starting_model_const: float,
        mesh: str,
        reference_model_options: str,
        reference_model_object: str,
        reference_model_data: str,
        reference_model_const: float,
        alpha_s: float,
        alpha_x: float,
        alpha_y: float,
        alpha_z: float,
        s_norm: float,
        x_norm: float,
        y_norm: float,
        z_norm: float,
        lower_bound_options: str,
        lower_bound_object: str,
        lower_bound_data: str,
        lower_bound_const: float,
        upper_bound_options: str,
        upper_bound_object: str,
        upper_bound_data: str,
        upper_bound_const: float,
        detrend_type: str,
        detrend_order: int,
        ignore_values: str,
        max_iterations: int,
        chi_factor: float,
        initial_beta_ratio: float,
        max_cg_iterations: int,
        tol_cg: float,
        n_cpu: int,
        tile_spatial: int,
        out_group: str,
        monitoring_directory: str,
        inducing_field_strength: float = None,
        inducing_field_inclination: float = None,
        inducing_field_declination: float = None,
        starting_inclination_options: str = None,
        starting_inclination_object: str = None,
        starting_inclination_data: str = None,
        starting_inclination_const: float = None,
        reference_inclination_options: str = None,
        reference_inclination_object: str = None,
        reference_inclination_data: str = None,
        reference_inclination_const: float = None,
        starting_declination_options: str = None,
        starting_declination_object: str = None,
        starting_declination_data: str = None,
        starting_declination_const: float = None,
        reference_declination_options: str = None,
        reference_declination_object: str = None,
        reference_declination_data: str = None,
        reference_declination_const: float = None,
    ) -> (list, bool):
        """
        Update self.params and write out.

        :param n_clicks: Trigger for calling write_params.
        :param live_link: Checkbox showing whether monitoring directory is enabled.
        :param data_object: Input object uuid.
        :param full_components: Dictionary of components and corresponding channels, uncertainties, active.
        :param resolution: Resolution distance.
        :param window_center_x: Window center x.
        :param window_center_y: Window center y.
        :param window_width: Window width.
        :param window_height: Window height
        :param fix_aspect_ratio: Checkbox for plotting with fixed aspect ratio.
        :param topography_options: Type of topography selected (None, Data, Constant).
        :param topography_object: Topography object uuid.
        :param topography_data: Topography data uuid.
        :param topography_const: Topography constant.
        :param z_from_topo: Checkbox for getting z from topography.
        :param receivers_offset_x: Sensor offset East.
        :param receivers_offset_y: Sensor offset North.
        :param receivers_offset_z: Sensor offset up.
        :param receivers_radar_drape: Radar.
        :param forward_only: Checkbox for performing forward inversion.
        :param starting_model_options: Type of starting model selected (Model, Constant).
        :param starting_model_object: Starting model object uuid.
        :param starting_model_data: Starting model data uuid.
        :param starting_model_const: Starting model constant.
        :param mesh: Mesh object uuid.
        :param reference_model_options: Type of reference model selected (Model, Constant, None).
        :param reference_model_object: Reference model object uuid.
        :param reference_model_data: Reference model data uuid.
        :param reference_model_const: Reference model constant uuid.
        :param alpha_s: Scaling for reference model.
        :param alpha_x: Scaling for EW gradient.
        :param alpha_y: Scaling for NS gradient.
        :param alpha_z: Scaling for vertical gradient.
        :param s_norm: Lp-norm for reference model.
        :param x_norm: Lp-norm for EW gradient.
        :param y_norm: Lp-norm for NS gradient.
        :param z_norm: Lp-norm for vertical gradient.
        :param lower_bound_options: Type of lower bound selected (Model, Constant, None).
        :param lower_bound_object: Lower bound object uuid.
        :param lower_bound_data: Lower bound data uuid.
        :param lower_bound_const: Lower bound constant.
        :param upper_bound_options: Type of upper bound selected (Model, Constant, None).
        :param upper_bound_object: Upper bound object uuid.
        :param upper_bound_data: Upper bound data uuid.
        :param upper_bound_const: Upper bound constant.
        :param detrend_type: Detrend method (all, perimeter).
        :param detrend_order: Detrend order.
        :param ignore_values: Specified values to ignore.
        :param max_iterations: Max iterations.
        :param chi_factor: Target misfit.
        :param initial_beta_ratio: Beta ratio (phi_d/phi_m).
        :param max_cg_iterations: Max CG iterations.
        :param tol_cg: CG tolerance.
        :param n_cpu: Max CPUs.
        :param tile_spatial: Number of tiles.
        :param out_group: GA group name.
        :param monitoring_directory: Export path.
        :param inducing_field_strength: (Magnetic specific.) Inducing field strength (nT).
        :param inducing_field_inclination: (Magnetic specific.) Inducing field inclination.
        :param inducing_field_declination: (Magnetic specific.) Inducing field declination.
        :param starting_inclination_options: (Magnetic vector specific.) Type of starting inclination selected.
        :param starting_inclination_object: (Magnetic vector specific.) Starting model inclination object uuid.
        :param starting_inclination_data: (Magnetic vector specific.) Starting model inclination data uuid.
        :param starting_inclination_const: (Magnetic vector specific.) Starting model inclination constant.
        :param reference_inclination_options: (Magnetic vector specific.) Type of reference inclination selected.
        :param reference_inclination_object: (Magnetic vector specific.) Reference model inclination object uuid.
        :param reference_inclination_data: (Magnetic vector specific.) Reference model inclination data uuid.
        :param reference_inclination_const: (Magnetic vector specific.) Reference model inclination constant.
        :param starting_declination_options: (Magnetic vector specific.) Type of starting declination selected.
        :param starting_declination_object: (Magnetic vector specific.) Starting model declination object uuid.
        :param starting_declination_data: (Magnetic vector specific.) Starting model declination data uuid.
        :param starting_declination_const: (Magnetic vector specific.) Starting model declination constant.
        :param reference_declination_options: (Magnetic vector specific.) Type of Reference declination selected.
        :param reference_declination_object: (Magnetic vector specific.) Reference model declination object uuid.
        :param reference_declination_data: (Magnetic vector specific.) Reference model declination data uuid.
        :param reference_declination_const: (Magnetic vector specific.) Reference model declination constant.

        :return live_link: Checkbox showing whether monitoring directory is enabled.
        """

        if mesh is None:
            print("A mesh must be selected to write the input file.")
            return no_update
        if data_object is None:
            print("An object with data must be selected to write the input file.")
            return no_update
        if topography_object is None:
            print("A topography object must be selected to write the input file.")
            return no_update

        # Get dict of params from base dash application
        update_dict = {
            "resolution": resolution,
            "window_center_x": window_center_x,
            "window_center_y": window_center_y,
            "window_width": window_width,
            "window_height": window_height,
            "fix_aspect_ratio": fix_aspect_ratio,
            "z_from_topo": z_from_topo,
            "receivers_offset_x": receivers_offset_x,
            "receivers_offset_y": receivers_offset_y,
            "receivers_offset_z": receivers_offset_z,
            "receivers_radar_drape": receivers_radar_drape,
            "forward_only": forward_only,
            "alpha_s": alpha_s,
            "alpha_x": alpha_x,
            "alpha_y": alpha_y,
            "alpha_z": alpha_z,
            "s_norm": s_norm,
            "x_norm": x_norm,
            "y_norm": y_norm,
            "z_norm": z_norm,
            "detrend_type": detrend_type,
            "detrend_order": detrend_order,
            "ignore_values": ignore_values,
            "max_iterations": max_iterations,
            "chi_factor": chi_factor,
            "initial_beta_ratio": initial_beta_ratio,
            "max_cg_iterations": max_cg_iterations,
            "tol_cg": tol_cg,
            "n_cpu": n_cpu,
            "tile_spatial": tile_spatial,
            "out_group": out_group,
            "monitoring_directory": monitoring_directory,
            "inducing_field_strength": inducing_field_strength,
            "inducing_field_inclination": inducing_field_inclination,
            "inducing_field_declination": inducing_field_declination,
        }
        param_dict = self.get_params_dict(update_dict)

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
        temp_geoh5 = f"{out_group}_{time():.0f}.geoh5"
        ws, live_link = BaseApplication.get_output_workspace(
            live_link, monitoring_directory, temp_geoh5
        )
        if not live_link:
            param_dict["monitoring_directory"] = ""

        with ws as workspace:
            # Put entities in output workspace.
            param_dict["geoh5"] = workspace

            # Copy mesh to workspace
            mesh = self.workspace.get_entity(uuid.UUID(mesh))[0]
            param_dict["mesh"] = workspace.get_entity(mesh.uid)[0]
            if param_dict["mesh"] is None:
                param_dict["mesh"] = mesh.copy(parent=workspace, copy_children=False)

            # Copy data object to workspace
            data_object = self.workspace.get_entity(uuid.UUID(data_object))[0]
            param_dict["data_object"] = workspace.get_entity(data_object.uid)[0]
            if param_dict["data_object"] is None:
                param_dict["data_object"] = data_object.copy(
                    parent=workspace, copy_children=False
                )

            # Add inversion specific params to param_dict
            param_dict.update(
                self.get_inversion_params_dict(
                    workspace, locals(), param_dict["data_object"]
                )
            )

            if is_uuid(receivers_radar_drape):
                param_dict["receivers_radar_drape"] = self.workspace.get_entity(
                    uuid.UUID(receivers_radar_drape)
                )[0]
                if (
                    param_dict["receivers_radar_drape"] is not None
                    and workspace.get_entity(param_dict["receivers_radar_drape"].uid)[0]
                    is None
                ):
                    param_dict["receivers_radar_drape"].copy(
                        parent=param_dict["data_object"]
                    )

            if self._inversion_type == "dcip":
                param_dict["resolution"] = None  # No downsampling for dcip

            self._run_params = self.params.__class__(**param_dict)
            self._run_params.write_input_file(
                name=temp_geoh5.replace(".geoh5", ".ui.json"),
                path=monitoring_directory,
            )

            if self._run_params.monitoring_directory is not None and os.path.exists(
                os.path.abspath(self._run_params.monitoring_directory)
            ):
                for obj in ws.objects:
                    monitored_directory_copy(
                        os.path.abspath(self._run_params.monitoring_directory),
                        obj,
                    )

        if live_link:
            print("Live link active. Check your ANALYST session for new mesh.")
            return [True]
        else:
            print("Saved to " + os.path.abspath(monitoring_directory))
            return []

    def trigger_click(self, _):
        """
        Triggered from clicking compute button. Checks parameters and runs inversion.
        """
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
