#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

# https://stackoverflow.com/questions/49851280/showing-a-simple-matplotlib-plot-in-plotly-dash

from __future__ import annotations

import os
import warnings
from time import time

import numpy as np
from dash import Input, Output
from flask import Flask
from geoh5py.objects import BlockModel, Curve, Octree, Points, Surface
from geoh5py.shared import Entity
from geoh5py.ui_json import InputFile
from geoh5py.workspace import Workspace
from jupyter_dash import JupyterDash

from geoapps.base.application import BaseApplication
from geoapps.base.plot import PlotSelection2D
from geoapps.base.selection import ObjectDataSelection, TopographyOptions
from geoapps.inversion.potential_fields.gravity.params import GravityParams
from geoapps.inversion.potential_fields.layout import potential_fields_layout
from geoapps.inversion.potential_fields.magnetic_scalar.params import (
    MagneticScalarParams,
)
from geoapps.inversion.potential_fields.magnetic_vector.constants import app_initializer
from geoapps.inversion.potential_fields.magnetic_vector.params import (
    MagneticVectorParams,
)
from geoapps.utils import geophysical_systems, warn_module_not_found
from geoapps.utils.list import find_value
from geoapps.utils.string import string_2_list


class InversionApp:
    """
    Application for the inversion of potential field data using SimPEG
    """

    _param_class = MagneticVectorParams
    _select_multiple = True
    _add_groups = False
    _run_params = None
    _sensor = None
    _topography = None
    inversion_parameters = None
    defaults = {}

    def __init__(self, ui_json=None, **kwargs):
        app_initializer.update(kwargs)
        if ui_json is not None and os.path.exists(ui_json.path):
            self.params = self._param_class(ui_json)
        else:
            self.params = self._param_class(**app_initializer)

        # super().__init__(**self.params.to_dict())

        external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
        server = Flask(__name__)
        self.app = JupyterDash(
            server=server,
            url_base_pathname=os.environ.get("JUPYTERHUB_SERVICE_PREFIX", "/"),
            external_stylesheets=external_stylesheets,
        )

        self.app.layout = potential_fields_layout

        # Callbacks relating to layout
        self.app.callback(
            Output(component_id="inducing_params_div", component_property="style"),
            Input(component_id="inversion_type", component_property="value"),
        )(InversionApp.update_inducing_params_visibility)
        self.app.callback(
            Output(component_id="topography_none", component_property="style"),
            Output(component_id="topography_object", component_property="style"),
            Output(component_id="topography_sensor", component_property="style"),
            Output(component_id="topography_constant", component_property="style"),
            Input(component_id="topography_options", component_property="value"),
        )(InversionApp.update_topography_visibility)

        for model_type in ["starting", "ref"]:
            for param in ["susceptibility", "inclination", "declination"]:
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
            Output(component_id="lower_bounds_const_div", component_property="style"),
            Output(component_id="lower_bounds_mod_div", component_property="style"),
            Input(component_id="lower_bounds_options", component_property="value"),
        )(InversionApp.update_model_visibility)
        self.app.callback(
            Output(component_id="upper_bounds_const_div", component_property="style"),
            Output(component_id="upper_bounds_mod_div", component_property="style"),
            Input(component_id="upper_bounds_options", component_property="value"),
        )(InversionApp.update_model_visibility)

        self.app.callback(
            Output(component_id="starting_model_div", component_property="style"),
            Output(component_id="mesh_div", component_property="style"),
            Output(component_id="reference_model_div", component_property="style"),
            Output(component_id="regularization_div", component_property="style"),
            Output(component_id="upper_lower_bounds_div", component_property="style"),
            Output(component_id="detrend_div", component_property="style"),
            Output(component_id="ignore_values_div", component_property="style"),
            Output(component_id="optimization_div", component_property="style"),
            Input(component_id="param_dropdown", component_property="value"),
        )(InversionApp.update_inversion_params_visibility)

        # Update from changing inversion type
        self.app.callback(
            Output(component_id="component", component_property="options"),
            Input(component_id="inversion_type", component_property="value"),
        )(self.update_component_list)

        # Update mesh
        self.app.callback(
            Input(component_id="window_width", component_property="value"),
            Input(component_id="window_height", component_property="value"),
            Input(component_id="resolution", component_property="value"),
        )(self.update_octree_param)

        # Callbacks for clicking buttons
        self.app.callback(
            Input(component_id="write_input", component_property="n_clicks"),
        )(self.write_trigger)
        self.app.callback(
            Input(component_id="compute", component_property="n_clicks"),
        )(self.trigger_click)

    @staticmethod
    def update_inducing_params_visibility(selection):
        if selection == "magnetic vector" or selection == "magnetic scalar":
            return {"display": "inline-block"}
        else:
            return {"display": "none"}

    @staticmethod
    def update_topography_visibility(selection):
        if selection == "None":
            return (
                {"display": "block"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
            )
        elif selection == "Object":
            return (
                {"display": "none"},
                {"display": "block"},
                {"display": "none"},
                {"display": "none"},
            )
        elif selection == "Relative to Sensor":
            return (
                {"display": "none"},
                {"display": "none"},
                {"display": "block"},
                {"display": "none"},
            )
        elif selection == "Constant":
            return (
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "block"},
            )

    @staticmethod
    def update_model_visibility(selection):
        if selection == "Constant":
            return {"display": "block"}, {"display": "none"}
        elif selection == "Model":
            return {"display": "none"}, {"display": "block"}
        elif selection == "None":
            return {"display": "none"}, {"display": "none"}

    @staticmethod
    def update_inversion_params_visibility(selection):
        if selection == "starting model":
            return (
                {"display": "inline-block"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
            )
        elif selection == "mesh":
            return (
                {"display": "none"},
                {"display": "inline-block"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
            )
        elif selection == "reference model":
            return (
                {"display": "none"},
                {"display": "none"},
                {"display": "inline-block"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
            )
        elif selection == "regularization":
            return (
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "inline-block"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
            )
        elif selection == "upper-lower bounds":
            return (
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "inline-block"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
            )
        elif selection == "detrend":
            return (
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "inline-block"},
                {"display": "none"},
                {"display": "none"},
            )
        elif selection == "ignore values":
            return (
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "inline-block"},
                {"display": "none"},
            )
        elif selection == "optimization":
            return (
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "inline-block"},
            )

    def update_component_list(self, inversion_type):
        if self.inversion_type.value in ["magnetic vector", "magnetic scalar"]:
            data_type_list = [
                "tmi",
                "bx",
                "by",
                "bz",
                "bxx",
                "bxy",
                "bxz",
                "byy",
                "byz",
                "bzz",
            ]
        else:
            data_type_list = [
                "gx",
                "gy",
                "gz",
                "gxx",
                "gxy",
                "gxz",
                "gyy",
                "gyz",
                "gzz",
                "uv",
            ]
        return data_type_list

    def update_octree_param(self, window_width, window_height, resolution):
        dl = resolution
        self._mesh_octree.u_cell_size.value = f"{dl/2:.0f}"
        self._mesh_octree.v_cell_size.value = f"{dl / 2:.0f}"
        self._mesh_octree.w_cell_size.value = f"{dl / 2:.0f}"
        self._mesh_octree.depth_core.value = np.ceil(
            np.min([window_width, window_height]) / 2.0
        )
        self._mesh_octree.horizontal_padding.value = (
            np.max([window_width, window_width]) / 2
        )
        resolution.indices = None
        self.write.button_style = "warning"
        self._run_params = None
        self.trigger.button_style = "danger"

    def write_trigger(self, _):
        # Widgets values populate params dictionary
        # *** populate self.params with component values
        param_dict = {}

        # Create a new workapce and copy objects into it
        temp_geoh5 = f"{self.ga_group_name.value}_{time():.0f}.geoh5"
        ws, self.live_link.value = BaseApplication.get_output_workspace(
            self.live_link.value, self.export_directory.selected_path, temp_geoh5
        )
        with ws as new_workspace:

            param_dict["geoh5"] = new_workspace

            for elem in [
                self,
                self._mesh_octree,
                self._topography_group,
                self._starting_model_group,
                self._reference_model_group,
                self._lower_bound_group,
                self._upper_bound_group,
            ]:
                obj, data = elem.get_selected_entities()

                if obj is not None:
                    new_obj = new_workspace.get_entity(obj.uid)[0]
                    if new_obj is None:
                        new_obj = obj.copy(parent=new_workspace, copy_children=False)
                    for d in data:
                        if new_workspace.get_entity(d.uid)[0] is None:
                            d.copy(parent=new_obj)

            if self.inversion_type.value == "magnetic vector":
                for elem in [
                    self._starting_inclination_group,
                    self._starting_declination_group,
                    self._reference_inclination_group,
                    self._reference_declination_group,
                ]:
                    obj, data = elem.get_selected_entities()
                    if obj is not None:
                        new_obj = new_workspace.get_entity(obj.uid)[0]
                        if new_obj is None:
                            new_obj = obj.copy(
                                parent=new_workspace, copy_children=False
                            )
                        for d in data:
                            if new_workspace.get_entity(d.uid)[0] is None:
                                d.copy(parent=new_obj)

            new_obj = new_workspace.get_entity(self.objects.value)
            if len(new_obj) == 0 or new_obj[0] is None:
                print("An object with data must be selected to write the input file.")
                return

            new_obj = new_obj[0]
            for key in self.data_channel_choices.options:
                widget = getattr(self, f"{key}_uncertainty_channel")
                if widget.value is not None:
                    param_dict[f"{key}_uncertainty"] = str(widget.value)
                    if new_workspace.get_entity(widget.value)[0] is None:
                        self.workspace.get_entity(widget.value)[0].copy(
                            parent=new_obj, copy_children=False
                        )
                else:
                    widget = getattr(self, f"{key}_uncertainty_floor")
                    param_dict[f"{key}_uncertainty"] = widget.value

                if getattr(self, f"{key}_channel_bool").value:
                    if not self.forward_only.value:
                        self.workspace.get_entity(
                            getattr(self, f"{key}_channel").value
                        )[0].copy(parent=new_obj)

            if self.receivers_radar_drape.value is not None:
                self.workspace.get_entity(self.receivers_radar_drape.value)[0].copy(
                    parent=new_obj
                )

            for key in self.__dict__:
                attr = getattr(self, key)
                if isinstance(attr, Widget) and hasattr(attr, "value"):
                    value = attr.value
                    if isinstance(value, uuid.UUID):
                        value = new_workspace.get_entity(value)[0]
                    if hasattr(self.params, key):
                        param_dict[key.lstrip("_")] = value
                else:
                    sub_keys = []
                    if isinstance(attr, (ModelOptions, TopographyOptions)):
                        sub_keys = [attr.identifier + "_object", attr.identifier]
                        attr = self
                    elif isinstance(attr, (MeshOctreeOptions, SensorOptions)):
                        sub_keys = attr.params_keys
                    for sub_key in sub_keys:
                        value = getattr(attr, sub_key)
                        if isinstance(value, Widget) and hasattr(value, "value"):
                            value = value.value
                        if isinstance(value, uuid.UUID):
                            value = new_workspace.get_entity(value)[0]

                        if hasattr(self.params, sub_key):
                            param_dict[sub_key.lstrip("_")] = value

            # Create new params object and write
            ifile = InputFile(
                ui_json=self.params.input_file.ui_json,
                validation_options={"disabled": True},
            )
            param_dict["geoh5"] = new_workspace
            param_dict["resolution"] = None  # No downsampling for dcip
            self._run_params = self.params.__class__(input_file=ifile, **param_dict)
            self._run_params.write_input_file(
                name=temp_geoh5.replace(".geoh5", ".ui.json"),
                path=self.export_directory.selected_path,
            )

        self.write.button_style = ""
        self.trigger.button_style = "success"

    def trigger_click(self, _):
        """"""
        if self._run_params is None:
            warnings.warn("Input file must be written before running.")
            return

        self.run(self._run_params)
        self.trigger.button_style = ""

    @staticmethod
    def run(params):
        """
        Trigger the inversion.
        """
        if not isinstance(
            params, (MagneticVectorParams, MagneticScalarParams, GravityParams)
        ):
            raise ValueError(
                "Parameter 'inversion_type' must be one of "
                "'magnetic vector', 'magnetic scalar' or 'gravity'"
            )

        os.system(
            "start cmd.exe @cmd /k "
            + f"python -m {params.run_command} "
            + f'"{params.input_file.path_name}"'
        )


app = InversionApp()
app.app.run_server(host="127.0.0.1", port=8050, debug=True)
