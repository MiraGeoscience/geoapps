#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


from __future__ import annotations

import os
import uuid
from time import time

import matplotlib
import numpy as np
from dash import Input, Output, State, callback_context, no_update
from flask import Flask
from geoh5py.data import ReferencedData
from geoh5py.objects import BlockModel, Curve, Grid2D, Octree, Points, Surface
from geoh5py.shared import Entity
from geoh5py.shared.utils import is_uuid
from geoh5py.ui_json import InputFile
from geoh5py.workspace import Workspace
from jupyter_dash import JupyterDash
from matplotlib import colors
from notebook import notebookapp
from plotly import graph_objects as go

from geoapps.base.application import BaseApplication
from geoapps.base.dash_application import BaseDashApplication
from geoapps.base.selection import TopographyOptions
from geoapps.inversion.base_inversion_application import InversionApp
from geoapps.inversion.potential_fields.gravity.constants import app_initializer
from geoapps.inversion.potential_fields.gravity.layout import (
    gravity_inversion_params,
    gravity_layout,
)
from geoapps.inversion.potential_fields.gravity.params import GravityParams
from geoapps.inversion.potential_fields.magnetic_scalar.params import (
    MagneticScalarParams,
)
from geoapps.inversion.potential_fields.magnetic_vector.params import (
    MagneticVectorParams,
)
from geoapps.shared_utils.utils import filter_xy


class GravityApp(InversionApp):
    """
    Application for the inversion of potential field data using SimPEG
    """

    _param_class = GravityParams
    _inversion_type = "gravity"
    _inversion_params = gravity_inversion_params

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

        self.app.layout = gravity_layout

        # Callbacks relating to layout
        self.app.callback(
            Output(component_id="uncertainty_floor", component_property="style"),
            Output(component_id="uncertainty_channel", component_property="style"),
            Input(component_id="uncertainty_options", component_property="value"),
        )(InversionApp.update_uncertainty_visibility)
        self.app.callback(
            Output(component_id="topography_none_div", component_property="style"),
            Output(component_id="topography_object_div", component_property="style"),
            Output(component_id="topography_constant_div", component_property="style"),
            Input(component_id="topography_options", component_property="value"),
        )(InversionApp.update_topography_visibility)
        for model_type in ["starting", "reference"]:
            for param in gravity_inversion_params:
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

        self.app.callback(
            Output(component_id="mesh_object", component_property="options"),
            Input(component_id="data_object", component_property="options"),
        )(InversionApp.update_remaining_object_options)
        self.app.callback(
            Output(component_id="topography_object", component_property="options"),
            Input(component_id="data_object", component_property="options"),
        )(InversionApp.update_remaining_object_options)
        self.app.callback(
            Output(component_id="topography_data", component_property="options"),
            # Output(component_id=param+"_data", component_property="value"),
            Input(component_id="ui_json_data", component_property="data"),
            Input(component_id="topography_object", component_property="value"),
        )(self.update_channel_options)

        for model_type in ["starting", "reference"]:
            for param in gravity_inversion_params:
                self.app.callback(
                    Output(
                        component_id=model_type + "_" + param + "_object",
                        component_property="options",
                    ),
                    Input(component_id="data_object", component_property="options"),
                )(InversionApp.update_remaining_object_options)
                self.app.callback(
                    Output(
                        component_id=model_type + "_" + param + "_data",
                        component_property="options",
                    ),
                    # Output(component_id=param+"_data", component_property="value"),
                    Input(component_id="ui_json_data", component_property="data"),
                    Input(
                        component_id=model_type + "_" + param + "_object",
                        component_property="value",
                    ),
                )(self.update_channel_options)

        self.app.callback(
            Output(component_id="channel", component_property="options"),
            # Output(component_id="channel", component_property="value"),
            Input(component_id="ui_json_data", component_property="data"),
            Input(component_id="data_object", component_property="value"),
        )(self.update_channel_options)
        self.app.callback(
            Output(component_id="uncertainty_channel", component_property="options"),
            # Output(component_id="uncertainty_channel", component_property="value"),
            Input(component_id="ui_json_data", component_property="data"),
            Input(component_id="data_object", component_property="value"),
        )(self.update_channel_options)
        self.app.callback(
            Output(component_id="receivers_radar_drape", component_property="options"),
            Output(component_id="receivers_radar_drape", component_property="value"),
            Input(component_id="ui_json_data", component_property="data"),
            Input(component_id="data_object", component_property="value"),
        )(self.update_data_options)

        # Update input data channel and uncertainties from component
        self.app.callback(
            Output(component_id="full_components", component_property="data"),
            Input(component_id="ui_json_data", component_property="data"),
            Input(component_id="full_components", component_property="data"),
            Input(component_id="channel_bool", component_property="value"),
            Input(component_id="channel", component_property="value"),
            Input(component_id="uncertainty_options", component_property="value"),
            Input(component_id="uncertainty_floor", component_property="value"),
            Input(component_id="uncertainty_channel", component_property="value"),
            State(component_id="component", component_property="value"),
            State(component_id="component", component_property="options"),
        )(self.update_full_components)
        self.app.callback(
            Output(component_id="channel_bool", component_property="value"),
            Output(component_id="channel", component_property="value"),
            Output(component_id="uncertainty_options", component_property="value"),
            Output(component_id="uncertainty_floor", component_property="value"),
            Output(component_id="uncertainty_channel", component_property="value"),
            Input(component_id="component", component_property="value"),
            Input(component_id="full_components", component_property="data"),
        )(self.update_input_channel)

        for model_type in ["starting", "reference"]:
            for param in gravity_inversion_params:
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
                        component_id=model_type + "_" + param + "_data",
                        component_property="value",
                    ),
                    Input(component_id="ui_json_data", component_property="data"),
                )(self.update_inversion_params_from_ui_json)
        for param in ["topography", "lower_bound", "upper_bound"]:
            self.app.callback(
                Output(component_id=param + "_options", component_property="value"),
                Output(component_id=param + "_const", component_property="value"),
                Output(component_id=param + "_object", component_property="value"),
                Output(component_id=param + "_data", component_property="value"),
                Input(component_id="ui_json_data", component_property="data"),
            )(InversionApp.update_general_param_from_ui_json)

        # Update from ui.json
        self.app.callback(
            # Input Data
            Output(component_id="resolution", component_property="value"),
            # Topography
            Output(component_id="z_from_topo", component_property="value"),
            Output(component_id="receivers_offset_x", component_property="value"),
            Output(component_id="receivers_offset_y", component_property="value"),
            Output(component_id="receivers_offset_z", component_property="value"),
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
            Input(component_id="ui_json_data", component_property="data"),
        )(self.update_remainder_from_ui_json)

        # Plot callbacks
        # Update plot
        self.app.callback(
            Output(component_id="plot", component_property="figure"),
            Output(component_id="data_count", component_property="children"),
            Input(component_id="ui_json_data", component_property="data"),
            Input(component_id="plot", component_property="figure"),
            Input(component_id="data_object", component_property="value"),
            Input(component_id="channel", component_property="value"),
            Input(component_id="resolution", component_property="value"),
            Input(component_id="azimuth", component_property="value"),
            Input(component_id="colorbar", component_property="value"),
            Input(component_id="fix_aspect_ratio", component_property="value"),
        )(self.plot_selection)

        # Button callbacks
        self.app.callback(
            Output(component_id="live_link", component_property="value"),
            Input(component_id="write_input", component_property="n_clicks"),
            State(component_id="live_link", component_property="value"),
            # Object Selection
            State(component_id="data_object", component_property="value"),
            # Input Data
            State(component_id="full_components", component_property="data"),
            State(component_id="resolution", component_property="value"),
            State(component_id="plot", component_property="figure"),
            State(component_id="colorbar", component_property="value"),
            State(component_id="fix_aspect_ratio", component_property="value"),
            # Topography
            State(component_id="topography_object", component_property="value"),
            State(component_id="topography_data", component_property="value"),
            State(component_id="topography_const", component_property="value"),
            State(component_id="z_from_topo", component_property="value"),
            State(component_id="receivers_offset_x", component_property="value"),
            State(component_id="receivers_offset_y", component_property="value"),
            State(component_id="receivers_offset_z", component_property="value"),
            State(component_id="receivers_radar_drape", component_property="value"),
            # Inversion Parameters
            State(component_id="forward_only", component_property="value"),
            # Starting Model
            State(component_id="starting_density_object", component_property="value"),
            State(component_id="starting_density_data", component_property="value"),
            State(component_id="starting_density_const", component_property="value"),
            # Mesh
            # State(component_id="mesh_object", component_property="value"),
            # Reference Model
            State(component_id="reference_density_object", component_property="value"),
            State(component_id="reference_density_data", component_property="value"),
            State(component_id="reference_density_const", component_property="value"),
            # Regularization
            State(component_id="alpha_s", component_property="value"),
            State(component_id="alpha_x", component_property="value"),
            State(component_id="alpha_y", component_property="value"),
            State(component_id="alpha_z", component_property="value"),
            State(component_id="s_norm", component_property="value"),
            State(component_id="x_norm", component_property="value"),
            State(component_id="y_norm", component_property="value"),
            State(component_id="z_norm", component_property="value"),
            # Upper-Lower Bounds
            State(component_id="lower_bound_object", component_property="value"),
            State(component_id="lower_bound_data", component_property="value"),
            State(component_id="lower_bound_const", component_property="value"),
            State(component_id="upper_bound_object", component_property="value"),
            State(component_id="upper_bound_data", component_property="value"),
            State(component_id="upper_bound_const", component_property="value"),
            # Detrend
            State(component_id="detrend_type", component_property="value"),
            State(component_id="detrend_order", component_property="value"),
            # Ignore Values
            State(component_id="ignore_values", component_property="value"),
            # Optimization
            State(component_id="max_iterations", component_property="value"),
            State(component_id="chi_factor", component_property="value"),
            State(component_id="initial_beta_ratio", component_property="value"),
            State(component_id="max_cg_iterations", component_property="value"),
            State(component_id="tol_cg", component_property="value"),
            State(component_id="n_cpu", component_property="value"),
            State(component_id="tile_spatial", component_property="value"),
            # Output
            State(component_id="out_group", component_property="value"),
            State(component_id="export_directory", component_property="value"),
            prevent_initial_call=True,
        )(self.write_trigger)
        self.app.callback(
            Output(component_id="compute", component_property="n_clicks"),
            Input(component_id="compute", component_property="n_clicks"),
            prevent_initial_call=True,
        )(self.trigger_click)
        self.app.callback(
            Output(component_id="open_mesh", component_property="n_clicks"),
            Input(component_id="open_mesh", component_property="n_clicks"),
            prevent_initial_call=True,
        )(self.open_mesh_app)

    def write_trigger(
        self,
        n_clicks,
        live_link,
        data_object,
        full_components,
        resolution,
        plot,
        colorbar,
        fix_aspect_ratio,
        topography_object,
        topography_data,
        topography_const,
        z_from_topo,
        receivers_offset_x,
        receivers_offset_y,
        receivers_offset_z,
        receivers_radar_drape,
        forward_only,
        starting_density_object,
        starting_density_data,
        starting_density_const,
        mesh_object,
        reference_density_object,
        reference_density_data,
        reference_density_const,
        alpha_s,
        alpha_x,
        alpha_y,
        alpha_z,
        s_norm,
        x_norm,
        y_norm,
        z_norm,
        lower_bound_object,
        lower_bound_data,
        lower_bound_const,
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
        export_directory,
    ):
        # Widgets values populate params dictionary
        param_dict = self.get_params_dict(locals())

        # Create a new workspace and copy objects into it
        temp_geoh5 = f"{ga_group_name}_{time():.0f}.geoh5"
        ws, live_link = BaseApplication.get_output_workspace(
            live_link, export_directory, temp_geoh5
        )
        with ws as new_workspace:
            param_dict["geoh5"] = new_workspace

            for elem in [
                "topography",
                "starting_model",
                "reference_model",
                "lower_bound",
                "upper_bound",
            ]:
                param_dict[elem + "_object"] = None
                param_dict[elem] = None
                if locals()[elem + "_options"] == "Object":
                    obj, data = locals()[elem + "_object"], locals()[elem + "_data"]

                    if obj is not None:
                        new_obj = new_workspace.get_entity(obj.uid)[0]
                        if new_obj is None:
                            new_obj = obj.copy(
                                parent=new_workspace, copy_children=False
                            )
                        for d in data:
                            if new_workspace.get_entity(d.uid)[0] is None:
                                d.copy(parent=new_obj)

                    param_dict[elem + "_object"] = locals()[elem + "_object"]
                    param_dict[elem] = locals()[elem + "_data"]

                elif locals()[elem + "_options"] == "Constant":
                    param_dict[elem] = locals()[elem + "_const"]

            new_obj = new_workspace.get_entity(uuid.UUID(data_object))
            if len(new_obj) == 0 or new_obj[0] is None:
                print("An object with data must be selected to write the input file.")
                return

            new_obj = new_obj[0]

            for comp, value in full_components.items():
                if value["channel_bool"]:
                    if not forward_only:
                        self.workspace.get_entity(value["channel"])[0].copy(
                            parent=new_obj
                        )

                if value["uncertainty_type"] == "Floor":
                    param_dict[comp + "_uncertainty"] = value["uncertainty_floor"]
                elif value["uncertainty_type"] == "Channel":
                    if (
                        new_workspace.get_entity(value["uncertainty_channel"])[0]
                        is None
                    ):
                        self.workspace.get_entity(value["uncertainty_channel"])[0].copy(
                            parent=new_obj, copy_children=False
                        )
                else:
                    param_dict[comp + "_uncertainty"] = None

            if receivers_radar_drape is not None:
                self.workspace.get_entity(receivers_radar_drape)[0].copy(parent=new_obj)

            # Create new params object and write
            ifile = InputFile(
                ui_json=self.params.input_file.ui_json,
                validation_options={"disabled": True},
            )

            param_dict["resolution"] = None  # No downsampling for dcip
            self._run_params = self.params.__class__(input_file=ifile, **param_dict)
            self._run_params.write_input_file(
                name=temp_geoh5.replace(".geoh5", ".ui.json"),
                path=export_directory,
            )
