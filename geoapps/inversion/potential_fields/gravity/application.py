#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


from __future__ import annotations

import os

from dash import Input, Output, State
from flask import Flask
from jupyter_dash import JupyterDash

from geoapps.inversion.base_inversion_application import InversionApp
from geoapps.inversion.potential_fields.gravity.constants import app_initializer
from geoapps.inversion.potential_fields.gravity.layout import (
    gravity_inversion_params,
    gravity_layout,
)
from geoapps.inversion.potential_fields.gravity.params import GravityParams


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
        # Update components from forward only checkbox
        for param in gravity_inversion_params:
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
            # Plot
            Output(component_id="fix_aspect_ratio", component_property="value"),
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
            Input(component_id="ui_json_data", component_property="data"),
        )(self.update_remainder_from_ui_json)

        # Plot callbacks
        # Update plot
        self.app.callback(
            Output(component_id="plot", component_property="figure"),
            Output(component_id="data_count", component_property="children"),
            Input(component_id="ui_json_data", component_property="data"),
            Input(component_id="plot", component_property="figure"),
            Input(component_id="plot", component_property="relayoutData"),
            Input(component_id="data_object", component_property="value"),
            Input(component_id="channel", component_property="value"),
            Input(component_id="resolution", component_property="value"),
            Input(component_id="colorbar", component_property="value"),
            Input(component_id="fix_aspect_ratio", component_property="value"),
        )(self.plot_selection)

        # Button callbacks
        self.app.callback(
            Output(component_id="live_link", component_property="value"),
            Input(component_id="write_input", component_property="n_clicks"),
            State(component_id="live_link", component_property="value"),
            # Data Selection
            State(component_id="data_object", component_property="value"),
            State(component_id="full_components", component_property="data"),
            State(component_id="resolution", component_property="value"),
            State(component_id="plot", component_property="figure"),
            State(component_id="fix_aspect_ratio", component_property="value"),
            # Topography
            State(component_id="topography_options", component_property="value"),
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
            State(component_id="starting_density_options", component_property="value"),
            State(component_id="starting_density_object", component_property="value"),
            State(component_id="starting_density_data", component_property="value"),
            State(component_id="starting_density_const", component_property="value"),
            # Mesh
            State(component_id="mesh", component_property="value"),
            # Reference Model
            State(component_id="reference_density_options", component_property="value"),
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
            State(component_id="lower_bound_options", component_property="value"),
            State(component_id="lower_bound_object", component_property="value"),
            State(component_id="lower_bound_data", component_property="value"),
            State(component_id="lower_bound_const", component_property="value"),
            State(component_id="upper_bound_options", component_property="value"),
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
            State(component_id="monitoring_directory", component_property="value"),
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
        )(InversionApp.open_mesh_app)
