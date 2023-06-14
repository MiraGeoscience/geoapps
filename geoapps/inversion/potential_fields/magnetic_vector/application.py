#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


from __future__ import annotations

import pathlib

from dash import Input, Output, State

from geoapps.inversion.base_inversion_application import InversionApp
from geoapps.inversion.potential_fields.magnetic_vector.constants import app_initializer
from geoapps.inversion.potential_fields.magnetic_vector.layout import (
    magnetic_vector_inversion_params,
    magnetic_vector_layout,
)
from geoapps.inversion.potential_fields.magnetic_vector.params import (
    MagneticVectorParams,
)


class MagneticVectorApp(InversionApp):
    """
    Application for the inversion of potential field data using SimPEG
    """

    _param_class = MagneticVectorParams
    _inversion_type = "magnetic vector"
    _inversion_params = magnetic_vector_inversion_params
    _layout = magnetic_vector_layout

    def __init__(self, ui_json=None, **kwargs):
        app_initializer.update(kwargs)
        if ui_json is not None and pathlib.Path(ui_json.path).exists():
            self.params = self._param_class(ui_json)
        else:
            self.params = self._param_class(**app_initializer)

        super().__init__()

        # Update from ui.json
        self.app.callback(
            Output(component_id="inducing_field_strength", component_property="value"),
            Output(
                component_id="inducing_field_inclination", component_property="value"
            ),
            Output(
                component_id="inducing_field_declination", component_property="value"
            ),
            Input(component_id="ui_json_data", component_property="data"),
        )(self.update_remainder_from_ui_json)

        # Write params
        self.app.callback(
            Output(component_id="live_link", component_property="value"),
            Input(component_id="write_input", component_property="n_clicks"),
            State(component_id="live_link", component_property="value"),
            # Data Selection
            State(component_id="data_object", component_property="value"),
            State(component_id="full_components", component_property="data"),
            State(component_id="resolution", component_property="value"),
            State(component_id="window_center_x", component_property="value"),
            State(component_id="window_center_y", component_property="value"),
            State(component_id="window_width", component_property="value"),
            State(component_id="window_height", component_property="value"),
            State(component_id="fix_aspect_ratio", component_property="value"),
            State(component_id="colorbar", component_property="value"),
            # Topography
            State(component_id="topography_object", component_property="value"),
            State(component_id="topography", component_property="value"),
            State(component_id="z_from_topo", component_property="value"),
            State(component_id="receivers_offset_z", component_property="value"),
            State(component_id="receivers_radar_drape", component_property="value"),
            # Inversion Parameters
            State(component_id="forward_only", component_property="value"),
            # Starting Model
            State(component_id="starting_model_options", component_property="value"),
            State(component_id="starting_model_data", component_property="value"),
            State(component_id="starting_model_const", component_property="value"),
            # Mesh
            State(component_id="mesh", component_property="value"),
            # Reference Model
            State(component_id="reference_model_options", component_property="value"),
            State(component_id="reference_model_data", component_property="value"),
            State(component_id="reference_model_const", component_property="value"),
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
            State(component_id="lower_bound_data", component_property="value"),
            State(component_id="lower_bound_const", component_property="value"),
            State(component_id="upper_bound_options", component_property="value"),
            State(component_id="upper_bound_data", component_property="value"),
            State(component_id="upper_bound_const", component_property="value"),
            # Detrend
            State(component_id="detrend_type", component_property="value"),
            State(component_id="detrend_order", component_property="value"),
            # Ignore Values
            State(component_id="ignore_values", component_property="value"),
            # Optimization
            State(component_id="max_global_iterations", component_property="value"),
            State(component_id="max_irls_iterations", component_property="value"),
            State(component_id="coolingRate", component_property="value"),
            State(component_id="coolingFactor", component_property="value"),
            State(component_id="chi_factor", component_property="value"),
            State(component_id="initial_beta_ratio", component_property="value"),
            State(component_id="max_cg_iterations", component_property="value"),
            State(component_id="tol_cg", component_property="value"),
            State(component_id="n_cpu", component_property="value"),
            State(component_id="store_sensitivities", component_property="value"),
            State(component_id="tile_spatial", component_property="value"),
            # Output
            State(component_id="out_group", component_property="value"),
            State(component_id="monitoring_directory", component_property="value"),
            # Magnetic specific
            State(component_id="inducing_field_strength", component_property="value"),
            State(
                component_id="inducing_field_inclination", component_property="value"
            ),
            State(
                component_id="inducing_field_declination", component_property="value"
            ),
            # Magnetic vector specific
            State(
                component_id="starting_inclination_options", component_property="value"
            ),
            State(component_id="starting_inclination_data", component_property="value"),
            State(
                component_id="starting_inclination_const", component_property="value"
            ),
            State(
                component_id="reference_inclination_options", component_property="value"
            ),
            State(
                component_id="reference_inclination_data", component_property="value"
            ),
            State(
                component_id="reference_inclination_const", component_property="value"
            ),
            State(
                component_id="starting_declination_options", component_property="value"
            ),
            State(component_id="starting_declination_data", component_property="value"),
            State(
                component_id="starting_declination_const", component_property="value"
            ),
            State(
                component_id="reference_declination_options", component_property="value"
            ),
            State(
                component_id="reference_declination_data", component_property="value"
            ),
            State(
                component_id="reference_declination_const", component_property="value"
            ),
            prevent_initial_call=True,
        )(self.write_trigger)