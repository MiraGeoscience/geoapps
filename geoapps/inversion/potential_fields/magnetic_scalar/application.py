# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024-2026 Mira Geoscience Ltd.                                '
#                                                                              '
#  This file is part of geoapps.                                               '
#                                                                              '
#  geoapps is distributed under the terms and conditions of the MIT License    '
#  (see LICENSE file at the root of this source code package).                 '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from __future__ import annotations

from dash import Input, Output, State
from simpeg_drivers.potential_fields.magnetic_scalar.options import (
    MagneticForwardOptions,
    MagneticInversionOptions,
)

from geoapps.inversion.base_inversion_application import InversionApp
from geoapps.inversion.potential_fields.magnetic_scalar.constants import app_initializer
from geoapps.inversion.potential_fields.magnetic_scalar.layout import (
    component_list,
    magnetic_scalar_inversion_params,
    magnetic_scalar_layout,
)


class MagneticScalarApp(InversionApp):
    """
    Application for the inversion of potential field data using simpeg
    """

    _app_initializer = app_initializer
    _param_class = MagneticInversionOptions
    _param_class_forward = MagneticForwardOptions
    _inversion_type = "magnetic scalar"
    _inversion_params = magnetic_scalar_inversion_params
    _layout = magnetic_scalar_layout
    _components = component_list

    def __init__(self, ui_json=None, **kwargs):
        super().__init__(ui_json=ui_json, **kwargs)

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

        trigger_args = [
            State(component_id="inducing_field_strength", component_property="value"),
            State(
                component_id="inducing_field_inclination", component_property="value"
            ),
            State(
                component_id="inducing_field_declination", component_property="value"
            ),
        ]

        self.app.callback(
            *(self.default_trigger_args + trigger_args),
            prevent_initial_call=True,
        )(self.write_trigger)
