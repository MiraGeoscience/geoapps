# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024 Mira Geoscience Ltd.                                     '
#                                                                              '
#  This file is part of geoapps.                                               '
#                                                                              '
#  geoapps is distributed under the terms and conditions of the MIT License    '
#  (see LICENSE file at the root of this source code package).                 '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from __future__ import annotations

import pathlib

from dash import Input, Output, State

from geoapps.inversion.base_inversion_application import InversionApp
from geoapps.inversion.potential_fields.magnetic_vector.constants import app_initializer
from geoapps.inversion.potential_fields.magnetic_vector.layout import (
    component_list,
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
    _components = component_list

    def __init__(self, ui_json=None, **kwargs):
        if ui_json is not None and pathlib.Path(ui_json.path).exists():
            self.params = self._param_class(ui_json)
        else:
            app_initializer.update(kwargs)
            self.params = self._param_class(**app_initializer)
            extras = {
                key: value
                for key, value in app_initializer.items()
                if key not in self.params.param_names
            }
            self._app_initializer = extras

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

        trigger_args = [
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
        ]

        self.app.callback(
            *(self.default_trigger_args + trigger_args),
            prevent_initial_call=True,
        )(self.write_trigger)
