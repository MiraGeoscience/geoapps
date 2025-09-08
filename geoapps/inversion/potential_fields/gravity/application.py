# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024-2025 Mira Geoscience Ltd.                                '
#                                                                              '
#  This file is part of geoapps.                                               '
#                                                                              '
#  geoapps is distributed under the terms and conditions of the MIT License    '
#  (see LICENSE file at the root of this source code package).                 '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


from __future__ import annotations

from simpeg_drivers.potential_fields.gravity.options import (
    GravityForwardOptions,
    GravityInversionOptions,
)

from geoapps.inversion.base_inversion_application import InversionApp
from geoapps.inversion.potential_fields.gravity.constants import app_initializer
from geoapps.inversion.potential_fields.gravity.layout import (
    component_list,
    gravity_inversion_params,
    gravity_layout,
)


class GravityApp(InversionApp):
    """
    Application for the inversion of potential field data using simpeg
    """

    _param_class = GravityInversionOptions
    _param_class_forward = GravityForwardOptions
    _inversion_type = "gravity"
    _inversion_params = gravity_inversion_params
    _layout = gravity_layout
    _app_initializer = app_initializer
    _components = component_list

    def __init__(self, ui_json=None, **kwargs):
        super().__init__(ui_json=ui_json, **kwargs)

        self.app.callback(
            *self.default_trigger_args,
            prevent_initial_call=True,
        )(self.write_trigger)
