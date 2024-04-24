#  Copyright (c) 2024 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


from __future__ import annotations

import pathlib

from simpeg_drivers.potential_fields.gravity.constants import app_initializer
from simpeg_drivers.potential_fields.gravity.params import GravityParams

from geoapps.inversion.base_inversion_application import InversionApp
from geoapps.inversion.potential_fields.gravity.layout import (
    component_list,
    gravity_inversion_params,
    gravity_layout,
)


class GravityApp(InversionApp):
    """
    Application for the inversion of potential field data using SimPEG
    """

    _param_class = GravityParams
    _inversion_type = "gravity"
    _inversion_params = gravity_inversion_params
    _layout = gravity_layout
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

        self.app.callback(
            *self.default_trigger_args,
            prevent_initial_call=True,
        )(self.write_trigger)
