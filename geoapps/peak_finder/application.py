# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024 Mira Geoscience Ltd.                                     '
#                                                                              '
#  This file is part of geoapps.                                               '
#                                                                              '
#  geoapps is distributed under the terms and conditions of the MIT License    '
#  (see LICENSE file at the root of this source code package).                 '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from __future__ import annotations

import uuid
from pathlib import Path

from geoapps_utils.application.dash_application import ObjectSelection
from geoh5py.shared import Entity
from geoh5py.shared.exceptions import AssociationValidationError
from geoh5py.ui_json import InputFile
from ipywidgets import VBox
from peak_finder.application import PeakFinder as DashPeakFinder
from peak_finder.params import PeakFinderParams

from geoapps.base.selection import ObjectDataSelection
from geoapps.peak_finder.constants import app_initializer


class PeakFinder(ObjectDataSelection):
    """
    Application for the picking of targets along Time-domain EM profiles
    """

    _param_class = PeakFinderParams
    _add_groups = "only"

    def __init__(self, ui_json=None, **kwargs):

        app_initializer.update(kwargs)
        if ui_json is not None and Path(ui_json).is_file():
            self.params = self._param_class(InputFile(ui_json))
        else:
            try:
                self.params = self._param_class(**app_initializer)

            except AssociationValidationError:
                for key, value in app_initializer.items():
                    if isinstance(value, uuid.UUID):
                        app_initializer[key] = None

                self.params = self._param_class(**app_initializer)

        for key, value in self.params.to_dict().items():
            if isinstance(value, Entity):
                self.defaults[key] = value.uid
            else:
                self.defaults[key] = value

        super().__init__(**self.defaults)
        self.trigger.description = "Launch Application"
        self.trigger.on_click(self.trigger_click)

    @property
    def main(self):
        if self._main is None:
            self._main = VBox(
                [
                    self.project_panel,
                    self.data_panel,
                    self.output_panel,
                ]
            )
        return self._main

    def trigger_click(self, _):
        """
        Trigger the application
        """
        new_params = self.collect_parameter_values()
        ObjectSelection.run("Peak Finder", DashPeakFinder, new_params.input_file)
