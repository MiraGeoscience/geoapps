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

from geoh5py.data import ReferencedData
from geoh5py.objects import ObjectBase
from geoh5py.shared import Entity
from geoh5py.shared.exceptions import AssociationValidationError
from geoh5py.ui_json import InputFile
from ipywidgets import Dropdown, VBox
from peak_finder.application import PeakFinder as DashPeakFinder
from peak_finder.params import PeakFinderParams

from geoapps.base.dash_application import ObjectSelection
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

        self.group_a_data = self.data

        self.line_field = Dropdown(description="Line", options=[["", None]])
        super().__init__(**self.defaults)

        self._objects.observe(self.update_line_list, names="value")
        self.update_line_list(None)

        self.trigger.description = "Launch Dash App"
        self.trigger.on_click(self.trigger_click)

    def update_line_list(self, _):
        refresh = self.refresh.value
        self.refresh.value = False
        if getattr(self, "_workspace", None) is not None:
            obj: ObjectBase | None = self._workspace.get_entity(self.objects.value)[0]
            if obj is None or getattr(obj, "get_data_list", None) is None:
                self.line_field.options = [["", None]]
                self.refresh.value = refresh
                return

            options = self.get_data_list(False, False)
            reference_options = [["", None]] + [
                [name, uid]
                for name, uid in options
                if isinstance(self._workspace.get_entity(uid)[0], ReferencedData)
            ]

            self.line_field.options = reference_options

        else:
            self.line_field.options = []

    @property
    def main(self):
        if self._main is None:
            self._main = VBox(
                [
                    self.project_panel,
                    self.data_panel,
                    self.line_field,
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
