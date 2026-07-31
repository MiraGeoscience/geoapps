# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024-2026 Mira Geoscience Ltd.                                '
#                                                                              '
#  This file is part of geoapps.                                               '
#                                                                              '
#  geoapps is distributed under the terms and conditions of the MIT License    '
#  (see LICENSE file at the root of this source code package).                 '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from curve_apps.peak_finder.application import PeakFinder as DashPeakFinder
from curve_apps.peak_finder.params import PeakFinderParams

from geoh5py.data import ReferencedData
from geoh5py.objects import ObjectBase
from geoh5py.shared import Entity
from geoh5py.shared.exceptions import AssociationValidationError, OptionalValidationError
from geoh5py.ui_json import InputFile
from ipywidgets import Dropdown, VBox

from geoapps import assets_path
from geoapps.base.selection import ObjectDataSelection


APP_INITIALIZER = {
    "geoh5": str(assets_path() / "FlinFlon_airborne_tem.geoh5"),
    "objects": UUID("{34698019-cde6-4b43-8d53-a040b25c989a}"),
    "group_a_data": UUID("{22a9cf91-5cff-42b5-8bbb-2f1c6a559204}"),
}


class PeakFinder(ObjectDataSelection):
    """
    Application for the picking of targets along Time-domain EM profiles
    """

    _param_class = PeakFinderParams
    _add_groups = "only"

    def __init__(self, ui_json=None, **kwargs):
        APP_INITIALIZER.update(kwargs)
        if ui_json is not None and Path(ui_json).is_file():
            self.params = self._param_class(input_file=InputFile(ui_json=ui_json))
        else:
            try:
                self.params = self._param_class(**APP_INITIALIZER)

            except (AssociationValidationError, OptionalValidationError):
                for key, value in APP_INITIALIZER.items():
                    if isinstance(value, UUID):
                        APP_INITIALIZER[key] = None

                self.params = self._param_class(**APP_INITIALIZER)

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
        ObjectSelection.run("Peak Finder", DashPeakFinder, new_params)
