# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024 Mira Geoscience Ltd.                                     '
#                                                                              '
#  This file is part of geoapps.                                               '
#                                                                              '
#  geoapps is distributed under the terms and conditions of the MIT License    '
#  (see LICENSE file at the root of this source code package).                 '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from __future__ import annotations

import warnings
from pathlib import Path
from time import time
from uuid import UUID

from geoh5py.objects import ObjectBase
from geoh5py.shared.utils import fetch_active_workspace
from geoh5py.ui_json import InputFile
from surface_apps.iso_surfaces.driver import IsoSurfacesDriver
from surface_apps.iso_surfaces.params import IsoSurfaceParameters

from geoapps import assets_path
from geoapps.base.application import BaseApplication
from geoapps.base.selection import ObjectDataSelection
from geoapps.utils.formatters import string_name
from geoapps.utils.importing import warn_module_not_found

with warn_module_not_found():
    from ipywidgets import FloatText, HBox, Label, Text, VBox, Widget


class IsoSurface(ObjectDataSelection):
    """
    Application for the conversion of conductivity/depth curves to
    a pseudo 3D conductivity model on surface.
    """

    _param_class = IsoSurfaceParameters
    _add_groups = False
    _select_multiple = False
    _initializer = {
        "geoh5": str(assets_path() / "FlinFlon.geoh5"),
        "objects": UUID("{2e814779-c35f-4da0-ad6a-39a6912361f9}"),
        "data": UUID("{f3e36334-be0a-4210-b13e-06933279de25}"),
        "max_distance": 500.0,
        "resolution": 50.0,
        "interval_min": 0.005,
        "interval_max": 0.02,
        "interval_spacing": 0.005,
        "fixed_contours": "",
        "export_as": "Iso_Iteration_7_model",
        "out_group": None,
    }

    def __init__(self, ui_json=None, geoh5: str | None = None):
        defaults = {}

        if isinstance(geoh5, str):
            if Path(geoh5).exists():
                defaults["geoh5"] = geoh5
            else:
                warnings.warn("Path provided in 'geoh5' argument does not exist.")

        if ui_json is not None:
            if isinstance(ui_json, str):
                if not Path(ui_json).exists():
                    raise FileNotFoundError(
                        f"Provided uijson path {ui_json} not does not exist."
                    )
                defaults = InputFile.read_ui_json(ui_json).data
            elif isinstance(ui_json, InputFile):
                defaults = ui_json.data

        if not defaults:
            if Path(self._initializer["geoh5"]).exists():
                defaults = self._initializer.copy()
            else:
                warnings.warn(
                    "Geoapps is missing 'FlinFlon.geoh5' file in the assets folder."
                )

        self.defaults.update(defaults)

        self._max_distance = FloatText(
            description="Max Interpolation Distance (m):",
        )
        self._resolution = FloatText(
            description="Base grid resolution (m):",
        )
        self._interval_min = FloatText(
            description="Contour min:",
        )
        self._interval_max = FloatText(
            description="Contour max:",
        )
        self._interval_spacing = FloatText(
            description="Contour spacing:",
        )
        self._fixed_contours = Text(
            value=None,
            description="Fixed contours:",
            disabled=False,
            continuous_update=False,
        )
        self._export_as = Text("Iso_", description="Surface:")
        self.ga_group_name.value = self.defaults["out_group"] or "ISO"
        self.data.observe(self.data_change, names="value")
        self.data.description = "Value fields: "
        self.trigger.on_click(self.trigger_click)

        super().__init__(**self.defaults)

        self.contours = VBox(
            [
                self.interval_min,
                self.interval_max,
                self.interval_spacing,
                self.fixed_contours,
            ]
        )
        self.output_panel = VBox([self.export_as, self.output_panel])

    def is_computational(self, attr):
        """True if app attribute is required for the driver (belongs in params)."""
        out = isinstance(getattr(self, attr), Widget)
        return out & (attr.lstrip("_") in self._initializer)

    def trigger_click(self, _):
        param_dict = self.get_param_dict()
        temp_geoh5 = f"{string_name(param_dict.get('export_as'))}_{time():.0f}.geoh5"
        ws, self.live_link.value = BaseApplication.get_output_workspace(
            self.live_link.value, self.export_directory.selected_path, temp_geoh5
        )

        with fetch_active_workspace(ws) as new_workspace:
            with fetch_active_workspace(self.workspace):
                for key, value in param_dict.items():
                    if isinstance(value, ObjectBase):
                        param_dict[key] = value.copy(
                            parent=new_workspace, copy_children=True
                        )

            param_dict["geoh5"] = new_workspace
            param_dict["conda_environment"] = "geoapps"
            param_dict["out_group"] = self.ga_group_name.value

            if self.live_link.value:
                param_dict["monitoring_directory"] = self.monitoring_directory

            new_params = IsoSurfaceParameters.build(param_dict)
            new_params.input_file.write_ui_json(
                name=temp_geoh5.replace(".geoh5", ".ui.json")
            )
            driver = IsoSurfacesDriver(new_params)
            driver.run()

        if self.live_link.value:
            print("Live link active. Check your ANALYST session for new mesh.")

    def data_change(self, _):
        if self.data.value:
            self.export_as.value = "Iso_" + self.data.uid_name_map[self.data.value]

    @property
    def interval_min(self):
        """
        ipywidgets.FloatText(): Minimum value for contours.
        """
        return self._interval_min

    @property
    def interval_max(self):
        """
        ipywidgets.FloatText(): Maximum value for contours.
        """
        return self._interval_max

    @property
    def interval_spacing(self):
        """
        ipywidgets.FloatText(): Step size for contours.
        """
        return self._interval_spacing

    @property
    def fixed_contours(self):
        """
        :obj:`ipywidgets.Text`: String defining sets of fixed_contours.
        """
        return self._fixed_contours

    @property
    def export_as(self):
        """
        ipywidgets.Text()
        """
        return self._export_as

    @property
    def main(self):
        if self._main is None:
            self._main = HBox(
                [
                    VBox(
                        [
                            self.project_panel,
                            self.data_panel,
                            self.contours,
                            self.max_distance,
                            self.resolution,
                            Label("Output"),
                            self.output_panel,
                        ]
                    )
                ]
            )
        return self._main

    @property
    def max_distance(self):
        """
        ipywidgets.FloatText()
        """
        return self._max_distance

    @property
    def resolution(self):
        """
        ipywidgets.FloatText()
        """
        return self._resolution
