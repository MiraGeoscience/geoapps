# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024-2025 Mira Geoscience Ltd.                                '
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

from curve_apps.contours.driver import ContoursDriver
from curve_apps.contours.params import ContourParameters
from geoh5py.objects import Grid2D
from geoh5py.objects.object_base import ObjectBase
from geoh5py.shared.utils import fetch_active_workspace
from geoh5py.ui_json.input_file import InputFile
from ipywidgets import Checkbox, HBox, Label, Layout, Text, VBox, Widget

from geoapps import assets_path
from geoapps.base.plot import PlotSelection2D
from geoapps.inversion.components.preprocessing import grid_to_points
from geoapps.shared_utils.utils import filter_xy
from geoapps.utils.formatters import string_name


class ContourValues(PlotSelection2D):
    """
    Application for 2D contouring of spatial data.
    """

    _param_class = ContourParameters
    _initializer = {
        "geoh5": str(assets_path() / "FlinFlon.geoh5"),
        "objects": UUID("{538a7eb1-2218-4bec-98cc-0a759aa0ef4f}"),
        "data": UUID("{44822654-b6ae-45b0-8886-2d845f80f422}"),
        "interval_min": -400.0,
        "interval_max": 2000.0,
        "interval_spacing": 100.0,
        "fixed_contours": "-240",
        "resolution": 50.0,
        "z_value": False,
        "export_as": "Contours",
        "out_group": None,
    }

    def __init__(
        self,
        ui_json: InputFile | str | None = None,
        plot_result=True,
        geoh5: str | None = None,
    ):
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
        self.defaults["fixed_contours"] = (
            str(self.defaults["fixed_contours"]).replace("[", "").replace("]", "")
        )

        self._export_as = Text(value="Contours")
        self._z_value = Checkbox(
            value=False, indent=False, description="Assign Z from values"
        )
        self.data.observe(self.update_name, names="value")
        super().__init__(plot_result=plot_result, **self.defaults)

        self.contours = VBox(
            [
                self.interval_min,
                self.interval_max,
                self.interval_spacing,
                self.fixed_contours,
            ]
        )

        self.trigger.on_click(self.trigger_click)
        self.trigger.description = "Export"
        self.trigger.button_style = "danger"

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, val):
        if not isinstance(val, ContourParameters):
            raise TypeError("Input parameters must be of type ContourParameters.")
        self._params = val

    @property
    def export_as(self):
        """
        :obj:`ipywidgets.Text`: Name given to the Curve object
        """
        return self._export_as

    @property
    def z_value(self):
        """
        :obj:`ipywidgets.Checkbox`: Assign z-coordinate based on contour values
        """
        return self._z_value

    @property
    def main(self):
        """
        :obj:`ipywidgets.VBox`: A box containing all widgets forming the application.
        """
        if self._main is None:
            self._main = VBox(
                [
                    self.project_panel,
                    HBox(
                        [
                            VBox(
                                [
                                    Label("Input options:"),
                                    self.data_panel,
                                    self.contours,
                                    self.window_selection,
                                ]
                            ),
                            VBox(
                                [
                                    Label("Save as:"),
                                    self.export_as,
                                    self.z_value,
                                    self.output_panel,
                                ],
                                layout=Layout(width="50%"),
                            ),
                        ]
                    ),
                ]
            )
        return self._main

    def update_name(self, _):
        if self.data.value is not None:
            self.export_as.value = self.data.uid_name_map[self.data.value]
        else:
            self.export_as.value = "Contours"

    def is_computational(self, attr):
        """True if app attribute is required for the driver (belongs in params)."""
        out = isinstance(getattr(self, attr), Widget)
        return out & (attr.lstrip("_") in self._initializer)

    def trigger_click(self, _):
        param_dict = self.get_param_dict()
        temp_geoh5 = f"{string_name(param_dict['export_as'])}_{time():.0f}.geoh5"
        ws, self.live_link.value = self.get_output_workspace(
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

                if self.live_link.value:
                    param_dict["monitoring_directory"] = self.monitoring_directory

                if isinstance(param_dict["objects"], Grid2D):
                    param_dict["objects"] = grid_to_points(param_dict["objects"])

                x = param_dict["objects"].locations[:, 0]
                y = param_dict["objects"].locations[:, 1]
                window = {
                    "center": [self.window_center_x.value, self.window_center_y.value],
                    "size": [self.window_width.value, self.window_height.value],
                }
                indices = filter_xy(
                    x,
                    y,
                    self.resolution.value,
                    window=window,
                    angle=self.window_azimuth.value,
                )
                param_dict["objects"] = param_dict["objects"].copy(mask=indices)
                param_dict["data"] = param_dict["objects"].get_data(
                    param_dict["data"].name
                )[0]

                new_params = ContourParameters.build(param_dict)
                new_params.input_file.write_ui_json(
                    name=temp_geoh5.replace(".geoh5", ".ui.json")
                )
                driver = ContoursDriver(new_params)
                driver.run()

        if self.live_link.value:
            print("Live link active. Check your ANALYST session for new mesh.")
