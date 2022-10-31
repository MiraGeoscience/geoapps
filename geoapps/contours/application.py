#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import os
from time import time

from geoh5py.objects.object_base import Entity, ObjectBase
from geoh5py.ui_json.input_file import InputFile
from ipywidgets import Checkbox, HBox, Label, Layout, Text, VBox

from geoapps.base.plot import PlotSelection2D
from geoapps.contours.constants import app_initializer
from geoapps.contours.driver import ContoursDriver
from geoapps.contours.params import ContoursParams
from geoapps.utils.formatters import string_name


class ContourValues(PlotSelection2D):
    """
    Application for 2D contouring of spatial data.
    """

    _param_class = ContoursParams

    def __init__(self, ui_json=None, **kwargs):
        app_initializer.update(kwargs)
        if ui_json is not None and os.path.exists(ui_json):
            self.params = self._param_class(InputFile(ui_json))
        else:
            self.params = self._param_class(**app_initializer)

        for key, value in self.params.to_dict().items():
            if isinstance(value, Entity):
                self.defaults[key] = value.uid
            else:
                self.defaults[key] = value

        self.defaults["fixed_contours"] = (
            str(self.defaults["fixed_contours"]).replace("[", "").replace("]", "")
        )
        self._export_as = Text(value="Contours")
        self._z_value = Checkbox(
            value=False, indent=False, description="Assign Z from values"
        )
        self.data.observe(self.update_name, names="value")
        super().__init__(**self.defaults)

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

    def update_contours(self):
        """
        Assign
        """
        if self.data.value is not None:
            self.export_as.value = (
                self.data.uid_name_map[self.data.value]
                + "_"
                + ContoursDriver.get_contour_string(
                    self.interval_min,
                    self.interval_max,
                    self.interval_spacing,
                    self.fixed_contours,
                )
            )

    def update_name(self, _):
        if self.data.value is not None:
            self.export_as.value = self.data.uid_name_map[self.data.value]
        else:
            self.export_as.value = "Contours"

    def trigger_click(self, _):
        param_dict = self.get_param_dict()
        temp_geoh5 = f"{string_name(param_dict['export_as'])}_{time():.0f}.geoh5"
        ws, self.live_link.value = self.get_output_workspace(
            self.live_link.value, self.export_directory.selected_path, temp_geoh5
        )
        with ws as workspace:
            for key, value in param_dict.items():
                if isinstance(value, ObjectBase):
                    param_dict[key] = value.copy(parent=workspace, copy_children=True)

            param_dict["geoh5"] = workspace

            if self.live_link.value:
                param_dict["monitoring_directory"] = self.monitoring_directory

            ifile = InputFile(
                ui_json=self.params.input_file.ui_json,
                validation_options={"disabled": True},
            )
            new_params = ContoursParams(input_file=ifile, **param_dict)
            new_params.write_input_file()
            driver = ContoursDriver(new_params)
            driver.run()

        if self.live_link.value:
            print("Live link active. Check your ANALYST session for new mesh.")
