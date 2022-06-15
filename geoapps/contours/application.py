#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import os
from time import time

from geoh5py.objects.object_base import Entity, ObjectBase
from geoh5py.ui_json.input_file import InputFile
from geoh5py.ui_json.utils import monitored_directory_copy
from ipywidgets import Checkbox, HBox, Label, Layout, Text, VBox, interactive_output

from geoapps import PlotSelection2D
from geoapps.contours.constants import app_initializer
from geoapps.contours.driver import ContoursDriver
from geoapps.utils.formatters import string_name


class ContourValues(PlotSelection2D):
    """
    Application for 2D contouring of spatial data.
    """

    def __init__(self, ui_json=None, **kwargs):
        app_initializer.update(kwargs)
        if ui_json is not None and os.path.exists(ui_json):
            self.params = self._param_class(InputFile(ui_json))
        else:
            self.params = self._param_class(**app_initializer)

        self.defaults = {}
        for key, value in self.params.to_dict().items():
            if isinstance(value, Entity):
                self.defaults[key] = value.uid
            else:
                self.defaults[key] = value

        self._export_as = Text(value="Contours")
        self._z_value = Checkbox(
            value=False, indent=False, description="Assign Z from values"
        )
        self.data.observe(self.update_name, names="value")
        super().__init__(**self.defaults)

        self.selection = interactive_output(
            self.compute_plot,
            {
                "interval_min": self.interval_min,
                "interval_max": self.interval_max,
                "interval_spacing": self.interval_spacing,
                "fixed_contours": self.fixed_contours,
            },
        )

        self.trigger.on_click(self.trigger_click)
        self.trigger.description = "Export"
        self.trigger.button_style = "danger"

    @property
    def export(self):
        """
        :obj:`ipywidgets.ToggleButton`: Write contours to the target geoh5
        """
        return self._export

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
                                    self.interval_min,
                                    self.interval_max,
                                    self.interval_spacing,
                                    self.fixed_contours,
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
                    self.selection,
                ]
            )
        return self._main

    def compute_plot(
        self, interval_min, interval_max, interval_spacing, fixed_contours
    ):
        """
        Get current selection and trigger update
        """
        entity, data = self.get_selected_entities()
        if data is None:
            return

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
        temp_geoh5 = f"{string_name(self.params.export_as)}_{time():.3f}.geoh5"
        with self.get_output_workspace(
            self.export_directory.selected_path, temp_geoh5
        ) as workspace:
            for key, value in param_dict.items():
                if isinstance(value, ObjectBase):
                    param_dict[key] = value.copy(parent=workspace, copy_children=True)

            param_dict["geoh5"] = workspace

            if self.live_link.value:
                param_dict["monitoring_directory"] = self.monitoring_directory

            self.params.update(param_dict)
            self.params.write_input_file()

            driver = ContoursDriver(self.params)
            driver.run()

        if self.live_link.value:
            print("Live link active. Check your ANALYST session for new mesh.")
