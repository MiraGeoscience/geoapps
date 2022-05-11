#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import os
import uuid

from geoh5py.objects import ObjectBase
from geoh5py.shared import Entity
from geoh5py.ui_json import InputFile
from ipywidgets import FloatText, HBox, Label, Text, VBox, Widget

from geoapps.base.selection import ObjectDataSelection
from geoapps.iso_surfaces.driver import IsoSurfacesDriver
from geoapps.iso_surfaces.params import IsoSurfacesParams
from geoapps.iso_surfaces.constants import app_initializer


class IsoSurface(ObjectDataSelection):
    """
    Application for the conversion of conductivity/depth curves to
    a pseudo 3D conductivity model on surface.
    """

    _param_class = IsoSurfacesParams
    _add_groups = False
    _select_multiple = False

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

        self._max_distance = FloatText(
            description="Max Interpolation Distance (m):",
        )
        self._resolution = FloatText(
            description="Base grid resolution (m):",
        )
        self._contours = Text(
            value="", description="Iso-values", disabled=False, continuous_update=False
        )
        self._export_as = Text("Iso_", description="Surface:")

        self.ga_group_name.value = "ISO"
        self.data.observe(self.data_change, names="value")
        self.data.description = "Value fields: "
        self.trigger.on_click(self.trigger_click)

        super().__init__(**self.defaults)

        self.output_panel = VBox([self.export_as, self.output_panel])

    def trigger_click(self, _):

        param_dict = {}
        for key in self.__dict__:
            try:
                if isinstance(getattr(self, key), Widget) and hasattr(self.params, key):
                    value = getattr(self, key).value
                    if key[0] == "_":
                        key = key[1:]

                    if (
                        isinstance(value, uuid.UUID)
                        and self.workspace.get_entity(value)[0] is not None
                    ):
                        value = self.workspace.get_entity(value)[0]

                    param_dict[key] = value

            except AttributeError:
                continue

        new_workspace = self.get_output_workspace(
            self.export_directory.selected_path, self.ga_group_name.value
        )
        for key, value in param_dict.items():
            if isinstance(value, ObjectBase):
                param_dict[key] = value.copy(parent=new_workspace, copy_children=True)

        param_dict["geoh5"] = new_workspace

        if self.live_link.value:
            param_dict["monitoring_directory"] = self.monitoring_directory

        ifile = InputFile(
            ui_json=self.params.input_file.ui_json,
            validation_options={"disabled": True},
        )

        new_params = IsoSurfacesParams(input_file=ifile, **param_dict)
        new_params.write_input_file()

        driver = IsoSurfacesDriver(new_params)
        driver.run()

        if self.live_link.value:
            print("Live link active. Check your ANALYST session for new mesh.")

    def data_change(self, _):

        if self.data.value:
            self.export_as.value = "Iso_" + self.data.uid_name_map[self.data.value]

    @property
    def convert(self):
        """
        ipywidgets.ToggleButton()
        """
        return self._convert

    @property
    def contours(self):
        """
        :obj:`ipywidgets.Text`: String defining sets of contours.
        Contours can be defined over an interval `50:200:10` and/or at a fix value `215`.
        Any combination of the above can be used:
        50:200:10, 215 => Contours between values 50 and 200 every 10, with a contour at 215.
        """
        return self._contours

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
                            self._contours,
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
