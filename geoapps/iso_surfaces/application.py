#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import os
import uuid
from time import time

from geoh5py.shared import Entity
from geoh5py.ui_json import InputFile

from geoapps.base.application import BaseApplication
from geoapps.base.selection import ObjectDataSelection
from geoapps.iso_surfaces.constants import app_initializer
from geoapps.iso_surfaces.driver import IsoSurfacesDriver
from geoapps.iso_surfaces.params import IsoSurfacesParams
from geoapps.utils.importing import warn_module_not_found

with warn_module_not_found():
    from ipywidgets import FloatText, HBox, Label, Text, VBox, Widget


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
            value="",
            description="Fixed contours:",
            disabled=False,
            continuous_update=False,
        )
        self._export_as = Text("Iso_", description="Surface:")

        self.ga_group_name.value = "ISO"
        self.data.observe(self.data_change, names="value")
        self.data.description = "Value fields: "
        self.trigger.on_click(self.trigger_click)

        self.defaults["fixed_contours"] = str(self.defaults["fixed_contours"])[1:-1]
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

    def trigger_click(self, _) -> str:

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

        temp_geoh5 = f"Isosurface_{time():.0f}.geoh5"
        ws, self.live_link.value = BaseApplication.get_output_workspace(
            self.live_link.value, self.export_directory.selected_path, temp_geoh5
        )
        with ws as new_workspace:
            with self.workspace.open(mode="r"):
                param_dict["objects"] = param_dict["objects"].copy(
                    parent=new_workspace, copy_children=False
                )
                param_dict["data"] = param_dict["data"].copy(
                    parent=param_dict["objects"]
                )

            param_dict["geoh5"] = new_workspace

            if self.live_link.value:
                param_dict["monitoring_directory"] = self.monitoring_directory

            new_params = IsoSurfacesParams(**param_dict)
            new_params.write_input_file(name=temp_geoh5.replace(".geoh5", ".ui.json"))
            driver = IsoSurfacesDriver(new_params)
            driver.run()

        if self.live_link.value:
            print("Live link active. Check your ANALYST session for new mesh.")

        return new_workspace.h5file

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
