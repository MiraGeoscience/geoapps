#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import os
from time import time

from geoh5py.objects import ObjectBase
from geoh5py.objects.object_base import Entity
from geoh5py.ui_json.input_file import InputFile
from geoh5py.workspace import Workspace

from geoapps.base.selection import ObjectDataSelection, TopographyOptions
from geoapps.interpolation.constants import app_initializer
from geoapps.interpolation.driver import DataInterpolationDriver
from geoapps.interpolation.params import DataInterpolationParams
from geoapps.utils import warn_module_not_found

with warn_module_not_found():
    from ipywidgets import Dropdown, FloatText, HBox, Label, RadioButtons, VBox


class DataInterpolation(ObjectDataSelection):
    """
    Transfer data from one object to another, or onto a 3D BlockModel
    """

    _param_class = DataInterpolationParams
    _topography = None

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
        self.defaults["topography"] = self.params.topography

        super().__init__(**self.defaults)

        self._max_distance = FloatText(
            description="Maximum distance (m)",
        )
        self._max_depth = FloatText(
            description="Maximum depth (m)",
        )
        self._method = RadioButtons(
            options=["Nearest", "Inverse Distance"],
        )
        self._no_data_value = FloatText()
        self._out_object = Dropdown(description="Object: ")
        self._skew_angle = FloatText(
            description="Azimuth (d.dd)",
        )
        self._skew_factor = FloatText(
            description="Factor (>0)",
        )
        self._space = RadioButtons(options=["Linear", "Log"])
        self._xy_extent = Dropdown(
            description="Object hull",
        )
        self.method_skew = VBox(
            [Label("Skew parameters"), self.skew_angle, self.skew_factor]
        )
        self.method_panel = VBox([self.method])
        self.destination_panel = VBox([self.out_object])
        self.method.observe(self.method_update)
        self.parameters = {
            "Method": self.method_panel,
            "Scaling": self.space,
            "Horizontal Extent": VBox(
                [
                    self.max_distance,
                    self.xy_extent,
                ]
            ),
            "Vertical Extent": VBox([]),
            "No-data-value": self.no_data_value,
        }

        self.parameter_choices = Dropdown(
            description="Interpolation Parameters",
            options=list(self.parameters.keys()),
            style={"description_width": "initial"},
        )

        self.parameter_panel = HBox([self.parameter_choices, self.method_panel])
        self.ga_group_name.description = "Output Label:"
        self.ga_group_name.value = "_Interp"
        self.parameter_choices.observe(self.parameter_change)

        super().__init__(**self.defaults)

        self.xy_extent.options = self.objects.options
        self.parameters["Vertical Extent"].children = [
            self.topography.data_panel,
            self.max_depth,
        ]

        self.trigger.on_click(self.trigger_click)
        self.trigger.description = "Interpolate"

    @property
    def main(self):
        if self._main is None:
            self._main = VBox(
                [
                    self.project_panel,
                    HBox(
                        [
                            VBox([Label("Source"), self.data_panel]),
                            VBox([Label("Destination"), self.destination_panel]),
                        ]
                    ),
                    self.parameter_panel,
                    self.output_panel,
                ]
            )
        return self._main

    @property
    def max_distance(self):
        """
        :obj:`ipywidgets.FloatText()`
        """
        return self._max_distance

    @property
    def max_depth(self):
        """
        :obj:`ipywidgets.FloatText()`
        """
        return self._max_depth

    @property
    def method(self):
        """
        :obj:`ipywidgets.RadioButtons()`
        """
        return self._method

    @property
    def no_data_value(self):
        """
        :obj:`ipywidgets.FloatText()`
        """
        return self._no_data_value

    @property
    def out_object(self):
        """
        :obj:`ipywidgets.Dropdown()`
        """
        return self._out_object

    @property
    def skew_angle(self):
        """
        :obj:`ipywidgets.FloatText()`
        """
        return self._skew_angle

    @property
    def skew_factor(self):
        """
        :obj:`ipywidgets.FloatText()`
        """
        return self._skew_factor

    @property
    def space(self):
        """
        :obj:`ipywidgets.RadioButtons()`
        """
        return self._space

    @property
    def topography(self):
        """
        :obj:`geoapps.TopographyOptions()`
        """
        if getattr(self, "_topography", None) is None:
            self._topography = TopographyOptions(
                option_list=["None", "Object", "Constant"],
                workspace=self.workspace,
                add_xyz=False,
                **self.defaults["topography"],
            )
            self._topography.options.value = "Object"
        return self._topography

    @property
    def xy_extent(self):
        """
        :obj:`ipywidgets.Dropdown()`
        """
        return self._xy_extent

    @property
    def workspace(self):
        """
        Target geoh5py workspace
        """
        if (
            getattr(self, "_workspace", None) is None
            and getattr(self, "_h5file", None) is not None
        ):
            self.workspace = Workspace(self.h5file)
        return self._workspace

    @workspace.setter
    def workspace(self, workspace):
        assert isinstance(workspace, Workspace), f"Workspace must of class {Workspace}"
        self.base_workspace_changes(workspace)
        self.update_objects_list()
        self.out_object.options = self.objects.options
        self.topography.workspace = workspace

    def parameter_change(self, _):
        self.parameter_panel.children = [
            self.parameter_choices,
            self.parameters[self.parameter_choices.value],
        ]

    def method_update(self, _):
        if self.method.value == "Inverse Distance":
            self.method_panel.children = [self.method, self.method_skew]
        else:
            self.method_panel.children = [self.method]

    def trigger_click(self, _):
        param_dict = self.get_param_dict()

        param_dict["topography_objects"] = self.params.geoh5.get_entity(
            self._topography.objects.value
        )[0]
        param_dict["topography_data"] = self.params.geoh5.get_entity(
            self._topography.data.value
        )[0]

        temp_geoh5 = f"Interpolation_{time():.0f}.geoh5"

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

            new_params = DataInterpolationParams(**param_dict)
            new_params.method = self.method.value

            if new_params.method == "Nearest":
                new_params.skew_angle = None
                new_params.skew_factor = None
                new_params.input_file.ui_json["skew_angle"]["enabled"] = False
                new_params.input_file.ui_json["skew_factor"]["enabled"] = False
            elif new_params.method == "Inverse Distance":
                new_params.input_file.ui_json["skew_angle"]["enabled"] = True
                new_params.input_file.ui_json["skew_factor"]["enabled"] = True

            new_params.write_input_file(
                name=temp_geoh5.replace(".geoh5", ".ui.json"), validate=False
            )

            driver = DataInterpolationDriver(new_params)
            print("Running data transfer . . .")
            driver.run()

        if self.live_link.value:
            print("Live link active. Check your ANALYST session for new mesh.")
        else:
            print("Saved to " + new_params.geoh5.h5file)
