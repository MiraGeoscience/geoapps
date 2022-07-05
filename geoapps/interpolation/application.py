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
    from ipywidgets import Dropdown, FloatText, HBox, Label, RadioButtons, Text, VBox


class DataInterpolation(ObjectDataSelection):
    """
    Transfer data from one object to another, or onto a 3D BlockModel
    """

    _param_class = DataInterpolationParams
    _select_multiple = True
    # _topography = None

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
        self.defaults["topography"] = self.params.topography

        super().__init__(**self.defaults)

        self._core_cell_size = Text(
            description="Smallest cells",
        )
        self._depth_core = FloatText(
            description="Core depth (m)",
        )
        self._expansion_fact = FloatText(
            description="Expansion factor",
        )
        self._max_distance = FloatText(
            description="Maximum distance (m)",
        )
        self._max_depth = FloatText(
            description="Maximum depth (m)",
        )
        self._method = RadioButtons(
            options=["Nearest", "Inverse Distance"],
        )
        self._new_grid = Text(
            description="Name",
        )
        self._no_data_value = FloatText()
        self._out_mode = RadioButtons(
            options=["To Object", "Create 3D Grid"],
        )
        self._out_object = Dropdown()
        self._padding_distance = Text(
            description="Pad Distance (W, E, S, N, D, U)",
        )
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
        self._xy_reference = Dropdown(
            description="Lateral Extent", style={"description_width": "initial"}
        )
        self.objects.observe(self.object_pick, names="value")
        self.method_skew = VBox(
            [Label("Skew parameters"), self.skew_angle, self.skew_factor]
        )
        self.method_panel = VBox([self.method])
        self.destination_panel = VBox([self.out_mode, self.out_object])
        self.new_grid_panel = VBox(
            [
                self.new_grid,
                self.xy_reference,
                self.core_cell_size,
                self.depth_core,
                self.padding_distance,
                self.expansion_fact,
            ]
        )
        self.method.observe(self.method_update)
        self.out_mode.observe(self.out_update)
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

        self.parameters["Vertical Extent"].children = [
            self.topography.main,
            self.max_depth,
        ]
        self.trigger.on_click(self.trigger_click)
        self.trigger.description = "Interpolate"

    @property
    def core_cell_size(self):
        """
        :obj:`ipywidgets.Text()`
        """
        return self._core_cell_size

    @property
    def depth_core(self):
        """
        :obj:`ipywidgets.FloatText()`
        """
        return self._depth_core

    @property
    def expansion_fact(self):
        """
        :obj:`ipywidgets.FloatText()`
        """
        return self._expansion_fact

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
    def new_grid(self):
        """
        :obj:`ipywidgets.Text()`
        """
        return self._new_grid

    @property
    def no_data_value(self):
        """
        :obj:`ipywidgets.FloatText()`
        """
        return self._no_data_value

    @property
    def out_mode(self):
        """
        :obj:`ipywidgets.RadioButtons()`
        """
        return self._out_mode

    @property
    def out_object(self):
        """
        :obj:`ipywidgets.Dropdown()`
        """
        return self._out_object

    @property
    def padding_distance(self):
        """
        :obj:`ipywidgets.Text()`
        """
        return self._padding_distance

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

        return self._topography

    @property
    def xy_extent(self):
        """
        :obj:`ipywidgets.Dropdown()`
        """
        return self._xy_extent

    @property
    def xy_reference(self):
        """
        :obj:`ipywidgets.Dropdown()`
        """
        return self._xy_reference

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
        elif self.method.value == "Linear":
            self.method_panel.children = [
                self.method,
                Label("Warning! Very slow on 3D objects"),
            ]
        else:
            self.method_panel.children = [self.method]

    def out_update(self, _):
        if self.out_mode.value == "To Object":
            self.destination_panel.children = [self.out_mode, self.out_object]
        else:
            self.destination_panel.children = [self.out_mode, self.new_grid_panel]

    def trigger_click(self, _):
        param_dict = self.get_param_dict()
        for key in ["options", "objects", "data", "constant"]:
            param_dict["topography_" + key] = getattr(self.params, "topography_" + key)

        temp_geoh5 = f"Interpolation_{time():.3f}.geoh5"
        with self.get_output_workspace(
            self.export_directory.selected_path, temp_geoh5
        ) as workspace:
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
            new_params = DataInterpolationParams(input_file=ifile, **param_dict)
            driver = DataInterpolationDriver(new_params)
            driver.run()

        if self.live_link.value:
            print("Live link active. Check your ANALYST session for new mesh.")

    def object_pick(self, _):
        if self.objects.value in list(dict(self.xy_reference.options).values()):
            self.xy_reference.value = self.objects.value
