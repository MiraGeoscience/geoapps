# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024-2026 Mira Geoscience Ltd.                                '
#                                                                              '
#  This file is part of geoapps.                                               '
#                                                                              '
#  geoapps is distributed under the terms and conditions of the MIT License    '
#  (see LICENSE file at the root of this source code package).                 '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# pylint: disable=E0401

from __future__ import annotations

import sys
import uuid
import warnings
from pathlib import Path
from time import time

from geoh5py.objects import Curve, ObjectBase, Octree, Points, Surface
from geoh5py.shared import Entity
from geoh5py.shared.utils import fetch_active_workspace
from geoh5py.ui_json import BaseUIJson
from geoh5py.workspace import Workspace
from grid_apps.octree_creation.driver import OctreeDriver
from grid_apps.octree_creation.options import OctreeOptions

from geoapps.base.application import BaseApplication
from geoapps.base.selection import ObjectDataSelection
from geoapps.octree_creation.constants import app_initializer
from geoapps.utils import warn_module_not_found


with warn_module_not_found():
    from ipywidgets import (
        Checkbox,
        Dropdown,
        FloatText,
        IntText,
        Label,
        Layout,
        Text,
        VBox,
        Widget,
    )
    from ipywidgets.widgets.widget_selection import TraitError


class OctreeMesh(ObjectDataSelection):
    """
    Widget used for the creation of an octree mesh
    """

    _param_class = OctreeOptions
    _object_types = (Curve, Octree, Points, Surface)
    _refinement_keywords = ("refinement_object", "levels", "horizon", "distance")

    def __init__(self, ui_json=None, **kwargs):
        self._u_cell_size = None
        self._v_cell_size = None
        self._w_cell_size = None
        self._depth_core = None
        self._horizontal_padding = None
        self._vertical_padding = None
        self._diagonal_balance = None
        self._minimum_level = None

        app_initializer.update(kwargs)
        if ui_json is not None and Path(ui_json).is_file():
            ui_json = BaseUIJson.read(ui_json)
            self.params = self._param_class.build(**ui_json.to_params())
        else:
            ui_json = BaseUIJson.read(self._param_class.default_ui_json)
            ui_json.set_values(**app_initializer)
            self.params = self._param_class.build(**ui_json.to_params())

        for key, value in self.params.model_dump().items():
            if isinstance(value, Entity):
                self.defaults[key] = value.uid
            else:
                self.defaults[key] = value

        self.refinement_list = VBox([])

        super().__init__(**self.defaults)

        self.required = VBox(
            [
                self.project_panel,
                VBox(
                    [
                        Label("Base Parameters"),
                        self.objects,
                        self.depth_core,
                        Label("Core cell size"),
                        self.u_cell_size,
                        self.v_cell_size,
                        self.w_cell_size,
                        Label("Padding distance"),
                        self.horizontal_padding,
                        self.vertical_padding,
                        Label("Basic"),
                        self.diagonal_balance,
                        self.minimum_level,
                    ],
                    layout=Layout(border="solid"),
                ),
            ]
        )

        self.objects.description = "Core hull extent:"
        self.trigger.description = "Create"
        self.ga_group_name.description = "Name:"
        self.ga_group_name.observe(self.update_output_name, names="value")
        self.trigger.on_click(self.trigger_click)

        for obj in self.__dict__:
            if hasattr(getattr(self, obj), "style"):
                getattr(self, obj).style = {"description_width": "initial"}

    def __populate__(self, **kwargs):
        super().__populate__(**kwargs)

        refinement_list = []
        for label in range(len(self.params.refinements)):
            refinement_list += [self.add_refinement_widget(label)]

        self.refinement_list.children = refinement_list

    @property
    def main(self):
        """
        :obj:`ipywidgets.VBox`: A box containing all widgets forming the application.
        """
        if self._main is None:
            self._main = VBox([self.required, self.refinement_list, self.output_panel])

        return self._main

    @property
    def u_cell_size(self) -> FloatText:
        """
        Widget controlling the u-cell size
        """
        if getattr(self, "_u_cell_size", None) is None:
            self._u_cell_size = FloatText(
                description="Easting",
            )
        return self._u_cell_size

    @property
    def v_cell_size(self) -> FloatText:
        """
        Widget controlling the v-cell size
        """
        if getattr(self, "_v_cell_size", None) is None:
            self._v_cell_size = FloatText(
                description="Northing",
            )
        return self._v_cell_size

    @property
    def w_cell_size(self) -> FloatText:
        """
        Widget controlling the w-cell size
        """
        if getattr(self, "_w_cell_size", None) is None:
            self._w_cell_size = FloatText(
                description="Vertical",
            )
        return self._w_cell_size

    @property
    def depth_core(self) -> FloatText:
        """
        Widget controlling the depth core
        """
        if getattr(self, "_depth_core", None) is None:
            self._depth_core = FloatText(
                description="Minimum depth (m)",
            )
        return self._depth_core

    @property
    def horizontal_padding(self) -> FloatText:
        """
        Widget controlling the horizontal padding
        """
        if getattr(self, "_horizontal_padding", None) is None:
            self._horizontal_padding = FloatText(
                description="Horizontal (m)",
            )
        return self._horizontal_padding

    @property
    def vertical_padding(self) -> FloatText:
        """
        Widget controlling the vertical padding
        """
        if getattr(self, "_vertical_padding", None) is None:
            self._vertical_padding = FloatText(
                description="Vertical (m)",
            )
        return self._vertical_padding

    @property
    def diagonal_balance(self) -> Checkbox:
        """
        Widget controlling the diagonal balance.
        """
        if getattr(self, "_diagonal_balance", None) is None:
            self._diagonal_balance = Checkbox(
                description="UBC compatible",
            )
        return self._diagonal_balance

    @property
    def minimum_level(self) -> IntText:
        """
        Widget controlling the minimum refinement level.
        """
        if getattr(self, "_minimum_level", None) is None:
            self._minimum_level = IntText(
                description="Minimum refinement level",
            )
        return self._minimum_level

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
        assert isinstance(workspace, Workspace), (
            f"Workspace must be of class {Workspace}"
        )
        self.base_workspace_changes(workspace)
        self.update_objects_choices()

    def update_objects_choices(self):
        # Refresh the list of objects for all
        self.update_objects_list()

        for widget in self.refinement_list.children:
            widget.children[1].options = self.objects.options

    def update_output_name(self, _):
        self.params.ga_group_name = self.ga_group_name.value

    def trigger_click(self, _):
        param_dict = {}

        out_ui_json = BaseUIJson.read(self.params.default_ui_json)
        for key in self.__dict__:
            try:
                alias = key
                if alias[0] == "_":
                    alias = alias[1:]

                matches = [True for word in self._refinement_keywords if word in alias]
                if any(matches):
                    alias = alias.replace("refinement_", "")
                    alias = "Refinement " + alias

                if isinstance(getattr(self, key), Widget) and hasattr(
                    out_ui_json, alias
                ):
                    value = getattr(self, key).value

                    if (
                        isinstance(value, uuid.UUID)
                        and self.workspace.get_entity(value)[0] is not None
                    ):
                        value = self.workspace.get_entity(value)[0]

                    param_dict[alias] = value

            except AttributeError:
                continue

        temp_geoh5 = f"{self.ga_group_name.value}_{time():.0f}.geoh5"
        ws, self.live_link.value = BaseApplication.get_output_workspace(
            self.live_link.value, self.export_directory.selected_path, temp_geoh5
        )
        with ws as new_workspace:
            param_dict["geoh5"] = new_workspace

            with fetch_active_workspace(self.workspace):
                for key, value in param_dict.items():
                    if isinstance(value, ObjectBase):
                        obj = new_workspace.get_entity(value.uid)[0]
                        if obj is None:
                            obj = value.copy(parent=new_workspace, copy_children=True)
                        param_dict[key] = obj.uid

            if self.live_link.value:
                param_dict["monitoring_directory"] = self.monitoring_directory

            out_ui_json.set_values(**param_dict)
            out_path = Path(self.export_directory.selected_path) / temp_geoh5
            out_ui_json.write(out_path.with_suffix(".ui.json"))
            new_params = OctreeOptions.build(
                out_ui_json.to_params(workspace=new_workspace)
            )

            self.run(new_params)

        if self.live_link.value:
            print("Live link active. Check your ANALYST session for new mesh.")

    @classmethod
    def run(cls, params: OctreeOptions) -> Octree:
        """
        Create an octree mesh from input values
        """

        driver = OctreeDriver(params)
        with params.geoh5.open(mode="r+"):
            octree = driver.run()

        return octree

    def add_refinement_widget(self, ind: int):
        """
        Add a refinement from dictionary
        """
        label = "ABC"[ind]
        widget_list = [Label(label)]
        for key in self._refinement_keywords:
            attr_name = label + f" {key}"
            value = getattr(self.params.refinements[ind], key)
            if "object" in key:
                setattr(
                    self,
                    attr_name,
                    Dropdown(
                        description=key.capitalize(),
                        options=self.objects.options,
                    ),
                )

                try:
                    getattr(self, attr_name).value = value.uid
                except (TraitError, AttributeError):
                    pass

            elif "levels" in key:
                setattr(
                    self,
                    attr_name,
                    Text(
                        description=key.capitalize(), value=", ".join(map(str, value))
                    ),
                )
            elif "horizon" in key:
                setattr(
                    self, attr_name, Checkbox(description=key.capitalize(), value=value)
                )
            elif "distance" in key:
                setattr(
                    self,
                    attr_name,
                    FloatText(description=key.capitalize(), value=value),
                )
            widget_list += [getattr(self, attr_name, None)]

        return VBox(widget_list, layout=Layout(border="solid"))


if __name__ == "__main__":
    file = sys.argv[1]
    warnings.warn(
        "'geoapps.octree_creation.application' replaced by "
        "'geoapps.octree_creation.driver' in version 0.7.0. "
        "This warning is likely due to the execution of older ui.json files. Please update."
    )
    uijson = BaseUIJson.read(file)
    params_class = OctreeOptions(**uijson.to_params())
    OctreeMesh.run(params_class)
