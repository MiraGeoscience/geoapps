#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import sys
import uuid
from copy import deepcopy
from os import path

from discretize.utils import mesh_builder_xyz, refine_tree_xyz
from geoh5py.objects import Curve, Octree, Points, Surface
from geoh5py.workspace import Workspace
from ipywidgets import Dropdown, FloatText, Label, Layout, Text, VBox, Widget
from ipywidgets.widgets.widget_selection import TraitError

from geoapps.base import BaseApplication
from geoapps.io import InputFile
from geoapps.io.Octree.constants import app_initializer, default_ui_json
from geoapps.io.Octree.params import OctreeParams
from geoapps.selection import ObjectDataSelection
from geoapps.utils.utils import string_2_list, treemesh_2_octree


class OctreeMesh(ObjectDataSelection):
    """
    Widget used for the creation of an octree mesh
    """

    defaults = {}
    _param_class = OctreeParams
    _object_types = (Curve, Octree, Points, Surface)
    _u_cell_size = None
    _v_cell_size = None
    _w_cell_size = None
    _depth_core = None
    _horizontal_padding = None
    _vertical_padding = None

    def __init__(self, ui_json=None, **kwargs):
        app_initializer.update(kwargs)
        if ui_json is not None and path.exists(ui_json):
            self.params = self._param_class(InputFile(ui_json))
        else:
            if "h5file" in app_initializer.keys():
                app_initializer["geoh5"] = app_initializer.pop("h5file")
                app_initializer["workspace"] = app_initializer["geoh5"]

            self.params = self._param_class(**app_initializer)

        self.defaults.update(self.params.to_dict(ui_json_format=False))
        self.defaults.pop("workspace", None)
        self.refinement_list = VBox([])

        super().__init__()

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
        for label, params in self.params._free_param_dict.items():
            refinement_list += [self.add_refinement_widget(label, params)]

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
        self.update_objects_choices()

    def update_objects_choices(self):
        # Refresh the list of objects for all
        self.update_objects_list()

        for widget in self.refinement_list.children:
            widget.children[1].options = self.objects.options

    def update_output_name(self, _):
        self.params.ga_group_name = self.ga_group_name.value

    def trigger_click(self, _):
        for key, value in self.__dict__.items():
            try:
                if isinstance(getattr(self, key), Widget):
                    setattr(self.params, key, getattr(self, key).value)
            except AttributeError:
                continue

        self.params._free_param_dict = {}
        ui_json = deepcopy(default_ui_json)
        for group, refinement in zip("ABCDFEGH", self.refinement_list.children):
            self.params._free_param_dict[refinement.children[0].value] = {
                "object": refinement.children[1].value,
                "levels": string_2_list(refinement.children[2].value),
                "type": refinement.children[3].value,
                "distance": refinement.children[4].value,
            }

        self.params.write_input_file(ui_json=ui_json, name=self.params.ga_group_name)
        self.run(self.params)

    @staticmethod
    def run(params: OctreeParams) -> Octree:
        """
        Create an octree mesh from input values
        """

        obj = params.workspace.get_entity(params.objects)

        if not any(obj):
            return

        p_d = [
            [
                params.horizontal_padding,
                params.horizontal_padding,
            ],
            [
                params.horizontal_padding,
                params.horizontal_padding,
            ],
            [params.vertical_padding, params.vertical_padding],
        ]

        print("Setting the mesh extent")
        treemesh = mesh_builder_xyz(
            obj[0].vertices,
            [
                params.u_cell_size,
                params.v_cell_size,
                params.w_cell_size,
            ],
            padding_distance=p_d,
            mesh_type="tree",
            depth_core=params.depth_core,
        )

        for label, value in params._free_param_dict.items():

            try:
                uid = (
                    uuid.UUID(value["object"])
                    if isinstance(value["object"], str)
                    else value["object"]
                )
                entity = params.workspace.get_entity(uid)

            except (ValueError, TypeError):
                continue

            if any(entity):
                print(f"Applying {label} on: {entity[0].name}")
                treemesh = refine_tree_xyz(
                    treemesh,
                    entity[0].vertices,
                    method=value["type"],
                    octree_levels=value["levels"],
                    max_distance=value["distance"],
                    finalize=False,
                )

        print("Finalizing...")
        treemesh.finalize()

        print("Writing to file ")
        octree = treemesh_2_octree(
            params.workspace, treemesh, name=params.ga_group_name
        )

        if params.monitoring_directory is not None and path.exists(
            params.monitoring_directory
        ):
            BaseApplication.live_link_output(params.monitoring_directory, octree)

        print(
            f"Octree mesh '{octree.name}' completed and exported to {path.abspath(params.workspace.h5file)}"
        )

        assert octree.workspace is not None
        return octree

    def add_refinement_widget(self, label: str, params: dict):
        """
        Add a refinement from dictionary
        """
        widget_list = [Label(label.title())]
        for key, value in params.items():
            attr_name = (label + f" {key}").title()

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
                    getattr(self, attr_name).value = value
                except TraitError:
                    pass

            elif "levels" in key:
                setattr(
                    self,
                    attr_name,
                    Text(
                        description=key.capitalize(), value=", ".join(map(str, value))
                    ),
                )
            elif "type" in key:
                setattr(
                    self,
                    attr_name,
                    Dropdown(
                        description=key.capitalize(),
                        options=["surface", "radial"],
                        value=value,
                    ),
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
    params = OctreeParams(InputFile(sys.argv[1]))
    OctreeMesh.run(params)
