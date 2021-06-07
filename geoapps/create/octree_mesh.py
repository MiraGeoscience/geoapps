#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import sys
import uuid
from os import path

from discretize.utils import mesh_builder_xyz, refine_tree_xyz
from geoh5py.objects import Curve, Octree, Points, Surface
from geoh5py.workspace import Workspace
from ipywidgets import Dropdown, FloatText, Label, Layout, Text, VBox, Widget
from ipywidgets.widgets.widget_selection import TraitError

from geoapps.base import BaseApplication
from geoapps.io.Octree.params import OctreeParams
from geoapps.selection import ObjectDataSelection
from geoapps.utils.utils import string_2_list, treemesh_2_octree


class OctreeMesh(ObjectDataSelection):
    """
    Widget used for the creation of an octree meshes
    """

    _object_types = (Curve, Octree, Points, Surface)

    def __init__(self, ui_json=None):

        if ui_json is not None and path.exists(ui_json):
            self.params = OctreeParams.from_path(ui_json)
        else:
            self.params = OctreeParams()
            self.params.init_from_dict(self.params.default_ui_json)

        self.defaults = self.update_defaults(**self.params.__dict__)

        self._u_cell_size = FloatText(
            description="Easting",
        )
        self._v_cell_size = FloatText(
            description="Northing",
        )
        self._w_cell_size = FloatText(
            description="Vertical",
        )
        self._depth_core = FloatText(
            description="Minimum depth (m)",
        )
        self._horizontal_padding = FloatText(
            description="Horizontal (m)",
        )
        self._vertical_padding = FloatText(
            description="Vertical (m)",
        )
        self._refinements = self.params.refinements
        self.refinement_list = []

        super().__init__(**self.defaults)

        self.required = [
            self.project_panel,
            VBox(
                [
                    Label("Base Parameters"),
                    self.objects,
                    self._depth_core,
                    Label("Core cell size"),
                    self._u_cell_size,
                    self._v_cell_size,
                    self._w_cell_size,
                    Label("Padding distance"),
                    self._horizontal_padding,
                    self._vertical_padding,
                ],
                layout=Layout(border="solid"),
            ),
        ]

        self.objects.description = "Core hull extent:"
        self.trigger.description = "Create"
        self.ga_group_name.description = "Name:"
        self.ga_group_name.observe(self.update_output_name, names="value")
        self.trigger.on_click(self.trigger_click)

        for obj in self.__dict__:
            if hasattr(getattr(self, obj), "style"):
                getattr(self, obj).style = {"description_width": "initial"}

    def __populate__(self, **kwargs):

        for key, value in kwargs.items():
            if key[0] == "_":
                key = key[1:]

            if hasattr(self, "_" + key) or hasattr(self, key):

                if isinstance(value, dict) and "value" in list(value.keys()):
                    value = value["value"]

                try:
                    if isinstance(getattr(self, key, None), Widget) and not isinstance(
                        value, Widget
                    ):
                        try:
                            value = uuid.UUID(value)
                        except:
                            pass
                        setattr(getattr(self, key), "value", value)
                        if hasattr(getattr(self, key), "style"):
                            getattr(self, key).style = {"description_width": "initial"}

                    elif isinstance(value, BaseApplication) and isinstance(
                        getattr(self, "_" + key, None), BaseApplication
                    ):
                        setattr(self, "_" + key, value)
                    else:
                        setattr(self, key, value)
                except:
                    pass

        self.refinement_list = []
        for label, params in self._refinements.items():
            self.refinement_list += [self.add_refinement_widget(label, params)]

    @property
    def main(self):
        """
        :obj:`ipywidgets.VBox`: A box containing all widgets forming the application.
        """
        if self._main is None:
            self._main = VBox(
                self.required + self.refinement_list + [self.output_panel]
            )

        return self._main

    @property
    def u_cell_size(self):
        return self._u_cell_size

    @property
    def v_cell_size(self):
        return self._v_cell_size

    @property
    def w_cell_size(self):
        return self._w_cell_size

    @property
    def depth_core(self):
        return self._depth_core

    @property
    def horizontal_padding(self):
        return self._horizontal_padding

    @property
    def vertical_padding(self):
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
        self._workspace = workspace
        self._h5file = workspace.h5file
        self.update_objects_choices()
        self.params.input_file.filepath = path.join(
            path.dirname(self._h5file), self.ga_group_name.value + ".ui.json"
        )

    def update_objects_choices(self):
        # Refresh the list of objects for all
        self.update_objects_list()

        for widget in self.refinement_list:
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

        for refinement, params_refinement in zip(
            self.refinement_list, self.params.refinements.values()
        ):
            params_refinement["object"] = refinement.children[1].value
            params_refinement["levels"] = string_2_list(refinement.children[2].value)
            params_refinement["type"] = refinement.children[3].value
            params_refinement["distance"] = refinement.children[4].value

        self.params.write_input_file(name=self.params.ga_group_name)
        self.run(self.params)

    @staticmethod
    def run(params: OctreeParams) -> Octree:
        """
        Create an octree mesh from input values
        """

        workspace = params.workspace
        obj = workspace.get_entity(params.objects)

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

        for label, value in params.refinements.items():

            try:
                uid = (
                    uuid.UUID(value["object"])
                    if isinstance(value["object"], str)
                    else value["object"]
                )
                entity = workspace.get_entity(uid)

            except (ValueError, TypeError):
                continue

            if any(entity):
                print(f"Applying refinement {label} on: {entity[0].name}")
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
        octree = treemesh_2_octree(workspace, treemesh, name=params.ga_group_name)

        if params.monitoring_directory is not None and path.exists(
            params.monitoring_directory
        ):
            BaseApplication.live_link_output(params.monitoring_directory, octree)

        print(
            f"Octree mesh '{octree.name}' completed and exported to {workspace.h5file}"
        )
        return octree

    def add_refinement_widget(self, label: str, params: dict):
        """
        Add a refinement from dictionary
        """
        widget_list = [Label(label.capitalize())]
        for key, value in params.items():
            attr_name = (label + f"{key}").title()

            if "object" in key:
                try:
                    value = uuid.UUID(value)
                except (ValueError, TypeError):
                    value = None

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
    params = OctreeParams.from_path(sys.argv[1])
    print(params.geoh5)
    OctreeMesh.run(params)
