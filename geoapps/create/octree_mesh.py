#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import json
import sys
import uuid
from os import path
from typing import Union

from discretize.utils import mesh_builder_xyz, refine_tree_xyz
from geoh5py.objects import Curve, Octree, Points, Surface
from geoh5py.workspace import Workspace
from ipywidgets import Dropdown, FloatText, Label, Layout, Text, VBox, Widget
from ipywidgets.widgets.widget_selection import TraitError

from geoapps.base import BaseApplication
from geoapps.selection import ObjectDataSelection
from geoapps.utils.utils import load_json_params, string_2_list, treemesh_2_octree


class OctreeMesh(ObjectDataSelection):
    """
    Widget used for the creation of an octree meshes
    """

    def __init__(self, **kwargs):

        params = self.default_ui.copy()
        for key, value in kwargs.items():
            params[key] = value

        self.object_types = [Curve, Octree, Points, Surface]
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

        super().__init__(**params)

        self.required = [
            self.project_panel,
            VBox(
                [
                    Label("Base Parameters"),
                    self._objects,
                    self._depth_core,
                    Label("Core cell size"),
                    self._u_cell_size,
                    self._v_cell_size,
                    self._w_cell_size,
                    Label("Padding distance"),
                    self.horizontal_padding,
                    self.vertical_padding,
                ],
                layout=Layout(border="solid"),
            ),
        ]

        self._main = VBox(self.required + self.refinement_list + [self.output_panel])

        self.objects.description = "Core hull extent:"
        self.trigger.description = "Create"
        self.ga_group_name.description = "Name:"
        self.trigger.on_click(self.trigger_click)

        for obj in self.__dict__:
            if hasattr(getattr(self, obj), "style"):
                getattr(self, obj).style = {"description_width": "initial"}

    def __populate__(self, **kwargs):

        refinements = self.get_refinement_params(kwargs)

        for key, value in kwargs.items():

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
        for label, params in refinements.items():
            self.refinement_list += [self.add_refinement_widget(label, params)]

    @property
    def extent(self):
        """
        Alias of `objects` property
        """
        return self.objects

    @extent.setter
    def extent(self, value: Union[uuid.UUID, str]):
        if isinstance(value, str):
            try:
                value = uuid.UUID(value)
            except ValueError:
                print(
                    f"Extent value must be a {uuid.UUID} or a string convertible to uuid"
                )
        self.objects.value = value

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
    def main(self):
        return self._main

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

    @staticmethod
    def get_refinement_params(values: dict):
        """
        Extract refinement parameters from input ui.json
        """
        refinements = {}
        for key, value in values.items():
            if "Refinement" in key:
                if value["group"] not in list(refinements.keys()):
                    refinements[value["group"]] = {}

                refinements[value["group"]][value["label"]] = value["value"]

        return refinements

    def update_objects_choices(self):
        # Refresh the list of objects for all
        self.update_objects_list()

        for widget in self.refinement_list:
            widget.children[1].options = self.objects.options

    def trigger_click(self, _):

        params = self.save_json_params(self.ga_group_name.value, self.default_ui.copy())
        self.run(**params)

    @staticmethod
    def run(**kwargs):
        """
        Create an octree mesh from input values
        """
        workspace = Workspace(kwargs["geoh5"])
        obj = workspace.get_entity(uuid.UUID(kwargs["extent"]["value"]))

        if not any(obj):
            return

        p_d = [
            [
                kwargs["horizontal_padding"]["value"],
                kwargs["horizontal_padding"]["value"],
            ],
            [
                kwargs["horizontal_padding"]["value"],
                kwargs["horizontal_padding"]["value"],
            ],
            [kwargs["vertical_padding"]["value"], kwargs["vertical_padding"]["value"]],
        ]

        print("Setting the mesh extent")
        treemesh = mesh_builder_xyz(
            obj[0].vertices,
            [
                kwargs["u_cell_size"]["value"],
                kwargs["v_cell_size"]["value"],
                kwargs["w_cell_size"]["value"],
            ],
            padding_distance=p_d,
            mesh_type="tree",
            depth_core=kwargs["depth_core"]["value"],
        )

        # Extract refinement
        refinements = OctreeMesh.get_refinement_params(kwargs)

        for label, value in refinements.items():
            print(f"Applying refinement {label}")
            try:
                print(value["Object"])
                entity = workspace.get_entity(uuid.UUID(value["Object"]))
            except (ValueError, TypeError):
                continue

            if any(entity):
                treemesh = refine_tree_xyz(
                    treemesh,
                    entity[0].vertices,
                    method=value["Type"],
                    octree_levels=string_2_list(value["Levels"]),
                    max_distance=value["Max Distance"],
                    finalize=False,
                )

        print("Finalizing...")
        treemesh.finalize()

        print("Writing to file ")
        octree = treemesh_2_octree(
            workspace, treemesh, name=kwargs["ga_group_name"]["value"]
        )

        if kwargs["monitoring_directory"] != "" and path.exists(
            kwargs["monitoring_directory"]
        ):
            BaseApplication.live_link_output(kwargs["monitoring_directory"], octree)

        print(
            f"Octree mesh '{octree.name}' completed and exported to {workspace.h5file}"
        )
        return octree

    def add_refinement_widget(self, label: str, params: dict):
        """
        Add a refinement from dictionary
        """
        widget_list = [Label(label)]
        for key, value in params.items():
            if "Object" in key:
                try:
                    value = uuid.UUID(value)
                except ValueError:
                    value = None

                setattr(
                    self,
                    label + f" {key}",
                    Dropdown(
                        description=key,
                        options=self.objects.options,
                    ),
                )

                try:
                    getattr(self, label + f" {key}").value = value
                except TraitError:
                    pass

            elif "Levels" in key:
                setattr(self, label + f" {key}", Text(description=key, value=value))
            elif "Type" in key:
                setattr(
                    self,
                    label + f" {key}",
                    Dropdown(
                        description=key,
                        options=["surface", "radial"],
                        value=value,
                    ),
                )
            elif "Max" in key:
                setattr(
                    self, label + f" {key}", FloatText(description=key, value=value)
                )
            widget_list += [getattr(self, label + f" {key}", None)]

        return VBox(widget_list, layout=Layout(border="solid"))

    @property
    def default_ui(self):
        return {
            "title": "Octree Mesh Creator",
            "geoh5": "../../assets/FlinFlon.geoh5",
            "extent": {
                "enabled": True,
                "group": "1- Core",
                "label": "Core hull extent",
                "main": True,
                "meshType": [
                    "{202C5DB1-A56D-4004-9CAD-BAAFD8899406}",
                    "{6A057FDC-B355-11E3-95BE-FD84A7FFCB88}",
                    "{F26FEBA3-ADED-494B-B9E9-B2BBCBE298E1}",
                ],
                "value": "{656acd40-25de-4865-814c-cb700f6ee51a}",
            },
            "u_cell_size": {
                "enabled": True,
                "group": "2- Core cell size",
                "label": "Easting (m)",
                "main": True,
                "value": 25,
            },
            "v_cell_size": {
                "enabled": True,
                "group": "2- Core cell size",
                "label": "Northing (m)",
                "main": True,
                "value": 25,
            },
            "w_cell_size": {
                "enabled": True,
                "group": "2- Core cell size",
                "label": "Vertical (m)",
                "main": True,
                "value": 25,
            },
            "horizontal_padding": {
                "enabled": True,
                "group": "3- Padding distance",
                "label": "Horizontal (m)",
                "main": True,
                "value": 1000.0,
            },
            "vertical_padding": {
                "enabled": True,
                "group": "3- Padding distance",
                "label": "Vertical (m)",
                "main": True,
                "value": 1000.0,
            },
            "depth_core": {
                "enabled": True,
                "group": "1- Core",
                "label": "Minimum Depth (m)",
                "main": True,
                "value": 500.0,
            },
            "ga_group_name": {
                "enabled": True,
                "group": "",
                "label": "Name:",
                "value": "Octree_Mesh",
            },
            "Refinement A Object": {
                "enabled": True,
                "group": "Refinement A",
                "label": "Object",
                "meshType": [
                    "{202C5DB1-A56D-4004-9CAD-BAAFD8899406}",
                    "{6A057FDC-B355-11E3-95BE-FD84A7FFCB88}",
                    "{F26FEBA3-ADED-494B-B9E9-B2BBCBE298E1}",
                ],
                "value": "{656acd40-25de-4865-814c-cb700f6ee51a}",
            },
            "Refinement A Levels": {
                "enabled": True,
                "group": "Refinement A",
                "label": "Levels",
                "value": "4,4,4",
            },
            "Refinement A Type": {
                "choiceList": ["surface", "radial"],
                "enabled": True,
                "group": "Refinement A",
                "label": "Type",
                "value": "radial",
            },
            "Refinement A Max Distance": {
                "enabled": True,
                "group": "Refinement A",
                "label": "Max Distance",
                "value": 1000.0,
            },
            "Refinement B Object": {
                "enabled": True,
                "group": "Refinement B",
                "label": "Object",
                "meshType": [
                    "{202C5DB1-A56D-4004-9CAD-BAAFD8899406}",
                    "{6A057FDC-B355-11E3-95BE-FD84A7FFCB88}",
                    "{F26FEBA3-ADED-494B-B9E9-B2BBCBE298E1}",
                ],
                "value": "",
            },
            "Refinement B Levels": {
                "enabled": True,
                "group": "Refinement B",
                "label": "Levels",
                "value": "0,0,2",
            },
            "Refinement B Type": {
                "choiceList": ["surface", "radial"],
                "enabled": True,
                "group": "Refinement B",
                "label": "Type",
                "value": "surface",
            },
            "Refinement B Max Distance": {
                "enabled": True,
                "group": "Refinement B",
                "label": "Max Distance",
                "value": 1000.0,
            },
            "run_command": ("geoapps.create.octree_mesh"),
            "monitoring_directory": "",
        }


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        input_dict = json.load(f)

    # input_params = load_json_params()
    OctreeMesh.run(**input_dict)
