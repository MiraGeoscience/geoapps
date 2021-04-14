#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import os
import re

from geoh5py.objects import Curve, Octree, Points, Surface
from geoh5py.workspace import Workspace
from ipywidgets import Dropdown, FloatText, HBox, IntText, Label, VBox

from geoapps.selection import ObjectDataSelection


class OctreeMesh(ObjectDataSelection):
    """
    Widget used for the creation of an octree meshes
    """

    def __init__(self, **kwargs):

        self.defaults = {}
        for key, params in self.default_ui.items():
            key = re.sub(r"\d+-", "", key)
            if isinstance(params, dict):
                self.defaults[key] = params["value"]
            else:
                self.defaults[key] = params

        kwargs = self.apply_defaults(**kwargs)
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

        self.refinement_list = []
        labels = ["A", "B"]
        for label in labels:
            setattr(
                self,
                f"_refinement_{label}",
                Dropdown(description="Object", style={"description_width": "initial"}),
            )
            setattr(
                self,
                f"_octree_{label}1",
                IntText(
                    description="Level 1",
                ),
            )
            setattr(
                self,
                f"_octree_{label}2",
                IntText(
                    description="Level 2",
                ),
            )
            setattr(
                self,
                f"_octree_{label}3",
                IntText(
                    description="Level 3",
                ),
            )
            setattr(
                self,
                f"_method_{label}",
                Dropdown(description="Type", options=["surface", "radial"]),
            )
            setattr(
                self, f"_max_distance_{label}", FloatText(description="Max distance")
            )

            self.refinement_list.append(
                VBox(
                    [
                        Label(f"Refinement {label}:"),
                        getattr(self, f"_refinement_{label}"),
                        HBox(
                            [
                                getattr(self, f"_octree_{label}1"),
                                getattr(self, f"_octree_{label}2"),
                                getattr(self, f"_octree_{label}3"),
                            ]
                        ),
                        getattr(self, f"_method_{label}"),
                        getattr(self, f"_max_distance_{label}"),
                    ]
                )
            )

        super().__init__(**kwargs)

        self._main = VBox(
            [
                self.project_panel,
                Label("Base Parameters"),
                self._objects,
                Label("Core cell size"),
                HBox([self._u_cell_size, self._v_cell_size, self._w_cell_size]),
                self._depth_core,
                Label("Padding distance"),
                HBox([self.horizontal_padding, self.vertical_padding]),
                Label("Refinements"),
                self.refinement_list[0],
                Label("Optional"),
                self.refinement_list[1],
                self.output_panel,
            ]
        )
        self.objects.description = "Core hull extent:"
        self.trigger.description = "Create"
        self.ga_group_name.description = "Name:"
        self.trigger.on_click(self.trigger_click)

        for obj in self.__dict__:
            if hasattr(getattr(self, obj), "style"):
                getattr(self, obj).style = {"description_width": "initial"}

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
    def max_distance_A(self):
        return self._max_distance_A

    @property
    def max_distance_B(self):
        return self._max_distance_B

    @property
    def horizontal_padding(self):
        return self._horizontal_padding

    @property
    def vertical_padding(self):
        return self._vertical_padding

    @property
    def octree_A1(self):
        return self._octree_A1

    @property
    def octree_A2(self):
        return self._octree_A2

    @property
    def octree_A3(self):
        return self._octree_A3

    @property
    def octree_B1(self):
        return self._octree_B1

    @property
    def octree_B2(self):
        return self._octree_B2

    @property
    def octree_B3(self):
        return self._octree_B3

    @property
    def refinement_A(self):
        return self._refinement_A

    @property
    def refinement_B(self):
        return self._refinement_B

    @property
    def method_A(self):
        return self._method_A

    @property
    def method_B(self):
        return self._method_B

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

    def update_objects_choices(self):
        # Refresh the list of objects for all
        self.update_objects_list()

        for widget in self.refinement_list:
            widget.children[1].options = self.objects.options

    def trigger_click(self, _):

        file = self.save_json_params(self.ga_group_name.value)
        os.system(self.default_ui["executable"] + file)

    @property
    def default_ui(self):
        return {
            "0-h5file": {"enabled": False, "value": "../../assets/FlinFlon.geoh5"},
            "1-objects": {
                "enabled": True,
                "group": "1- Core",
                "label": "Core hull extent",
                "main": True,
                "meshType": "{202C5DB1-A56D-4004-9CAD-BAAFD8899406}",
                "value": "Data_FEM_pseudo3D",
            },
            "2-u_cell_size": {
                "enabled": True,
                "group": "2- Core cell size",
                "label": "Easting (m)",
                "main": True,
                "value": 25,
            },
            "3-v_cell_size": {
                "enabled": True,
                "group": "2- Core cell size",
                "label": "Northing (m)",
                "main": True,
                "value": 25,
            },
            "4-w_cell_size": {
                "enabled": True,
                "group": "2- Core cell size",
                "label": "Vertical (m)",
                "main": True,
                "value": 25,
            },
            "5-horizontal_padding": {
                "enabled": True,
                "group": "3- Padding distance",
                "label": "Horizontal (m)",
                "main": True,
                "value": 0.0,
            },
            "6-vertical_padding": {
                "enabled": True,
                "group": "3- Padding distance",
                "label": "Vertical (m)",
                "main": True,
                "value": 0.0,
            },
            "7-depth_core": {
                "enabled": True,
                "group": "1- Core",
                "label": "Minimum Depth (m)",
                "main": True,
                "value": 500.0,
            },
            "8-refinement_A": {
                "enabled": True,
                "group": "Refinement A",
                "label": "Object",
                "meshType": "{202C5DB1-A56D-4004-9CAD-BAAFD8899406}",
                "value": "Data_FEM_pseudo3D",
            },
            "9-octree_A1": {
                "enabled": True,
                "group": "Refinement A",
                "label": "#Cells @ 1 x core",
                "value": 2,
            },
            "10-octree_A2": {
                "enabled": True,
                "group": "Refinement A",
                "label": "#Cells @ 2 x core",
                "value": 2,
            },
            "11-octree_A3": {
                "enabled": True,
                "group": "Refinement A",
                "label": "#Cells @ 4 x core",
                "value": 2,
            },
            "12-method_A": {
                "choiceList": ["surface", "radial"],
                "enabled": True,
                "group": "Refinement A",
                "label": "Type",
                "value": "radial",
            },
            "17-max_distance_A": {
                "enabled": True,
                "group": "Refinement A",
                "label": "Max Distance (m)",
                "value": 1000.0,
            },
            "18-refinement_B": {
                "enabled": True,
                "group": "Refinement B",
                "label": "Object",
                "meshType": "{202C5DB1-A56D-4004-9CAD-BAAFD8899406}",
                "value": "Topography",
            },
            "19-octree_B1": {
                "enabled": True,
                "group": "Refinement B",
                "label": "#Cells @ 1 x core",
                "value": 2,
            },
            "20-octree_B2": {
                "enabled": True,
                "group": "Refinement B",
                "label": "#Cells @ 2 x core",
                "value": 2,
            },
            "21-octree_B3": {
                "enabled": True,
                "group": "Refinement B",
                "label": "#Cells @ 4 x core",
                "value": 2,
            },
            "22-method_B": {
                "choiceList": ["surface", "radial"],
                "enabled": True,
                "group": "Refinement B",
                "label": "Type",
                "value": "surface",
            },
            "23-max_distance_B": {
                "enabled": True,
                "group": "Refinement B",
                "label": "Max Distance (m)",
                "value": 1000.0,
            },
            "24-ga_group_name": {
                "enabled": True,
                "group": "",
                "label": "Name:",
                "value": "Octree_Mesh",
            },
            "title": "Something",
            "working_directory": "../../assets/",
            "executable": (
                "start cmd.exe @cmd /k " + 'python -m geoapps.utils.create_octree "'
            ),
            "monitoring_directory": "../../assets/Temp",
        }
