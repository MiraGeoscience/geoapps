#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import os

from geoh5py.objects import Curve, Octree, Points, Surface
from geoh5py.workspace import Workspace
from ipywidgets import Dropdown, FloatText, HBox, IntText, Label, Layout, VBox, Widget

from geoapps.base import BaseApplication
from geoapps.selection import ObjectDataSelection


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

        refinements = {}
        for key, value in kwargs.items():

            if "Refinement" in key:
                if value["group"] not in list(refinements.keys()):
                    refinements[value["group"]] = [value]
                else:
                    refinements[value["group"]].append(value)

            elif hasattr(self, "_" + key) or hasattr(self, key):

                if isinstance(value, dict) and "value" in list(value.keys()):
                    value = value["value"]

                try:
                    if isinstance(getattr(self, key, None), Widget) and not isinstance(
                        value, Widget
                    ):
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
        for refinement in refinements.values():
            self.refinement_list += [self.add_refinement_widget(refinement)]

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

    def update_objects_choices(self):
        # Refresh the list of objects for all
        self.update_objects_list()

        for widget in self.refinement_list:
            widget.children[1].options = self.objects.options

    def trigger_click(self, _):

        file = self.save_json_params(self.ga_group_name.value)
        os.system(self.default_ui["executable"] + file)

    def add_refinement_widget(self, params: dict):
        """
        Add a refinement from dictionary
        """
        widget_list = [Label(params[0]["group"])]
        for param in params:
            if "Object" in param["label"]:
                widget_list += [
                    Dropdown(
                        description=param["label"],
                        options=self.objects.options,
                        value=param["value"],
                    )
                ]
            elif "Level" in param["label"]:
                widget_list += [
                    IntText(description=param["label"], value=param["value"])
                ]
            elif "Type" in param["label"]:
                widget_list += [
                    Dropdown(
                        description=param["label"],
                        options=["surface", "radial"],
                        value=param["value"],
                    )
                ]
            elif "Max" in param["label"]:
                widget_list += [
                    FloatText(description=param["label"], value=param["value"])
                ]

        return VBox(widget_list, layout=Layout(border="solid"))

    @property
    def default_ui(self):
        return {
            "title": "Octree Mesh Creator",
            "workspace_geoh5": "../../assets/FlinFlon.geoh5",
            "objects": {
                "enabled": True,
                "group": "1- Core",
                "label": "Core hull extent",
                "main": True,
                "meshType": [
                    "{202C5DB1-A56D-4004-9CAD-BAAFD8899406}",
                    "{6A057FDC-B355-11E3-95BE-FD84A7FFCB88}",
                    "{F26FEBA3-ADED-494B-B9E9-B2BBCBE298E1}",
                ],
                "value": "Data_FEM_pseudo3D",
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
                "value": 0.0,
            },
            "vertical_padding": {
                "enabled": True,
                "group": "3- Padding distance",
                "label": "Vertical (m)",
                "main": True,
                "value": 0.0,
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
            "Octree Refinement A": {
                "enabled": True,
                "group": "Octree Refinement A",
                "label": "Object",
                "meshType": [
                    "{202C5DB1-A56D-4004-9CAD-BAAFD8899406}",
                    "{6A057FDC-B355-11E3-95BE-FD84A7FFCB88}",
                    "{F26FEBA3-ADED-494B-B9E9-B2BBCBE298E1}",
                ],
                "value": "Data_FEM_pseudo3D",
            },
            "Octree Refinement A Level 1": {
                "enabled": True,
                "group": "Octree Refinement A",
                "label": "Level 1",
                "value": 2,
            },
            "Octree Refinement A Level 2": {
                "enabled": True,
                "group": "Octree Refinement A",
                "label": "Level 2",
                "value": 2,
            },
            "Octree Refinement A Level 3": {
                "enabled": True,
                "group": "Octree Refinement A",
                "label": "Level 3",
                "value": 2,
            },
            "Octree Refinement A Type": {
                "choiceList": ["surface", "radial"],
                "enabled": True,
                "group": "Octree Refinement A",
                "label": "Type",
                "value": "radial",
            },
            "Octree Refinement A Max Distance": {
                "enabled": True,
                "group": "Octree Refinement A",
                "label": "Max Distance (m)",
                "value": 1000.0,
            },
            "Octree Refinement B": {
                "enabled": True,
                "group": "Octree Refinement B",
                "label": "Object",
                "meshType": [
                    "{202C5DB1-A56D-4004-9CAD-BAAFD8899406}",
                    "{6A057FDC-B355-11E3-95BE-FD84A7FFCB88}",
                    "{F26FEBA3-ADED-494B-B9E9-B2BBCBE298E1}",
                ],
                "value": "Topography",
            },
            "Octree Refinement B Level 1": {
                "enabled": True,
                "group": "Octree Refinement B",
                "label": "Level 1",
                "value": 0,
            },
            "Octree Refinement B Level 2": {
                "enabled": True,
                "group": "Octree Refinement B",
                "label": "Level 2",
                "value": 0,
            },
            "Octree Refinement B Level 3": {
                "enabled": True,
                "group": "Octree Refinement B",
                "label": "Level 3",
                "value": 2,
            },
            "Octree Refinement B Type": {
                "choiceList": ["surface", "radial"],
                "enabled": True,
                "group": "Octree Refinement B",
                "label": "Type",
                "value": "surface",
            },
            "Octree Refinement B Max Distance": {
                "enabled": True,
                "group": "Octree Refinement B",
                "label": "Max Distance (m)",
                "value": 1000.0,
            },
            "executable": (
                "start cmd.exe @cmd /k " + 'python -m geoapps.utils.create_octree "'
            ),
            "monitoring_directory": "../../assets/Temp",
        }
