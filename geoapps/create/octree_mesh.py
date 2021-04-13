#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from discretize.utils import meshutils
from geoh5py.objects import Curve, Octree, Points, Surface
from geoh5py.workspace import Workspace
from ipywidgets import Dropdown, FloatText, HBox, Label, Text, VBox

from geoapps.selection import ObjectDataSelection
from geoapps.utils import string_2_list, treemesh_2_octree


class OctreeMesh(ObjectDataSelection):
    """
    Widget used for the creation of an octree meshes
    """

    defaults = {
        "h5file": "../../assets/FlinFlon.geoh5",
        "objects": "Data_FEM_pseudo3D",
        "core_cell_size": "25, 25, 25",
        "depth_core": 500,
        "octree_levels_obs": "0, 0, 0, 2",
        "padding_distance": "0, 0, 0, 0, 0, 0",
        "refinement0": "Topography",
        "ga_group_name": "NewOctree",
    }

    def __init__(self, **kwargs):
        kwargs = self.apply_defaults(**kwargs)
        self.object_types = [Curve, Octree, Points, Surface]
        self._core_cell_size = Text(
            description="Smallest cells",
        )
        self._depth_core = FloatText(
            description="Minimum depth (m)",
        )
        self._padding_distance = Text(
            description="Padding [W,E,N,S,D,U] (m)",
        )

        self._refinement0 = Dropdown(
            description="Object", style={"description_width": "initial"}
        )
        self._refinement1 = Dropdown(
            description="Object", style={"description_width": "initial"}
        )
        self._refinement2 = Dropdown(
            description="Object", style={"description_width": "initial"}
        )

        self.refinement_list = []
        labels = ["", "B", "C"]
        for ii in range(3):
            self.refinement_list.append(
                VBox(
                    [
                        Label(f"Refinement {labels[ii]}:"),
                        getattr(self, f"_refinement{ii}"),
                        HBox(
                            [
                                Text(
                                    description="# Cells / Octree size",
                                    value="2,2,2",
                                    style={"description_width": "initial"},
                                ),
                                Dropdown(
                                    description="Type", options=["Surface", "Radial"]
                                ),
                                FloatText(description="Max distance", value=1000),
                            ]
                        ),
                    ]
                )
            )

        super().__init__(**kwargs)

        self._main = VBox(
            [
                self.project_panel,
                Label("Base Parameters"),
                self._objects,
                self._core_cell_size,
                self._depth_core,
                self._padding_distance,
                self.refinement_list[0],
                Label("Optional"),
                self.refinement_list[1],
                self.refinement_list[2],
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
    def core_cell_size(self):
        return self._core_cell_size

    @property
    def depth_core(self):
        return self._depth_core

    @property
    def main(self):
        return self._main

    @property
    def padding_distance(self):
        return self._padding_distance

    @property
    def refinement0(self):
        return self._refinement0

    @property
    def refinement1(self):
        return self._refinement1

    @property
    def refinement2(self):
        return self._refinement2

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

        obj, _ = self.get_selected_entities()
        p_d = string_2_list(self.padding_distance.value)
        p_d = [
            [p_d[0], p_d[1]],
            [p_d[2], p_d[3]],
            [p_d[4], p_d[5]],
        ]
        treemesh = meshutils.mesh_builder_xyz(
            obj.vertices,
            string_2_list(self.core_cell_size.value),
            padding_distance=p_d,
            mesh_type="tree",
            depth_core=self.depth_core.value,
        )

        for child in self.refinement_list:

            entity = self.workspace.get_entity(child.children[1].value)
            if any(entity):
                treemesh = meshutils.refine_tree_xyz(
                    treemesh,
                    entity[0].vertices,
                    method=child.children[2].children[1].value.lower(),
                    octree_levels=string_2_list(child.children[2].children[0].value),
                    max_distance=child.children[2].children[2].value,
                    finalize=False,
                )

        treemesh.finalize()
        treemesh_2_octree(self.workspace, treemesh, name=self.ga_group_name.value)
