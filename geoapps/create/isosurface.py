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

from geoapps.selection import ObjectDataSelection, TopographyOptions
from geoapps.utils.utils import (
    input_string_2_float,
    iso_surface,
    string_2_list,
    treemesh_2_octree,
)


class IsoSurface(ObjectDataSelection):
    """
    Application for the conversion of conductivity/depth curves to
    a pseudo 3D conductivity model on surface.
    """

    defaults = {
        "add_groups": False,
        "select_multiple": False,
        "h5file": "../../assets/FlinFlon.geoh5",
        "objects": "Inversion_VTEM_Model",
        "data": "Iteration_7_model",
        "max_distance": 500,
        "resolution": 50,
        "contours": "0.005: 0.02: 0.005, 0.0025",
    }

    def __init__(self, **kwargs):
        kwargs = self.apply_defaults(**kwargs)

        self._topography = TopographyOptions()
        self._max_distance = FloatText(
            description="Max Interpolation Distance (m):",
        )
        self._resolution = FloatText(
            description="Base grid resolution (m):",
        )
        self._contours = Text(
            value="", description="Iso-values", disabled=False, continuous_update=False
        )
        self._export_as = Text("Iso_", description="Surface:")

        super().__init__(**kwargs)

        self.ga_group_name.value = "ISO"
        self.data.observe(self.data_change, names="value")
        self.data_change(None)
        self.data.description = "Value fields: "
        self.data_panel = self.main
        self.trigger.on_click(self.compute_trigger)

        self.output_panel = VBox([self.export_as, self.output_panel])
        self._main = HBox(
            [
                VBox(
                    [
                        self.project_panel,
                        self.data_panel,
                        self._contours,
                        self.max_distance,
                        self.resolution,
                        Label("Output"),
                        self.output_panel,
                    ]
                )
            ]
        )

    def compute_trigger(self, _):

        if not self.workspace.get_entity(self.objects.value):
            return

        obj, data_list = self.get_selected_entities()

        levels = input_string_2_float(self.contours.value)

        if levels is None:
            return

        surfaces = iso_surface(
            obj,
            data_list[0].values,
            levels,
            resolution=self.resolution.value,
            max_distance=self.max_distance.value,
        )

        result = []
        for ii, (surface, level) in enumerate(zip(surfaces, levels)):
            if len(surface[0]) > 0 and len(surface[1]) > 0:
                result += [
                    Surface.create(
                        self.workspace,
                        name=self.export_as.value + f"_{level}",
                        vertices=surface[0],
                        cells=surface[1],
                        parent=self.ga_group,
                    )
                ]
        self.result = result
        if self.live_link.value:
            self.live_link_output(self.ga_group)

        self.workspace.finalize()

    def data_change(self, _):

        if self.data.value:
            self.export_as.value = "Iso_" + self.data.value

    @property
    def convert(self):
        """
        ipywidgets.ToggleButton()
        """
        return self._convert

    @property
    def contours(self):
        """
        :obj:`ipywidgets.Text`: String defining sets of contours.
        Contours can be defined over an interval `50:200:10` and/or at a fix value `215`.
        Any combination of the above can be used:
        50:200:10, 215 => Contours between values 50 and 200 every 10, with a contour at 215.
        """
        return self._contours

    @property
    def export_as(self):
        """
        ipywidgets.Text()
        """
        return self._export_as

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
        assert isinstance(
            workspace, Workspace
        ), f"`workspace` must be of class {Workspace}"
        self._workspace = workspace
        self._h5file = workspace.h5file

        # Refresh the list of objects
        self.update_objects_list()


class OctreeMeshCreations(ObjectDataSelection):
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
