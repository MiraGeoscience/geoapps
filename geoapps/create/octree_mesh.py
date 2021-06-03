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
from geoapps.io.Octree.constants import default_ui_json
from geoapps.io.Octree.params import OctreeParams
from geoapps.selection import ObjectDataSelection
from geoapps.utils.utils import string_2_list, treemesh_2_octree


class OctreeMesh(ObjectDataSelection):
    """
    Widget used for the creation of an octree meshes
    """

    def __init__(self, **kwargs):

        self.params = OctreeParams()

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

        self.refinements = self.params.refinements
        self.refinement_list = []

        super().__init__(**self.params.__dict__)

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
        for label, params in self.refinements.items():
            self.refinement_list += [self.add_refinement_widget(label, params)]

    # @property
    # def extent(self):
    #     """
    #     Alias of `objects` property
    #     """
    #     return self.objects
    #
    # @extent.setter
    # def extent(self, value: Union[uuid.UUID, str]):
    #     if isinstance(value, str):
    #         try:
    #             value = uuid.UUID(value)
    #         except ValueError:
    #             print(
    #                 f"Extent value must be a {uuid.UUID} or a string convertible to uuid"
    #             )
    #     self.objects.value = value

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
        self.params.input_file.filepath = path.join(
            path.dirname(self._h5file), self.ga_group_name.value + ".ui.json"
        )

    # @staticmethod
    # def get_refinement_params(values: dict):
    #     """
    #     Extract refinement parameters from input ui.json
    #     """
    #     refinements = {}
    #     for key, value in values.items():
    #         if "Refinement" in key:
    #             if value["group"] not in list(refinements.keys()):
    #                 refinements[value["group"]] = {}
    #
    #             refinements[value["group"]][value["label"]] = value["value"]
    #
    #     return refinements

    def update_objects_choices(self):
        # Refresh the list of objects for all
        self.update_objects_list()

        for widget in self.refinement_list:
            widget.children[1].options = self.objects.options

    def trigger_click(self, _):

        # params = self.save_json_params(self.ga_group_name.value, self.default_ui.copy())
        for key, value in self.__dict__.items():
            try:
                if isinstance(getattr(self, key), Widget):
                    value = getattr(self, key).value
                    if isinstance(value, uuid.UUID):
                        value = str(value)

                    setattr(self.params, key, value)
            except AttributeError:
                continue

        self.params.update_input_data()
        self.params.write_input_file()
        self.run(self.params.input_file.filepath)

    @staticmethod
    def run(file_name):
        """
        Create an octree mesh from input values
        """
        params = OctreeParams.from_path(file_name)

        workspace = params.workspace
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
    # with open(sys.argv[1]) as f:
    #     input_dict = json.load(f)

    # input_params = load_json_params()
    OctreeMesh.run(sys.argv[1])
