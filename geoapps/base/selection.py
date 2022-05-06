#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from uuid import UUID

import numpy as np
from geoh5py.data import ReferencedData
from geoh5py.objects.object_base import ObjectBase
from geoh5py.workspace import Workspace

from geoapps.utils import soft_import

widgets = soft_import("ipywidgets")
(Dropdown, FloatText, SelectMultiple, VBox) = soft_import(
    "ipywidgets", objects=["Dropdown", "FloatText", "SelectMultiple", "VBox"]
)

from geoapps.base.application import BaseApplication
from geoapps.utils.general import find_value, sorted_children_dict


class ObjectDataSelection(BaseApplication):
    """
    Application to select an object and corresponding data
    """

    _data = None
    _objects = None
    _add_groups = False
    _select_multiple = False
    _object_types = ()
    _exclusion_types = ()
    _add_xyz = True
    _find_label = []

    def __init__(self, **kwargs):
        self._data_panel = None

        super().__init__(**kwargs)

    @property
    def add_groups(self) -> bool:
        """
        Add data groups to the list of data choices
        """
        return self._add_groups

    @add_groups.setter
    def add_groups(self, value):
        assert isinstance(value, (bool, str)), "add_groups must be of type bool"
        self._add_groups = value

    @property
    def add_xyz(self) -> bool:
        """
        Add cell or vertices XYZ coordinates in data list
        """
        return self._add_xyz

    @add_xyz.setter
    def add_xyz(self, value):
        assert isinstance(value, (bool, str)), "add_xyz must be of type bool"
        self._add_xyz = value

    @property
    def data(self) -> Dropdown | SelectMultiple:
        """
        Data selector
        """
        if getattr(self, "_data", None) is None:
            if self.select_multiple:
                self._data = SelectMultiple(
                    description="Data: ",
                )
            else:
                self._data = Dropdown(
                    description="Data: ",
                )
            if self._objects is not None:
                self.update_data_list(None)

        return self._data

    @data.setter
    def data(self, value):
        assert isinstance(
            value, (Dropdown, SelectMultiple)
        ), f"'Objects' must be of type {Dropdown} or {SelectMultiple}"
        self._data = value

    @property
    def data_panel(self) -> VBox:
        if getattr(self, "_data_panel", None) is None:
            self._data_panel = VBox([self.objects, self.data])

        return self._data_panel

    @property
    def main(self) -> VBox:
        """
        :obj:`ipywidgets.VBox`: A box containing all widgets forming the application.
        """
        # self.__populate__(**self.defaults)
        if self._main is None:
            self._main = self.data_panel
            self.update_data_list(None)

        return self._main

    @property
    def objects(self) -> Dropdown:
        """
        Object selector
        """
        if getattr(self, "_objects", None) is None:
            self.objects = Dropdown(description="Object:")

        return self._objects

    @objects.setter
    def objects(self, value):
        assert isinstance(value, Dropdown), f"'Objects' must be of type {Dropdown}"
        self._objects = value
        self._objects.observe(self.update_data_list, names="value")
        self.update_data_list(None)

    @property
    def object_types(self):
        """
        Entity type
        """
        return self._object_types

    @object_types.setter
    def object_types(self, entity_types):
        if not isinstance(entity_types, tuple):
            entity_types = tuple(entity_types)

        for entity_type in entity_types:
            assert issubclass(
                entity_type, ObjectBase
            ), f"Provided object_types must be instances of {ObjectBase}"

        self._object_types = entity_types

    @property
    def exclusion_types(self):
        """
        Entity type
        """
        if getattr(self, "_exclusion_types", None) is None:
            self._exclusion_types = []

        return self._exclusion_types

    @exclusion_types.setter
    def exclusion_types(self, entity_types):
        if not isinstance(entity_types, tuple):
            entity_types = tuple(entity_types)

        for entity_type in entity_types:
            assert issubclass(
                entity_type, ObjectBase
            ), f"Provided exclusion_types must be instances of {ObjectBase}"

        self._exclusion_types = tuple(entity_types)

    @property
    def find_label(self):
        """
        Object selector
        """
        if getattr(self, "_find_label", None) is None:
            return []

        return self._find_label

    @find_label.setter
    def find_label(self, values):
        """
        Object selector
        """
        if not isinstance(values, list):
            values = [values]

        for value in values:
            assert isinstance(
                value, str
            ), f"Labels to find must be strings. Value {value} of type {type(value)} provided"
        self._find_label = values

    @property
    def select_multiple(self):
        """
        bool: ALlow to select multiple data
        """
        if getattr(self, "_select_multiple", None) is None:
            self._select_multiple = False

        return self._select_multiple

    @select_multiple.setter
    def select_multiple(self, value):
        if getattr(self, "_data", None) is not None:
            options = self._data.options
        else:
            options = []

        self._select_multiple = value

        if value:
            self._data = SelectMultiple(description="Data: ", options=options)
        else:
            self._data = Dropdown(description="Data: ", options=options)

    @property
    def workspace(self) -> Workspace | None:
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

        # Refresh the list of objects
        self.update_objects_list()

    def get_selected_entities(self):
        """
        Get entities from an active geoh5py Workspace
        """
        if getattr(self, "_workspace", None) is not None and self._workspace.get_entity(
            self.objects.value
        ):
            for entity in self._workspace.get_entity(self.objects.value):
                if isinstance(entity, ObjectBase):
                    obj = entity

            if isinstance(self.data, Dropdown):
                values = [self.data.value]
            else:
                values = self.data.value

            data = []
            for value in values:
                if any([pg.uid == value for pg in obj.property_groups]):
                    data += [
                        self.workspace.get_entity(prop)[0]
                        for prop in obj.find_or_create_property_group(
                            name=self.data.uid_name_map[value]
                        ).properties
                    ]
                elif self.workspace.get_entity(value):
                    data += self.workspace.get_entity(value)

            return obj, data
        else:
            return None, None

    def update_data_list(self, _):
        refresh = self.refresh.value
        self.refresh.value = False
        if getattr(self, "_workspace", None) is not None and self._workspace.get_entity(
            self.objects.value
        ):
            for entity in self._workspace.get_entity(self.objects.value):
                if isinstance(entity, ObjectBase):
                    obj = entity

            if getattr(obj, "get_data_list", None) is None:
                return

            options = [["", None]]

            if (self.add_groups or self.add_groups == "only") and obj.property_groups:
                options = (
                    options
                    + [["-- Groups --", None]]
                    + [[p_g.name, p_g.uid] for p_g in obj.property_groups]
                )

            if self.add_groups != "only":
                options += [["--- Channels ---", None]]

                children = sorted_children_dict(obj)
                excl = ["visual parameter"]
                options += [
                    [k, v] for k, v in children.items() if k.lower() not in excl
                ]

                if self.add_xyz:
                    options += [["X", "X"], ["Y", "Y"], ["Z", "Z"]]

            value = self.data.value
            self.data.options = options

            self.update_uid_name_map()

            if self.select_multiple and any([val in options for val in value]):
                self.data.value = [val for val in value if val in options]
            elif value in dict(options).values():
                self.data.value = value
            elif self.find_label:
                self.data.value = find_value(self.data.options, self.find_label)
        else:
            self.data.options = []
            self.data.uid_name_map = {}

        self.refresh.value = refresh

    def update_objects_list(self):
        if getattr(self, "_workspace", None) is not None:
            value = self.objects.value
            data = self.data.value

            if len(self.object_types) > 0:
                obj_list = [
                    obj
                    for obj in self._workspace.objects
                    if isinstance(obj, self.object_types)
                ]
            else:
                obj_list = self._workspace.objects

            if len(self.exclusion_types) > 0:
                obj_list = [
                    obj for obj in obj_list if not isinstance(obj, self.exclusion_types)
                ]

            options = [["", None]] + [
                [obj.parent.name + "/" + obj.name, obj.uid] for obj in obj_list
            ]

            self.objects.options = options

            if value in dict(self.objects.options).values():
                self.objects.value = value
                self.data.value = data

    def update_uid_name_map(self):
        """
        Update the dictionary that maps uuid to name.
        """
        uid_name = {}
        for key, value in self.data.options:
            if isinstance(value, UUID):
                uid_name[value] = key
            elif isinstance(value, str) and value in "XYZ":
                uid_name[value] = value
        self.data.uid_name_map = uid_name


class LineOptions(ObjectDataSelection):
    """
    Unique lines selection from selected data channel
    """

    defaults = {"find_label": "line"}
    _multiple_lines = None
    _add_xyz = False

    def __init__(self, **kwargs):

        self.defaults.update(**kwargs)

        super().__init__(**self.defaults)

        self.objects.observe(self.update_data_list, names="value")
        self.data.observe(self.update_line_list, names="value")
        self.data.description = "Lines field"

    @property
    def main(self):
        if self._main is None:
            self._main = VBox([self._data, self.lines])

        return self._main

    @property
    def lines(self):
        """
        Widget.SelectMultiple or Widget.Dropdown
        """
        if getattr(self, "_lines", None) is None:
            if self.multiple_lines:
                self._lines = widgets.SelectMultiple(
                    description="Select lines:",
                )
            else:
                self._lines = widgets.Dropdown(
                    description="Select line:",
                )

        return self._lines

    @property
    def multiple_lines(self):
        if getattr(self, "_multiple_lines", None) is None:
            self._multiple_lines = True

        return self._multiple_lines

    @multiple_lines.setter
    def multiple_lines(self, value):
        assert isinstance(
            value, bool
        ), f"'multiple_lines' property must be of type {bool}"
        self._multiple_lines = value

    def update_line_list(self, _):
        _, data = self.get_selected_entities()
        if data and getattr(data[0], "values", None) is not None:
            if isinstance(data[0], ReferencedData):
                self.lines.options = [""] + list(data[0].value_map.map.values())
            else:
                self.lines.options = [""] + np.unique(data[0].values).tolist()

        else:
            self.lines.options = [""]


class TopographyOptions(ObjectDataSelection):
    """
    Define the topography used by the inversion
    """

    defaults = {}

    def __init__(
        self, option_list=["None", "Object", "Relative to Sensor", "Constant"], **kwargs
    ):
        self.defaults.update(**kwargs)
        self.find_label = ["topo", "dem", "dtm", "elevation", "Z"]
        self._offset = FloatText(description="Vertical offset (+ve up)")
        self._constant = FloatText(
            description="Elevation (m)",
        )
        self.option_list = {
            "None": widgets.Label("No topography"),
            "Object": self.data_panel,
            "Relative to Sensor": self._offset,
            "Constant": self._constant,
        }
        self._options = widgets.RadioButtons(
            options=option_list,
            description="Define by:",
        )
        self.options.observe(self.update_options)

        super().__init__(**self.defaults)

    @property
    def panel(self):
        return self._panel

    @property
    def constant(self):
        return self._constant

    @property
    def main(self):
        if self._main is None:
            self._main = VBox([self.options, self.option_list[self.options.value]])

        return self._main

    @property
    def offset(self):
        return self._offset

    @property
    def options(self):
        return self._options

    def update_options(self, _):
        self.main.children = [
            self.options,
            self.option_list[self.options.value],
        ]
