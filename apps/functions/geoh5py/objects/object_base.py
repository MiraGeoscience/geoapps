#  Copyright (c) 2020 Mira Geoscience Ltd.
#
#  This file is part of geoh5py.
#
#  geoh5py is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  geoh5py is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with geoh5py.  If not, see <https://www.gnu.org/licenses/>.

# pylint: disable=R0912

import uuid
from abc import abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np

from ..data import CommentsData, Data
from ..groups import PropertyGroup
from ..shared import Entity
from .object_type import ObjectType

if TYPE_CHECKING:
    from .. import workspace


class ObjectBase(Entity):
    """
    Object base class.
    """

    _attribute_map = Entity._attribute_map.copy()
    _attribute_map.update(
        {"Last focus": "last_focus", "PropertyGroups": "property_groups"}
    )

    def __init__(self, object_type: ObjectType, **kwargs):
        assert object_type is not None
        self._entity_type = object_type
        self._property_groups: List[PropertyGroup] = []
        self._last_focus = "None"
        self._comments = None
        # self._clipping_ids: List[uuid.UUID] = []

        super().__init__(**kwargs)

    def add_comment(self, comment: str, author: str = None):
        """
        Add text comment to an object.

        :param comment: Text to be added as comment.
        :param author: Author's name or :obj:`~geoh5py.workspace.workspace.Worspace.contributors`
        """

        date = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        if author is None:
            author = ",".join(self.workspace.contributors)

        comment_dict = {"Author": author, "Date": date, "Text": comment}

        if self.comments is None:
            self.add_data(
                {
                    "UserComments": {
                        "values": [comment_dict],
                        "association": "OBJECT",
                        "entity_type": {"primitive_type": "TEXT"},
                    }
                }
            )
        else:
            self.comments.values = self.comments.values + [comment_dict]

    def add_data(
        self, data: dict, property_group: str = None
    ) -> Union[Data, List[Data]]:
        """
        Create :obj:`~geoh5py.data.data.Data` from dictionary of name and arguments.
        The provided arguments can be any property of the target class.

        :param data: Dictionary of data to be added to the object, e.g.

        .. code-block:: python

            data = {
                "data_A": {
                    'values', [v_1, v_2, ...],
                    'association': 'VERTEX'
                    },
                "data_B": {
                    'values', [v_1, v_2, ...],
                    'association': 'CELLS'
                    },
            }

        :return data_list: List of new Data objects
        """
        data_objects = []
        for name, attr in data.items():
            assert isinstance(attr, dict), (
                f"Given value to data {name} should of type {dict}. "
                f"Type {type(attr)} given instead."
            )
            assert "values" in list(
                attr.keys()
            ), f"Given attr for data {name} should include 'values'"

            attr["name"] = name

            if "association" not in list(attr.keys()):
                if (
                    getattr(self, "n_cells", None) is not None
                    and attr["values"].ravel().shape[0] == self.n_cells
                ):
                    attr["association"] = "CELL"
                elif (
                    getattr(self, "n_vertices", None) is not None
                    and attr["values"].ravel().shape[0] == self.n_vertices
                ):
                    attr["association"] = "VERTEX"
                else:
                    attr["association"] = "OBJECT"

            if "entity_type" in list(attr.keys()):
                entity_type = attr["entity_type"]
            else:
                if isinstance(attr["values"], np.ndarray):
                    entity_type = {"primitive_type": "FLOAT"}
                elif isinstance(attr["values"], str):
                    entity_type = {"primitive_type": "TEXT"}
                else:
                    raise NotImplementedError(
                        "Only add_data values of type FLOAT and TEXT have been implemented"
                    )

            # Re-order to set parent first
            kwargs = {"parent": self, "association": attr["association"]}
            for key, val in attr.items():
                if key in ["parent", "association", "entity_type"]:
                    continue
                kwargs[key] = val

            data_object = self.workspace.create_entity(
                Data, entity=kwargs, entity_type=entity_type
            )

            if property_group is not None:
                self.add_data_to_group(data_object, property_group)

            data_objects.append(data_object)

        if len(data_objects) == 1:
            return data_object

        return data_objects

    def add_data_to_group(
        self, data: Union[List, Data, uuid.UUID, str], name: str
    ) -> PropertyGroup:
        """
        Append data children to a :obj:`~geoh5py.groups.property_group.PropertyGroup`
        All given data must be children of the parent object.

        :param data: :obj:`~geoh5py.data.data.Data` object,
            :obj:`~geoh5py.shared.entity.Entity.uid` or
            :obj:`~geoh5py.shared.entity.Entity.name` of data
        :param name: Name of a :obj:`~geoh5py.groups.property_group.PropertyGroup`.
            A new group is created if none exist with the given name.

        :return property_group: The target property_group
        """
        prop_group = self.find_or_create_property_group(name=name)

        children_uid = [child.uid for child in self.children]

        def reference_to_uid(value):
            if isinstance(value, Data):
                uid = [value.uid]
            elif isinstance(value, str):
                uid = [
                    obj.uid
                    for obj in self.workspace.get_entity(value)
                    if obj.uid in children_uid
                ]
            elif isinstance(value, uuid.UUID):
                uid = [value]
            return uid

        if isinstance(data, list):
            uid = []
            for datum in data:
                uid += reference_to_uid(datum)
        else:
            uid = reference_to_uid(data)

        for i in uid:
            assert i in [
                child.uid for child in self.children
            ], f"Given data with uuid {i} does not match any known children"

        prop_group.properties = uid
        self.modified_attributes = "property_groups"

        return prop_group

    @property
    def cells(self):
        ...

    @property
    def comments(self):
        """
        Fetch a :obj:`~geoh5py.data.text_data.CommentsData` entity from children.
        """
        for child in self.children:
            if isinstance(child, CommentsData):
                return child

        return None

    @classmethod
    @abstractmethod
    def default_type_uid(cls) -> uuid.UUID:
        ...

    @property
    def entity_type(self) -> ObjectType:
        """
        :obj:`~geoh5py.shared.entity_type.EntityType`: Object type
        """
        return self._entity_type

    @property
    def faces(self):
        ...

    @classmethod
    def find_or_create_type(
        cls, workspace: "workspace.Workspace", **kwargs
    ) -> ObjectType:
        """
        Find or create a type for a given object class

        :param workspace: Target :obj:`~geoh5py.workspace.workspace.Workspace`

        :return: A new or existing object type
        """
        return ObjectType.find_or_create(workspace, cls, **kwargs)

    def find_or_create_property_group(self, **kwargs) -> PropertyGroup:
        """
        Find or create :obj:`~geoh5py.groups.property_group.PropertyGroup`
        from given name and properties.

        :param kwargs: Any arguments taken by the
            :obj:`~geoh5py.groups.property_group.PropertyGroup` class

        :return: A new or existing :obj:`~geoh5py.groups.property_group.PropertyGroup`
        """
        prop_group = PropertyGroup(**kwargs)
        if any([pg.name == prop_group.name for pg in self.property_groups]):
            prop_group = [
                pg for pg in self.property_groups if pg.name == prop_group.name
            ][0]
        else:
            self.property_groups = [prop_group]

        return prop_group

    def get_data(self, name: str) -> List[Data]:
        """
        Get a child :obj:`~geoh5py.data.data.Data` by name.

        :param name: Name of the target child data

        :return: A list of children Data objects
        """
        entity_list = []

        for child in self.children:
            if isinstance(child, Data) and child.name == name:
                entity_list.append(child)

        return entity_list

    def get_data_list(self) -> List[str]:
        """
        Get a list of names of all children :obj:`~geoh5py.data.data.Data`.

        :return: List of names of data associated with the object
        """
        name_list = []
        for child in self.children:
            if isinstance(child, Data):
                name_list.append(child.name)
        return sorted(name_list)

    def get_property_group(
        self, group_id: Union[str, uuid.UUID]
    ) -> Optional[PropertyGroup]:
        """
        Retrieve a :obj:`~geoh5py.groups.property_group.PropertyGroup` from one of its
        identifier.

        :param uid: PropertyGroup identifier, either by its name or uuid

        :return: PropertyGroup with the given name
        """
        if isinstance(group_id, uuid.UUID):
            groups_list = [pg for pg in self.property_groups if pg.uid == group_id]

        else:  # Extract all groups uuid with matching group_id
            groups_list = [pg for pg in self.property_groups if pg.name == group_id]

        try:
            prop_group = groups_list[0]
        except IndexError:
            print(f"No property_group {group_id} found.")
            return None

        return prop_group

    @property
    def last_focus(self) -> str:
        """
        :obj:`bool`: Object visible in camera on start
        """
        return self._last_focus

    @last_focus.setter
    def last_focus(self, value: str):
        self._last_focus = value

    @property
    def n_cells(self) -> Optional[int]:
        """
        :obj:`int`: Number of cells
        """
        if self.cells is not None:
            return self.cells.shape[0]
        return None

    @property
    def n_vertices(self) -> Optional[int]:
        """
        :obj:`int`: Number of vertices
        """
        if self.vertices is not None:
            return self.vertices.shape[0]
        return None

    @property
    def property_groups(self) -> List[PropertyGroup]:
        """
        :obj:`list` of :obj:`~geoh5py.groups.property_group.PropertyGroup`
        """
        return self._property_groups

    @property_groups.setter
    def property_groups(self, prop_groups: List[PropertyGroup]):
        # Check for existing property_group
        for prop_group in prop_groups:
            if not any(
                [pg.uid == prop_group.uid for pg in self.property_groups]
            ) and not any([pg.name == prop_group.name for pg in self.property_groups]):
                prop_group.parent = self

                self.modified_attributes = "property_groups"
                self._property_groups = self.property_groups + [prop_group]

    @property
    def vertices(self):
        ...
