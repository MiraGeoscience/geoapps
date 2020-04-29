# pylint: disable=R0912

import uuid
from abc import abstractmethod
from typing import TYPE_CHECKING, List, Optional, Union

from numpy import ndarray

from ..data import Data
from ..groups import PropertyGroup
from ..shared import Entity
from .object_type import ObjectType

if TYPE_CHECKING:
    from .. import workspace


class ObjectBase(Entity):
    """
    Objects base class.
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

        # self._clipping_ids: List[uuid.UUID] = []

        super().__init__(**kwargs)

    @property
    def last_focus(self) -> str:
        """
        Object visible in camera on start: bool
        """
        return self._last_focus

    @last_focus.setter
    def last_focus(self, value: str):
        self._last_focus = value

    @property
    def entity_type(self) -> ObjectType:
        """
        Object type: EntityType
        """
        return self._entity_type

    @property
    def vertices(self):
        """
        Vertices
        """
        return None

    @property
    def cells(self):
        """
        Cells
        """
        return None

    @property
    def faces(self):
        """
        Faces
        """
        return None

    @property
    def n_vertices(self) -> Optional[int]:
        """
        Number of vertices

        :return: Number of vertices
        """
        if self.vertices is not None:
            return self.vertices.shape[0]
        return None

    @property
    def n_cells(self) -> Optional[int]:
        """
        n_cells

        Returns
        -------
        n_cells: int
            Number of cells
        """
        if self.cells is not None:
            return self.cells.shape[0]
        return None

    @property
    def property_groups(self) -> List[PropertyGroup]:
        """
        List of property (data) groups: List[PropertyGroup]
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

    @classmethod
    def find_or_create_type(
        cls, workspace: "workspace.Workspace", **kwargs
    ) -> ObjectType:
        """
        Find or create a type for a given object class

        :param Current workspace: Workspace

        :return: A new or existing object type
        """
        return ObjectType.find_or_create(workspace, cls, **kwargs)

    @classmethod
    @abstractmethod
    def default_type_uid(cls) -> uuid.UUID:
        ...

    def add_data_to_group(
        self, data: Union[Data, uuid.UUID, str], name: str
    ) -> PropertyGroup:
        """
        Append data to a property group where the data can be a Data object, its name
        or uid. The given group identifier (name or uid) is created if it does not exist already.
        All given data must be children of the object.

        :param data: Data object or uuid of data
        :param name: Name of a property group. A new group is created
            if none exist with the given name.

        :return property_group: The target property_group
        """
        # prop_group = self.get_property_group(name)
        #
        # if prop_group is None:
        prop_group = self.find_or_create_property_group(name=name)

        if isinstance(data, Data):
            uid = [data.uid]
        elif isinstance(data, str):
            uid = [obj.uid for obj in self.workspace.get_entity(data)]
        else:
            uid = [data]

        for i in uid:
            assert i in [
                child.uid for child in self.children
            ], f"Given data with uuid {i} does not match any known children"

        prop_group.properties = uid
        self.modified_attributes = "property_groups"

        return prop_group

    def find_or_create_property_group(self, **kwargs) -> PropertyGroup:
        """
        Create property groups from given group names and properties.
        An existing property_group is returned if one exists with the same name.

        :param kwargs: Any arguments taken by the PropertyGroup class

        :return: A new or existing property_group object
        """
        prop_group = PropertyGroup(**kwargs)
        if any([pg.name == prop_group.name for pg in self.property_groups]):
            prop_group = [
                pg for pg in self.property_groups if pg.name == prop_group.name
            ][0]
        else:
            self.property_groups = [prop_group]

        return prop_group

    def get_property_group(
        self, group_id: Union[str, uuid.UUID]
    ) -> Optional[PropertyGroup]:
        """
        Retrieve a property_group from one of its identifier, either by name or uuid

        :param group_id: PropertyGroup identifier, either name or uuid

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

    def add_data(
        self, data: dict, property_group: str = None
    ) -> Union[Data, List[Data]]:
        """
        Create data with association from dictionary of data objects name and values.
        The provided values can either be a dictionary of kwargs accepted by the target
        Data object class or an array of values. If not provided as argument, a data
        association type is assigned based on the length of the given values.

        :param data: Dictionary of data to be added to the object,
            e.g. data = {"data_name1": {'values', values1, 'association': 'VERTEX', ...}, ...}

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
                if isinstance(attr["values"], ndarray):
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
                if key in ["parent", "association"]:
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

    def get_data(self, name: str) -> List[Data]:
        """
        Get data objects by name

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
        :return: List of names of data associated with the object

        """
        name_list = []
        for child in self.children:
            if isinstance(child, Data):
                name_list.append(child.name)
        return sorted(name_list)
