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

import uuid
from typing import List, Union

from ..data import DataAssociationEnum
from ..shared import Entity


class PropertyGroup:
    """
    Property group listing data children of an object.
    This group is not registered to the workspace and only visible to the parent object.
    """

    _attribute_map = {
        "Association": "association",
        "Group Name": "name",
        "ID": "uid",
        "Properties": "properties",
        "Property Group Type": "property_group_type",
    }

    def __init__(self, **kwargs):

        self._name = "prop_group"
        self._uid = uuid.uuid4()
        self._association: DataAssociationEnum = DataAssociationEnum.VERTEX
        self._properties: List[uuid.UUID] = []
        self._property_group_type = "multi-element"
        self._parent = None

        for attr, item in kwargs.items():
            try:
                if attr in self._attribute_map.keys():
                    attr = self._attribute_map[attr]
                setattr(self, attr, item)
            except AttributeError:
                continue

    @property
    def parent(self) -> Entity:
        """
        The parent :obj:`~geoh5py.objects.object_base.ObjectBase`
        """
        return self._parent

    @parent.setter
    def parent(self, parent: Entity):
        self._parent = parent

    @property
    def attribute_map(self) -> dict:
        """
        :obj:`dict` Attribute names mapping between geoh5 and geoh5py
        """
        return self._attribute_map

    @property
    def uid(self) -> uuid.UUID:
        """
        :obj:`uuid.UUID` Unique identifier
        """
        return self._uid

    @uid.setter
    def uid(self, uid: Union[str, uuid.UUID]):
        if isinstance(uid, str):
            uid = uuid.UUID(uid)
        self._uid = uid

    @property
    def name(self) -> str:
        """
        :obj:`str` Name of the group
        """
        return self._name

    @name.setter
    def name(self, new_name: str):
        self._name = new_name

    @property
    def association(self) -> DataAssociationEnum:
        """
        :obj:`~geoh5py.data.data_association_enum.DataAssociationEnum` Data association
        """
        return self._association

    @association.setter
    def association(self, value):
        if self._association is None:

            if isinstance(value, str):
                value = getattr(DataAssociationEnum, value.upper())

            assert isinstance(
                value, DataAssociationEnum
            ), f"Association must be of type {DataAssociationEnum}"
            self._association = value

    @property
    def properties(self) -> List[uuid.UUID]:
        """
        List of unique identifiers for the :obj:`~geoh5py.data.data.Data`
        contained in the property group.
        """
        return self._properties

    @properties.setter
    def properties(self, uids: List[Union[str, uuid.UUID]]):

        properties = []
        for uid in uids:
            if isinstance(uid, str):
                uid = uuid.UUID(uid)
            properties.append(uid)
        self._properties += properties

    @property
    def property_group_type(self) -> str:
        return self._property_group_type

    @property_group_type.setter
    def property_group_type(self, group_type: str):
        self._property_group_type = group_type
