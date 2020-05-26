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

from abc import abstractmethod
from typing import TYPE_CHECKING, Optional, Type, Union

from ..shared import Entity
from .data_association_enum import DataAssociationEnum
from .data_type import DataType
from .primitive_type_enum import PrimitiveTypeEnum

if TYPE_CHECKING:
    from .. import workspace


class Data(Entity):
    """
    Base class for Data entities.
    """

    _attribute_map = Entity._attribute_map.copy()
    _attribute_map.update({"Association": "association"})

    def __init__(self, data_type: DataType, **kwargs):
        assert data_type is not None
        assert data_type.primitive_type == self.primitive_type()
        self._no_data_value = 1.17549435e-38
        self._entity_type = data_type
        self._association: Optional[DataAssociationEnum] = None
        self._values = None

        if "association" in kwargs.keys():
            setattr(self, "association", kwargs["association"])

        super().__init__(**kwargs)

        data_type.workspace._register_data(self)

    @property
    def no_data_value(self) -> float:
        """
        :obj:`float`: Default no-data-value
        """
        return self._no_data_value

    @property
    def n_values(self) -> Optional[int]:
        """
        :obj:`int`: Number of expected data values based on
        :obj:`~geoh5py.data.data.Data.association`
        """
        if self.association is DataAssociationEnum.VERTEX:
            return self.parent.n_vertices
        if self.association is DataAssociationEnum.CELL:
            return self.parent.n_cells
        if self.association is DataAssociationEnum.FACE:
            return self.parent.n_faces
        if self.association is DataAssociationEnum.OBJECT:
            return 1

        return None

    @property
    def values(self):
        """
        Data values
        """
        return self._values

    @property
    def association(self) -> Optional[DataAssociationEnum]:
        """
        :obj:`~geoh5py.data.data_association_enum.DataAssociationEnum`:
        Relationship made between the
        :func:`~geoh5py.data.data.Data.values` and elements of the
        :obj:`~geoh5py.shared.entity.Entity.parent` object.
        Association can be set from a :obj:`str` chosen from the list of available
        :obj:`~geoh5py.data.data_association_enum.DataAssociationEnum` options.
        """
        return self._association

    @association.setter
    def association(self, value: Union[str, DataAssociationEnum]):
        if isinstance(value, str):

            assert value.upper() in list(
                DataAssociationEnum.__members__.keys()
            ), f"Association flag should be one of {list(DataAssociationEnum.__members__.keys())}"

            self._association = getattr(DataAssociationEnum, value.upper())
        else:
            assert isinstance(
                value, DataAssociationEnum
            ), f"Association must be of type {DataAssociationEnum}"
            self._association = value

    @property
    def entity_type(self) -> DataType:
        """
        :obj:`~geoh5py.data.data_type.DataType`
        """
        return self._entity_type

    @entity_type.setter
    def entity_type(self, data_type: DataType):

        self._entity_type = data_type

        self.modified_attributes = "entity_type"
        return self._entity_type

    @classmethod
    @abstractmethod
    def primitive_type(cls) -> PrimitiveTypeEnum:
        ...

    @classmethod
    def find_or_create_type(
        cls: Type[Entity], workspace: "workspace.Workspace", **kwargs
    ) -> DataType:
        """
        Find or create a type for a given object class

        :param Current workspace: Workspace

        :return: A new or existing object type
        """
        return DataType.find_or_create(workspace, **kwargs)
