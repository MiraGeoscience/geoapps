from abc import abstractmethod
from typing import TYPE_CHECKING, Optional, Type

from ..shared import Entity
from .data_association_enum import DataAssociationEnum
from .data_type import DataType
from .primitive_type_enum import PrimitiveTypeEnum

if TYPE_CHECKING:
    from .. import workspace


class Data(Entity):
    """
    Data class
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
        :return: Default no-data-value
        """
        return self._no_data_value

    @property
    def n_values(self) -> Optional[int]:
        """
        :return: Number of expected data values
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
        return self._values

    @property
    def association(self) -> Optional[DataAssociationEnum]:
        return self._association

    @association.setter
    def association(self, value):
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
