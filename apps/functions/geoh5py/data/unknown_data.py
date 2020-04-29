import uuid

from .data import Data
from .data_association_enum import DataAssociationEnum
from .data_type import DataType
from .primitive_type_enum import PrimitiveTypeEnum


class UnknownData(Data):
    def __init__(
        self,
        data_type: DataType,
        association: DataAssociationEnum,
        name: str,
        uid: uuid.UUID = None,
    ):
        super().__init__(data_type, association=association, name=name, uid=uid)

    @classmethod
    def primitive_type(cls) -> PrimitiveTypeEnum:
        return PrimitiveTypeEnum.INVALID

    # TODO: Provide a partial implementation to access generic data,
    # for which primitive type would be provided by the H5 file.
    # raise NotImplementedError for method that are not supported
