from .data_type import DataType
from .float_data import FloatData
from .primitive_type_enum import PrimitiveTypeEnum
from .reference_value_map import ReferenceValueMap


class ReferencedData(FloatData):
    def __init__(self, data_type: DataType, **kwargs):
        super().__init__(data_type, **kwargs)

        self._value_map = ReferenceValueMap()

    @classmethod
    def primitive_type(cls) -> PrimitiveTypeEnum:
        return PrimitiveTypeEnum.REFERENCED
