import uuid

from .primitive_type_enum import PrimitiveTypeEnum


class GeometricDataConstants:
    __X_TYPE_UID = uuid.UUID(
        fields=(0xE9E6B408, 0x4109, 0x4E42, 0xB6, 0xA8, 0x685C37A802EE)
    )
    __Y_TYPE_UID = uuid.UUID(
        fields=(0xF55B07BD, 0xD8A0, 0x4DFF, 0xBA, 0xE5, 0xC975D490D71C)
    )
    __Z_TYPE_UID = uuid.UUID(
        fields=(0xDBAFB885, 0x1531, 0x410C, 0xB1, 0x8E, 0x6AC9A40B4466)
    )

    @classmethod
    def x_datatype_uid(cls) -> uuid.UUID:
        return cls.__X_TYPE_UID

    @classmethod
    def y_datatype_uid(cls) -> uuid.UUID:
        return cls.__Y_TYPE_UID

    @classmethod
    def z_datatype_uid(cls) -> uuid.UUID:
        return cls.__Z_TYPE_UID

    @classmethod
    def primitive_type(cls) -> PrimitiveTypeEnum:
        return PrimitiveTypeEnum.GEOMETRIC
