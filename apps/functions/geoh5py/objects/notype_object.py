import uuid

from .object_base import ObjectBase
from .object_type import ObjectType


class NoTypeObject(ObjectBase):
    """
    Generic Data object without a registered type
    """

    __TYPE_UID = uuid.UUID(
        fields=(0x849D2F3E, 0xA46E, 0x11E3, 0xB4, 0x01, 0x2776BDF4F982)
    )

    def __init__(self, object_type: ObjectType, **kwargs):
        super().__init__(object_type, **kwargs)

        if object_type.name == "None":
            self.entity_type.name = "Unknown"

        object_type.workspace._register_object(self)

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID
