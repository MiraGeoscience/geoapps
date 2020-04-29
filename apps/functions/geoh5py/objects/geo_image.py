import uuid

from .object_base import ObjectBase, ObjectType


class GeoImage(ObjectBase):
    """
    Image object class: NOT IMPLEMENTED
    """

    __TYPE_UID = uuid.UUID(
        fields=(0x77AC043C, 0xFE8D, 0x4D14, 0x81, 0x67, 0x75E300FB835A)
    )

    def __init__(self, object_type: ObjectType, **kwargs):

        # TODO
        self._vertices = None
        super().__init__(object_type, **kwargs)

        object_type.workspace._register_object(self)

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID
