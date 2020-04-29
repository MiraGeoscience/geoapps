import uuid
from typing import List, Optional

from numpy import ndarray

from .object_base import ObjectBase, ObjectType


class Drillhole(ObjectBase):
    """
    Drillhole object class: NOT IMPLEMENTED
    """

    __TYPE_UID = uuid.UUID(
        fields=(0x7CAEBF0E, 0xD16E, 0x11E3, 0xBC, 0x69, 0xE4632694AA37)
    )

    def __init__(self, object_type: ObjectType, **kwargs):

        # TODO
        self._vertices: Optional[ndarray] = None
        self._cells: Optional[ndarray] = None
        self._collar: Optional[ndarray] = None
        self._surveys: Optional[ndarray] = None
        self._trace: List = []

        super().__init__(object_type, **kwargs)
        object_type.workspace._register_object(self)

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID
