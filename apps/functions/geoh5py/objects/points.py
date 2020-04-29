import uuid
from typing import Optional

from numpy import ndarray

from ..shared import Coord3D
from .object_base import ObjectBase, ObjectType


class Points(ObjectBase):
    """
    Points object
    """

    __TYPE_UID = uuid.UUID("{202C5DB1-A56D-4004-9CAD-BAAFD8899406}")

    def __init__(self, object_type: ObjectType, **kwargs):
        self._vertices: Optional[Coord3D] = None

        super().__init__(object_type, **kwargs)

        if object_type.name == "None":
            self.entity_type.name = "Points"

        object_type.workspace._register_object(self)

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID

    @property
    def vertices(self) -> Optional[ndarray]:
        """
        Array of vertices coordinates, shape ("*", 3)
        """
        if (getattr(self, "_vertices", None) is None) and self.existing_h5_entity:
            self._vertices = self.workspace.fetch_vertices(self.uid)

        return self._vertices

    @vertices.setter
    def vertices(self, xyz: ndarray):
        self.modified_attributes = "vertices"
        self._vertices = xyz
