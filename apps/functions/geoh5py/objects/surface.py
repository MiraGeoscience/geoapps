import uuid
from typing import Optional

from numpy import arange, c_, ndarray

from .object_base import ObjectType
from .points import Points


class Surface(Points):
    """
    Surface object defined by vertices and cells
    """

    __TYPE_UID = uuid.UUID(
        fields=(0xF26FEBA3, 0xADED, 0x494B, 0xB9, 0xE9, 0xB2BBCBE298E1)
    )

    def __init__(self, object_type: ObjectType, **kwargs):

        self._cells: Optional[ndarray] = None

        super().__init__(object_type, **kwargs)

        if object_type.name == "None":
            self.entity_type.name = "Surface"

    @property
    def cells(self) -> Optional[ndarray]:
        """
        Array of vertices index forming triangles
        :return cells: array of int, shape ("*", 3)
        """
        if getattr(self, "_cells", None) is None:
            if self.existing_h5_entity:
                self._cells = self.workspace.fetch_cells(self.uid)
            else:
                if self.vertices is not None:
                    n_segments = self.vertices.shape[0]
                    self._cells = c_[
                        arange(0, n_segments - 1), arange(1, n_segments)
                    ].astype("uint32")

        return self._cells

    @cells.setter
    def cells(self, indices: ndarray):
        """
        :param indices: array of int, shape ("*", 3)
        """
        assert indices.dtype in [
            "int32",
            "uint32",
        ], "Indices array must be of type 'uint32'"

        if indices.dtype == "int32":
            indices.astype("uint32")
        self.modified_attributes = "cells"
        self._cells = indices

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID
