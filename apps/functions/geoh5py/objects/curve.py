import uuid
from typing import Optional

from numpy import arange, c_, ndarray, unique, zeros

from .object_base import ObjectType
from .points import Points


class Curve(Points):
    """
    A Curve object is defined by a series of cells (segments) connecting a set of
    vertices. Data can be associated to both the cells and vertices.
    """

    __TYPE_UID = uuid.UUID(
        fields=(0x6A057FDC, 0xB355, 0x11E3, 0x95, 0xBE, 0xFD84A7FFCB88)
    )

    def __init__(self, object_type: ObjectType, **kwargs):

        self._cells: Optional[ndarray] = None
        self._line_id: Optional[ndarray] = None
        super().__init__(object_type, **kwargs)

        if object_type.name == "None":
            self.entity_type.name = "Curve"

    @property
    def cells(self) -> Optional[ndarray]:
        """
        Array of indices defining the connection between vertices:
        numpy.ndarray of int, shape ("*", 2)
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
    def cells(self, indices):
        assert indices.dtype == "uint32", "Indices array must be of type 'uint32'"
        self.modified_attributes = "cells"
        self._cells = indices
        self._line_id = None

    @property
    def line_id(self):
        """
        Connections of cells forming unique line_id:
        list of numpy.ndarray of int, shape ("*", 2)
        """
        if getattr(self, "_line_id", None) is None and self.cells is not None:

            cells = self.cells
            line_id = zeros(self.cells.shape[0], dtype="int")
            count = 0
            for ind in range(1, cells.shape[0]):

                if cells[ind, 0] != cells[ind - 1, 1]:
                    count += 1

                line_id[ind] = count

            self._line_id = line_id

        return self._line_id

    @property
    def unique_lines(self):
        """
        Unique lines connected by cells
        """
        if self.line_id is not None:

            return unique(self.line_id).tolist()

        return None

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        """
        :return: Default unique identifier
        """
        return cls.__TYPE_UID


class SurveyAirborneMagnetics(Curve):
    """
    An airborne magnetic survey object
    """

    __TYPE_UID = uuid.UUID(
        fields=(0x4B99204C, 0xD133, 0x4579, 0xA9, 0x16, 0xA9C8B98CFCCB)
    )

    def __init__(self, object_type: ObjectType, **kwargs):
        super().__init__(object_type, **kwargs)

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        """
        :return: Default unique identifier
        """
        return cls.__TYPE_UID
