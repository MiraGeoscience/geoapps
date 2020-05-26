#  Copyright (c) 2020 Mira Geoscience Ltd.
#
#  This file is part of geoh5py.
#
#  geoh5py is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  geoh5py is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with geoh5py.  If not, see <https://www.gnu.org/licenses/>.

import uuid
from typing import List, Optional

import numpy as np

from .object_base import ObjectBase, ObjectType


class Drillhole(ObjectBase):
    """
    Drillhole object class

    .. warning:: Not yet implemented.

    """

    __TYPE_UID = uuid.UUID(
        fields=(0x7CAEBF0E, 0xD16E, 0x11E3, 0xBC, 0x69, 0xE4632694AA37)
    )

    def __init__(self, object_type: ObjectType, **kwargs):

        # TODO
        self._vertices: Optional[np.ndarray] = None
        self._cells: Optional[np.ndarray] = None
        self._collar: Optional[np.ndarray] = None
        self._surveys: Optional[np.ndarray] = None
        self._trace: List = []

        super().__init__(object_type, **kwargs)
        object_type.workspace._register_object(self)

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID
