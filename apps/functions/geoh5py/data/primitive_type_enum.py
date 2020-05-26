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

from enum import Enum


class PrimitiveTypeEnum(Enum):
    """
    Known data type.

    Available options:
    """

    INVALID = 0
    INTEGER = 1
    FLOAT = 2
    TEXT = 3
    REFERENCED = 4
    FILENAME = 5
    BLOB = 6
    VECTOR = 7
    DATETIME = 8
    GEOMETRIC = 9
