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


class DataAssociationEnum(Enum):
    """
    Known data association between :obj:`~geoh5py.data.data.Data.values` and
    the :obj:`~geoh5py.shared.entity.Entity.parent` object.
    Available options:
    """

    UNKNOWN = 0
    OBJECT = 1
    CELL = 2
    VERTEX = 3
    FACE = 4
    GROUP = 5
