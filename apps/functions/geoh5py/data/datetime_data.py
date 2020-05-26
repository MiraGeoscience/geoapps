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

from .data import Data, PrimitiveTypeEnum


class DatetimeData(Data):
    @classmethod
    def primitive_type(cls) -> PrimitiveTypeEnum:
        return PrimitiveTypeEnum.DATETIME

    # TODO: implement specialization to access values.
    # Stored as a 1D array of variable-length strings formatted according to the ISO 8601
    # extended specification for representations of UTC dates and times (Qt implementation),
    # taking the form YYYY-MM-DDTHH:mm:ss[Z|[+|-]HH:mm]
    # No data value : empty string
