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

from typing import Dict, Optional


class ReferenceValue:
    """ Represents a value for ReferencedData as a string.
    """

    def __init__(self, value: str = None):
        self._value = value

    @property
    def value(self) -> Optional[str]:
        return self._value

    def __str__(self):
        # TODO: representation for None?
        return str(self._value)


class ReferenceValueMap:
    """ Maps from reference index to reference value of ReferencedData.
    """

    def __init__(self, color_map: Dict[int, ReferenceValue] = None):
        self._map = dict() if color_map is None else color_map

    def __getitem__(self, item: int) -> ReferenceValue:
        return self._map[item]

    def __len__(self):
        return len(self._map)
