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

import numpy as np


class Coord3D:
    """
    Coordinate of vertices.

    .. warning:: Replaced by :obj:`numpy.array`

    """

    def __init__(self, xyz: np.ndarray = np.empty((1, 3))):
        self._xyz = xyz

    @property
    def x(self) -> float:
        return self._xyz[:, 0]

    @property
    def y(self) -> float:
        return self._xyz[:, 1]

    @property
    def z(self) -> float:
        return self._xyz[:, 2]

    @property
    def locations(self) -> np.ndarray:
        return self._xyz

    def __getitem__(self, item) -> float:
        return self._xyz[item, :]

    def __call__(self):
        return self._xyz
