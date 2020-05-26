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


class ColorMap:
    """ Records colors assigned to value ranges (where Value is the start of the range).
    """

    _attribute_map = {"File name": "name"}

    def __init__(self, **kwargs):

        self._values = dict()
        self._name = "Unknown"

        for attr, item in kwargs.items():
            try:
                if attr in self._attribute_map.keys():
                    attr = self._attribute_map[attr]
                setattr(self, attr, item)
            except AttributeError:
                continue

    @property
    def values(self) -> np.ndarray:
        """
        :obj:`numpy.array`: Colormap defined by values and corresponding RGBA:

        .. code-block:: python

            values = [
                [V_1, R_1, G_1, B_1, A_1],
                ..., [V_i, R_i, G_i, B_i, A_i]
            ]

        where V (Values) are sorted floats defining the position of each RGBA.
        R (Red), G (Green), B (Blue) and A (Alpha) are integer values between [0, 255].
        """
        return self._values

    @values.setter
    def values(self, values: np.ndarray):
        names = ["Value", "Red", "Green", "Blue", "Alpha"]
        formats = ["<f8", "u1", "u1", "u1", "u1"]

        if isinstance(values.dtype, np.dtype):
            assert all(
                [name in names for name in values.dtype.names]
            ), f"Input 'values' must contain fields with types {names}"
            self._values = np.asarray(values, dtype=list(zip(names, formats)))

        else:
            assert (
                values.shape[1] == 5
            ), "'values' must be a an array of shape (*, 5) for [value, r, g, b, a]"
            self._values = np.core.records.fromarrays(
                values.T, names=names, formats=formats
            )

    @property
    def name(self) -> str:
        """
        :obj:`str`: Name of the colormap
        """
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    def __len__(self):
        return len(self._values)
