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

from .data import Data, PrimitiveTypeEnum


class IntegerData(Data):
    @classmethod
    def primitive_type(cls) -> PrimitiveTypeEnum:
        return PrimitiveTypeEnum.INTEGER

    @property
    def values(self) -> np.ndarray:
        """
        :return: values: An array of :obj:`int` values
        """
        if (getattr(self, "_values", None) is None) and self.existing_h5_entity:
            self._values = self.workspace.fetch_values(self.uid)

        return self._values

    @values.setter
    def values(self, values):
        self.modified_attributes = "values"
        self._values = np.ravel(values).astype(int)

    def __call__(self):
        return self.values
