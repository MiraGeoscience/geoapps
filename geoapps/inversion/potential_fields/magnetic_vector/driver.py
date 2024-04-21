# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024 Mira Geoscience Ltd.                                     '
#                                                                              '
#  This file is part of geoapps.                                               '
#                                                                              '
#  geoapps is distributed under the terms and conditions of the MIT License    '
#  (see LICENSE file at the root of this source code package).                 '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from __future__ import annotations

from SimPEG import maps

from geoapps.inversion.driver import InversionDriver

from .constants import validations
from .params import MagneticVectorParams


class MagneticVectorDriver(InversionDriver):
    _params_class = MagneticVectorParams
    _validations = validations

    def __init__(self, params: MagneticVectorParams):
        super().__init__(params)

    @property
    def mapping(self) -> list[maps.Projection] | None:
        """Model mapping for the inversion."""
        if self._mapping is None:
            mapping = []
            start = 0
            for _ in range(3):
                mapping.append(
                    maps.Projection(
                        self.n_values * 3, slice(start, start + self.n_values)
                    )
                )
                start += self.n_values

            self._mapping = mapping

        return self._mapping

    @mapping.setter
    def mapping(self, value: list[maps.Projection]):
        if not isinstance(value, list) or len(value) != 3:
            raise TypeError(
                "'mapping' must be a list of 3 instances of maps.IdentityMap. "
                f"Provided {value}"
            )

        if not all(
            isinstance(val, maps.Projection)
            and val.shape == (self.n_values, 3 * self.n_values)
            for val in value
        ):
            raise TypeError(
                "'mapping' must be an instance of maps.Projection with shape (n_values, 3 * self.n_values). "
                f"Provided {value}"
            )

        self._mapping = value
