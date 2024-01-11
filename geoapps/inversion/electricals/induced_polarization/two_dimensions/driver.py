#  Copyright (c) 2024 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from geoapps.inversion.driver import InversionDriver

from .constants import validations
from .params import InducedPolarization2DParams


class InducedPolarization2DDriver(InversionDriver):
    _params_class = InducedPolarization2DParams
    _validations = validations

    def __init__(self, params: InducedPolarization2DParams):
        super().__init__(params)
