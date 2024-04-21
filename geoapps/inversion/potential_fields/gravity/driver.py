# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024 Mira Geoscience Ltd.                                     '
#                                                                              '
#  This file is part of geoapps.                                               '
#                                                                              '
#  geoapps is distributed under the terms and conditions of the MIT License    '
#  (see LICENSE file at the root of this source code package).                 '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from __future__ import annotations

from geoapps.inversion.driver import InversionDriver

from .constants import validations
from .params import GravityParams


class GravityDriver(InversionDriver):
    _params_class = GravityParams
    _validations = validations

    def __init__(self, params: GravityParams):
        super().__init__(params)
