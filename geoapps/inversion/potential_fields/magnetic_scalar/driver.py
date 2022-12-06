#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from geoapps.inversion.driver import InversionDriver

from .constants import validations
from .params import MagneticScalarParams


class MagneticScalarDriver(InversionDriver):

    _params_class = MagneticScalarParams
    _validations = validations

    def __init__(self, params: MagneticScalarParams, warmstart=True):
        super().__init__(params, warmstart)
