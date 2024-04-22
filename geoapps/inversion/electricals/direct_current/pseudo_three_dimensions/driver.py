# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024 Mira Geoscience Ltd.                                     '
#                                                                              '
#  This file is part of geoapps.                                               '
#                                                                              '
#  geoapps is distributed under the terms and conditions of the MIT License    '
#  (see LICENSE file at the root of this source code package).                 '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from __future__ import annotations

from geoapps.inversion.electricals.direct_current.pseudo_three_dimensions.constants import (
    validations,
)
from geoapps.inversion.electricals.direct_current.pseudo_three_dimensions.params import (
    DirectCurrentPseudo3DParams,
)
from geoapps.inversion.electricals.direct_current.two_dimensions.params import (
    DirectCurrent2DParams,
)
from geoapps.inversion.electricals.driver import BasePseudo3DDriver


class DirectCurrentPseudo3DDriver(BasePseudo3DDriver):
    _params_class = DirectCurrentPseudo3DParams
    _params_2d_class = DirectCurrent2DParams
    _validations = validations
