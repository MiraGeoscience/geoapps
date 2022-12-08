#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from geoapps.inversion.line_sweep.driver import LineSweepDriver

from .constants import validations
from .params import DirectCurrentPseudo3DParams


class DirectCurrentPseudo3DDriver(LineSweepDriver):

    _params_class = DirectCurrentPseudo3DParams
    _validations = validations

    def __init__(self, params: DirectCurrentPseudo3DParams):  # pylint: disable=W0235
        super().__init__(params)
        lookup = self.get_lookup()
        self.write_files(lookup)
