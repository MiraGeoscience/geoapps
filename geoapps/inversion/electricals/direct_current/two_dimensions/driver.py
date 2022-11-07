#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import sys

from geoapps.inversion.driver import InversionDriver

from .constants import validations
from .params import DirectCurrent2DParams


class DirectCurrent2DDriver(InversionDriver):

    _params_class = DirectCurrent2DParams
    _validations = validations

    def __init__(self, params: DirectCurrent2DParams):
        super().__init__(params)


if __name__ == "__main__":

    file = sys.argv[1]
    DirectCurrent2DDriver.start(file)
