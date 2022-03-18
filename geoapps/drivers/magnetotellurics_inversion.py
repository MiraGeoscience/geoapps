#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import sys
import warnings

from geoapps.drivers.inversion.magnetotellurics import start_inversion

if __name__ == "__main__":
    filepath = sys.argv[1]
    warnings.warn(
        "'geoapps.drivers.magnetotellurics_inversion' moved to "
        "'geoapps.drivers.magnetotellurics.inversion' in version 0.2.0. "
        "This warning is likely due to the execution of older ui.json files. Please update."
    )
    start_inversion(filepath)
