#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import sys
import warnings

from geoapps.inversion.driver import start_inversion

if __name__ == "__main__":
    filepath = sys.argv[1]
    warnings.warn(
        "'geoapps.drivers.direct_current_inversion' replaced by "
        "'geoapps.inversion.driver' in version 0.7.0. "
        "This warning is likely due to the execution of older ui.json files. Please update."
    )
    start_inversion(filepath)
