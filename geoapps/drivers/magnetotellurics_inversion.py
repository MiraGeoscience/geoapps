#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import sys

from geoapps.drivers.base_inversion import InversionDriver
from geoapps.io import InputFile
from geoapps.io.magnetotellurics import MagnetotelluricsParams


def start_inversion(filepath=None, warmstart=True, **kwargs):
    """Starts inversion with parameters defined in input file."""

    if filepath is None:
        input_file = InputFile.from_dict(kwargs)
    else:
        input_file = InputFile(filepath)

    params = MagnetotelluricsParams(input_file)
    driver = MagnetotelluricsDriver(params, warmstart=warmstart)
    driver.run()


class MagnetotelluricsDriver(InversionDriver):
    def __init__(self, params: MagnetotelluricsParams, warmstart=True):
        super().__init__(params, warmstart=warmstart)

    def run(self):
        super().run()


if __name__ == "__main__":
    filepath = sys.argv[1]
    # filepath = r"C:\Users\dominiquef\Desktop\lblock_inversion_v7.ui.json"
    start_inversion(filepath)
