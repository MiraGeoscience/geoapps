#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import sys

from geoapps.io import InputFile
from geoapps.io.magnetotellurics import MagnetotelluricsParams

from .base_inversion import InversionDriver


def start_inversion(filepath=None, **kwargs):
    """Starts inversion with parameters defined in input file."""

    if filepath is None:
        input_file = InputFile.from_dict(kwargs)
    else:
        input_file = InputFile(filepath)

    params = MagnetotelluricsParams(input_file)
    driver = MagnetotelluricsDriver(params)
    driver.run()


class MagnetotelluricsDriver(InversionDriver):
    def __init__(self, params: MagnetotelluricsParams):
        super().__init__(params)

    def run(self):
        super().run()


if __name__ == "__main__":

    filepath = sys.argv[1]
    start_inversion(filepath)
