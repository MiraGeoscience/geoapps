#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


import sys

from geoapps.io import InputFile
from geoapps.io.Gravity import GravityParams

from .base_inversion import InversionDriver


def start_inversion(filepath=None):
    """Starts inversion with parameters defined in input file."""

    input_file = InputFile(filepath)
    params = GravityParams(input_file)
    driver = GravityDriver(params)
    driver.run()


class GravityDriver(InversionDriver):
    def __init__(self, params: GravityParams):
        super().__init__(params)

    def run(self):
        super().run()


if __name__ == "__main__":

    filepath = sys.argv[1]
    start_inversion(filepath)
