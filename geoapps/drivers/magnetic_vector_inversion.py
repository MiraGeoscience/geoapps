#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import sys

from geoapps.drivers.base_inversion import InversionDriver
from geoapps.io import InputFile
from geoapps.io.MagneticVector import MagneticVectorParams


def start_inversion(filepath=None, **kwargs):
    """Starts inversion with parameters defined in input file."""

    if filepath is None:
        input_file = InputFile.from_dict(kwargs)
    else:
        input_file = InputFile(filepath)

    params = MagneticVectorParams(input_file)
    driver = MagneticVectorDriver(params)
    driver.run()


class MagneticVectorDriver(InversionDriver):
    def __init__(self, params: MagneticVectorParams):
        super().__init__(params)

    def run(self):
        super().run()


if __name__ == "__main__":
    filepath = sys.argv[1]
    start_inversion(filepath)
