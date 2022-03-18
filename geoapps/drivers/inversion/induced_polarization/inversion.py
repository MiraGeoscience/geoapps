#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


import sys

from geoh5py.ui_json import InputFile

from geoapps.drivers.inversion.inversion import InversionDriver

from .params import InducedPolarizationParams


def start_inversion(filepath=None, **kwargs):
    """Starts inversion with parameters defined in input file."""

    input_file = None
    if filepath is not None:
        input_file = InputFile.read_ui_json(filepath)

    params = InducedPolarizationParams(input_file=input_file, **kwargs)
    driver = InducedPolarizationDriver(params)
    driver.run()


class InducedPolarizationDriver(InversionDriver):
    def __init__(self, params: InducedPolarizationParams):
        super().__init__(params)

    def run(self):
        super().run()


if __name__ == "__main__":

    filepath = sys.argv[1]
    start_inversion(filepath)
