#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import sys

from dask.distributed import Client, LocalCluster

from geoapps.io.MVI import MVIParams

from .base_inversion import InversionDriver


def start_inversion(filepath=None):
    """ Starts inversion with parameters defined in input file. """

    cluster = LocalCluster(processes=False)
    client = Client(cluster)

    params = MVIParams.from_path(filepath)
    driver = MVIDriver(params)
    driver.run()


class MVIDriver(InversionDriver):
    def __init__(self, params: MVIParams):
        super().__init__(params)

    def run(self):
        super().run()


if __name__ == "__main__":

    filepath = sys.argv[1]
    start_inversion(filepath)
