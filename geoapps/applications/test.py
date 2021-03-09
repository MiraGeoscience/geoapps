#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from dask.distributed import Client

from geoapps.inversion import InversionApp
from geoapps.plotting import ScatterPlots
from geoapps.processing import ContourValues, PeakFinder


def run():
    #
    # app = InversionApp()
    # app.widget

    client = Client()
    app = PeakFinder()
    app.tem_checkbox.values = False
    app.data.value = ["Sf[12]"]
    app.widget


# from geoapps.processing import ContourValues
#
# app = ContourValues()
# app.widget

if __name__ == "__main__":
    run()
