#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from dask.distributed import Client

from geoapps.pf_inversion_app import InversionApp

# from geoapps.plotting import ScatterPlots
# from geoapps.processing.peak_finder import PeakFinder


def run():
    #
    app = InversionApp()
    app.inversion_type.value = "gravity"
    app.write.click()
    app.trigger.click()

    # client = Client()
    # app = PeakFinder()
    # app.tem_checkbox.values = False
    # app.data.value = ["Sf[12]"]
    # app.widget


# from geoapps.processing import ContourValues
#
# app = ContourValues()
# app.widget

if __name__ == "__main__":
    run()
