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
