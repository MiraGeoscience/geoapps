import tempfile
from pathlib import Path
import pytest
from geoapps.processing import (
    PeakFinder,
    Calculator,
    CoordinateTransformation,
    ContourValues,
    Surface2D,
    Clustering,
    DataInterpolation,
    EdgeDetectionApp,
)
from geoapps.export import Export
from geoapps.inversion import InversionApp
from shutil import copyfile

project = "Project_work.geoh5"


def test_calculator():
    copyfile(r"..\assets\FlinFlon.geoh5", project)
    app = Calculator(h5file=project)
    app.trigger.click()
    pass


def test_coordinate_transformation():
    app = CoordinateTransformation(h5file=project)
    app.trigger.click()
    pass


def test_contour_values():
    app = ContourValues(h5file=project)
    app.trigger.click()
    pass


def test_create_surface():
    app = Surface2D(h5file=project)
    app.trigger.click()
    pass


def test_clustering():
    app = Clustering(h5file=project)
    app.trigger.click()
    pass


def test_data_interpolation():
    app = DataInterpolation(h5file=project)
    app.trigger.click()
    pass


def test_edge_detection():
    app = EdgeDetectionApp(h5file=project)
    app.trigger.click()
    pass


def test_export():
    app = Export(h5file=project)
    app.trigger.click()
    pass


def test_inversion():
    app = InversionApp(h5file=project, inversion_parameters={"max_iterations": 1},)
    app.write.value = True
    app.run.value = True
    pass


def test_peak_finder():
    app = PeakFinder(h5file=project)
    app.run_all.click()
    app.trigger.click()
    pass
