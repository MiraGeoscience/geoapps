#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


import pytest
import requests

from geoapps.create.contours import ContourValues
from geoapps.create.isosurface import IsoSurface
from geoapps.create.surface_2d import Surface2D
from geoapps.export import Export
from geoapps.inversion import InversionApp
from geoapps.processing import (
    Calculator,
    Clustering,
    CoordinateTransformation,
    DataInterpolation,
    EdgeDetectionApp,
    PeakFinder,
)

project = "FlinFlon.geoh5"


def test_calculator():
    url = "https://github.com/MiraGeoscience/geoapps/raw/main/assets/FlinFlon.geoh5"

    r = requests.get(url)
    open(project, "wb").write(r.content)

    app = Calculator(h5file=project)
    app.trigger.click()


def test_coordinate_transformation():
    app = CoordinateTransformation(h5file=project)
    app.trigger.click()


def test_contour_values():
    app = ContourValues(h5file=project)
    app.trigger.click()


def test_create_surface():
    app = Surface2D(h5file=project)
    app.trigger.click()


def test_clustering():
    app = Clustering(h5file=project)
    app.trigger.click()


def test_data_interpolation():
    app = DataInterpolation(h5file=project)
    app.trigger.click()


def test_edge_detection():
    app = EdgeDetectionApp(h5file=project)
    app.trigger.click()


def test_export():
    app = Export(h5file=project)
    app.trigger.click()


def test_inversion():
    app = InversionApp(
        h5file=project,
        inversion_parameters={"max_iterations": 1},
    )
    app.write.value = True
    app.run.value = True


def test_peak_finder():
    app = PeakFinder(h5file=project)
    app.run_all.click()
    app.trigger.click()


def test_iso_surface():
    app = IsoSurface(h5file=project)
    app.trigger.click()
