#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


from uuid import UUID

from geoh5py.workspace import Workspace

from geoapps.create.contours import ContourValues
from geoapps.create.isosurface import IsoSurface
from geoapps.create.surface_2d import Surface2D
from geoapps.export import Export
from geoapps.pf_inversion_app import InversionApp
from geoapps.processing import (
    Calculator,
    Clustering,
    CoordinateTransformation,
    DataInterpolation,
    EdgeDetectionApp,
)
from geoapps.processing.peak_finder import PeakFinder
from geoapps.utils.testing import Geoh5Tester

project = "FlinFlon.geoh5"
workspace = Workspace(project)


def test_calculator():
    app = Calculator(h5file=project)
    app.trigger.click()


def test_coordinate_transformation():
    app = CoordinateTransformation(h5file=project)
    app.trigger.click()


def test_contour_values():
    app = ContourValues(h5file=project, plot_result=False)
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
    app = EdgeDetectionApp(h5file=project, plot_result=False)
    app.trigger.click()


def test_export():
    app = Export(h5file=project)
    app.trigger.click()


def test_inversion(tmp_path):
    test_ws_path = "test.geoh5"
    geotest = Geoh5Tester(workspace, tmp_path, test_ws_path)
    geotest.copy_entity(UUID("{e334f687-df71-4538-ad28-264e420210b8}"))
    geotest.copy_entity(UUID("{ab3c2083-6ea8-4d31-9230-7aad3ec09525}"))
    geotest.copy_entity(UUID("{538a7eb1-2218-4bec-98cc-0a759aa0ef4f}"))
    geotest.copy_entity(UUID("{44822654-b6ae-45b0-8886-2d845f80f422}"))
    geotest.copy_entity(UUID("{a603a762-f6cb-4b21-afda-3160e725bf7d}"))
    app = InversionApp(
        h5file=geotest.ws.h5file,
        inversion_parameters={"max_iterations": 1},
        plot_result=False,
    )
    app.write.value = True
    app.run.value = True


def test_iso_surface():
    app = IsoSurface(h5file=project)
    app.trigger.click()
