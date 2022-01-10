#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


from geoh5py.workspace import Workspace

from geoapps.create.contours import ContourValues
from geoapps.create.isosurface import IsoSurface
from geoapps.create.surface_2d import Surface2D
from geoapps.export import Export
from geoapps.processing import (
    Calculator,
    Clustering,
    CoordinateTransformation,
    DataInterpolation,
    EdgeDetectionApp,
)

# import pytest
# pytest.skip("eliminating conflicting test.", allow_module_level=True)

project = "./FlinFlon.geoh5"

geoh5 = Workspace(project)

project_dcip = "./FlinFlon_dcip.geoh5"


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


def test_iso_surface():
    app = IsoSurface(h5file=project)
    app.trigger.click()
