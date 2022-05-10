#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).
import os
from os import path

from geoh5py.workspace import Workspace

from geoapps.calculator import Calculator
from geoapps.clustering import Clustering
from geoapps.contours.application import ContourValues
from geoapps.coordinate_transformation import CoordinateTransformation
from geoapps.edge_detection import EdgeDetectionApp
from geoapps.export.application import Export
from geoapps.interpolation import DataInterpolation
from geoapps.iso_surfaces.application import IsoSurface
from geoapps.triangulated_surfaces.application import Surface2D

# import pytest
# pytest.skip("eliminating conflicting test.", allow_module_level=True)

project = "./FlinFlon.geoh5"

geoh5 = Workspace(project)

project_dcip = "./FlinFlon_dcip.geoh5"


def test_calculator(tmp_path):
    temp_workspace = path.join(tmp_path, "contour.geoh5")
    with Workspace(temp_workspace) as workspace:
        geoh5.get_entity("geochem")[0].copy(parent=workspace)

    app = Calculator(h5file=temp_workspace)
    app.trigger.click()

    files = os.listdir(path.join(tmp_path, "Temp"))
    with Workspace(path.join(tmp_path, "Temp", files[0])) as workspace:
        output = workspace.get_entity("NewChannel")[0]
        assert output.n_vertices == 2740, "Change in output. Need to verify."


def test_coordinate_transformation():
    app = CoordinateTransformation(h5file=project)
    app.trigger.click()


def test_contour_values(tmp_path):
    temp_workspace = path.join(tmp_path, "contour.geoh5")
    with Workspace(temp_workspace) as workspace:
        geoh5.get_entity("Gravity_Magnetics_drape60m")[0].copy(parent=workspace)

    app = ContourValues(h5file=temp_workspace, plot_result=False)
    app.trigger.click()

    files = os.listdir(path.join(tmp_path, "Temp"))
    with Workspace(path.join(tmp_path, "Temp", files[0])) as workspace:
        output = workspace.get_entity("Airborne_TMI")[0]
        assert output.n_vertices == 2740, "Change in output. Need to verify."


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
