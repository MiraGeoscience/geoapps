#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).
import os
import uuid
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


def test_coordinate_transformation(tmp_path):
    temp_workspace = path.join(tmp_path, "contour.geoh5")
    with Workspace(temp_workspace) as workspace:
        geoh5.get_entity("Gravity_Magnetics_drape60m")[0].copy(parent=workspace)
        geoh5.get_entity("Data_TEM_pseudo3D")[0].copy(parent=workspace)

    app = CoordinateTransformation(h5file=temp_workspace)
    app.trigger.click()

    files = os.listdir(path.join(tmp_path, "Temp"))
    with Workspace(path.join(tmp_path, "Temp", files[0])) as workspace:
        assert len(workspace.objects) == 2, "Coordinate transform failed."


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


def test_data_interpolation(tmp_path):
    temp_workspace = path.join(tmp_path, "contour.geoh5")
    with Workspace(temp_workspace) as workspace:
        for uid in [
            "{2e814779-c35f-4da0-ad6a-39a6912361f9}",
            "{f3e36334-be0a-4210-b13e-06933279de25}",
            "{7450be38-1327-4336-a9e4-5cff587b6715}",
            "{ab3c2083-6ea8-4d31-9230-7aad3ec09525}",
        ]:
            geoh5.get_entity(uuid.UUID(uid))[0].copy(parent=workspace)

    app = DataInterpolation(h5file=temp_workspace)
    app.trigger.click()

    files = os.listdir(path.join(tmp_path, "Temp"))
    with Workspace(path.join(tmp_path, "Temp", files[0])) as workspace:
        assert len(workspace.get_entity("Iteration_7_model_Interp")) == 1


def test_edge_detection():
    app = EdgeDetectionApp(h5file=project, plot_result=False)
    app.trigger.click()


def test_export():
    app = Export(h5file=project)
    app.trigger.click()


def test_iso_surface(tmp_path):
    temp_workspace = path.join(tmp_path, "contour.geoh5")
    with Workspace(temp_workspace) as workspace:
        for uid in [
            "{2e814779-c35f-4da0-ad6a-39a6912361f9}",
        ]:
            geoh5.get_entity(uuid.UUID(uid))[0].copy(parent=workspace)

    app = IsoSurface(h5file=temp_workspace)
    app.trigger.click()

    files = os.listdir(path.join(tmp_path, "Temp"))
    with Workspace(path.join(tmp_path, "Temp", files[0])) as workspace:
        group = workspace.get_entity("ISO")[0]
        assert len(group.children) == 5
