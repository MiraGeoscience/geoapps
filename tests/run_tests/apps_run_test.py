#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from uuid import UUID

from geoh5py.workspace import Workspace
from ipywidgets import Widget

from geoapps.create.contours import ContourValues
from geoapps.create.isosurface import IsoSurface
from geoapps.create.surface_2d import Surface2D
from geoapps.drivers.magnetic_vector_inversion import MagneticVectorParams
from geoapps.export import Export
from geoapps.pf_inversion_app import InversionApp
from geoapps.processing import (
    Calculator,
    Clustering,
    CoordinateTransformation,
    DataInterpolation,
    EdgeDetectionApp,
)

project = "./FlinFlon.geoh5"
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
    params = {
        "w_cell_size": 60,
        "z_from_topo": False,
        "forward_only": True,
        "starting_model": 0.01,
        "topography": None,
        "receivers_radar_drape": UUID("{6de9177a-8277-4e17-b76c-2b8b05dcf23c}"),
    }

    changes = {
        "inducing_field_inclination": 35,
        "detrend_data": True,
    }

    side_effects = {"starting_inclination": 35, "detrend_type": "all"}

    app = InversionApp(h5file=project, plot_result=False, **params)
    app.monitoring_directory = str(tmp_path)

    for param, value in changes.items():
        if isinstance(getattr(app, param), Widget):
            getattr(app, param).value = value
        else:
            setattr(app, param, value)

    app.write.click()

    params_reload = MagneticVectorParams.from_path(app.params.input_file.filepath)

    for param, value in params.items():
        assert (
            getattr(params_reload, param) == value
        ), f"Parameter {param} not saved and loaded correctly."

    for param, value in side_effects.items():
        assert (
            getattr(params_reload, param) == value
        ), f"Side effect parameter {param} not saved and loaded correctly."


def test_iso_surface():
    app = IsoSurface(h5file=project)
    app.trigger.click()
