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
from geoapps.drivers.magnetic_vector_inversion import (
    MagneticVectorDriver,
    MagneticVectorParams,
)
from geoapps.export import Export
from geoapps.pf_inversion_app import InversionApp
from geoapps.processing import (
    Calculator,
    Clustering,
    CoordinateTransformation,
    DataInterpolation,
    EdgeDetectionApp,
)
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

    params = {
        "w_cell_size": 60,
        "z_from_topo": False,
        "forward_only": True,
        "starting_model": 0.01,
        "receivers_radar_drape": UUID("{6de9177a-8277-4e17-b76c-2b8b05dcf23c}"),
    }

    changes = {
        "inducing_field_inclination": 35,
        "detrend_data": True,
    }

    side_effects = {"starting_inclination": 35, "detrend_type": "all"}

    app = InversionApp(h5file=geotest.ws.h5file, plot_result=False, **params)

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

    driver = MagneticVectorDriver(params_reload)
    driver.run()


def test_iso_surface():
    app = IsoSurface(h5file=project)
    app.trigger.click()
