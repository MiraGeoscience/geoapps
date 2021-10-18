#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from os import path
from uuid import UUID

from geoh5py.workspace import Workspace
from ipywidgets import Widget

from geoapps.create.contours import ContourValues
from geoapps.create.isosurface import IsoSurface
from geoapps.create.surface_2d import Surface2D
from geoapps.drivers.direct_current_inversion import DirectCurrentParams
from geoapps.drivers.magnetic_vector_inversion import MagneticVectorParams
from geoapps.export import Export
from geoapps.inversion.dcip_inversion_app import InversionApp as DCInversionApp
from geoapps.inversion.pf_inversion_app import InversionApp as MagInversionApp
from geoapps.processing import (
    Calculator,
    Clustering,
    CoordinateTransformation,
    DataInterpolation,
    EdgeDetectionApp,
)

project = "./FlinFlon.geoh5"

workspace = Workspace(project)

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


def test_mag_inversion(tmp_path):
    """Tests the jupyter application for mag-mvi"""
    ws = Workspace(project)
    new_workspace = Workspace(path.join(tmp_path, "invtest.geoh5"))

    new_topo = ws.get_entity(UUID("ab3c2083-6ea8-4d31-9230-7aad3ec09525"))[0].copy(
        parent=new_workspace
    )
    new_obj = ws.get_entity(UUID("{538a7eb1-2218-4bec-98cc-0a759aa0ef4f}"))[0].copy(
        parent=new_workspace
    )
    topo_val = new_topo.add_data({"elev": {"values": new_topo.vertices[:, 2]}})
    changes = {
        "data_object": new_obj.uid,
        "tmi_channel": UUID("{44822654-b6ae-45b0-8886-2d845f80f422}"),
        "inducing_field_inclination": 35,
        "detrend_data": True,
        "topography_object": new_topo.uid,
        "topography": topo_val.uid,
        "z_from_topo": False,
        "forward_only": False,
        "starting_model": 0.01,
    }
    side_effects = {"starting_inclination": 35, "detrend_type": "all"}
    app = MagInversionApp(h5file=project, plot_result=False)
    app.geoh5 = new_workspace

    for param, value in changes.items():
        if isinstance(getattr(app, param), Widget):
            getattr(app, param).value = value
        else:
            setattr(app, param, value)

    app.write.click()
    params_reload = MagneticVectorParams.from_path(app.params.input_file.filepath)

    for param, value in changes.items():
        assert (
            getattr(params_reload, param) == value
        ), f"Parameter {param} not saved and loaded correctly."

    for param, value in side_effects.items():
        assert (
            getattr(params_reload, param) == value
        ), f"Side effect parameter {param} not saved and loaded correctly."


def test_dc_inversion(tmp_path):
    """Tests the jupyter application for dc inversion"""
    ws = Workspace(project_dcip)
    new_workspace = Workspace(path.join(tmp_path, "invtest.geoh5"))
    new_topo = ws.get_entity(UUID("{ab3c2083-6ea8-4d31-9230-7aad3ec09525}"))[0].copy(
        parent=new_workspace
    )
    # dc object
    currents = ws.get_entity(UUID("{c2403ce5-ccfd-4d2f-9ffd-3867154cb871}"))[0]
    currents.copy(parent=new_workspace)
    changes = {
        "topography_object": new_topo.uid,
        "z_from_topo": False,
        "forward_only": False,
        "starting_model": 0.01,
    }
    side_effects = {}
    app = DCInversionApp(h5file=project_dcip, plot_result=False)
    app.geoh5 = new_workspace

    for param, value in changes.items():
        if isinstance(getattr(app, param), Widget):
            getattr(app, param).value = value
        else:
            setattr(app, param, value)

    app.write.click()
    params_reload = DirectCurrentParams.from_path(app.params.input_file.filepath)

    for param, value in changes.items():
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
