#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


from copy import deepcopy

import numpy as np
from geoh5py.objects import Points
from geoh5py.workspace import Workspace

from geoapps.drivers.components import (
    InversionData,
    InversionMesh,
    InversionModel,
    InversionModelCollection,
    InversionTopography,
    InversionWindow,
)
from geoapps.drivers.magnetic_vector import MagneticVectorParams, default_ui_json
from geoapps.utils import rotate_xy
from geoapps.utils.testing import Geoh5Tester

geoh5 = Workspace("./FlinFlon.geoh5")


def setup_params(path):

    geotest = Geoh5Tester(
        geoh5, path, "test.geoh5", deepcopy(default_ui_json), MagneticVectorParams
    )
    geotest.set_param("data_object", "{538a7eb1-2218-4bec-98cc-0a759aa0ef4f}")
    geotest.set_param("tmi_channel_bool", True)
    geotest.set_param("tmi_channel", "{44822654-b6ae-45b0-8886-2d845f80f422}")
    geotest.set_param("window_center_x", 314183.0)
    geotest.set_param("window_center_y", 6071014.0)
    geotest.set_param("window_width", 1000.0)
    geotest.set_param("window_height", 1000.0)
    geotest.set_param("out_group", "MVIInversion")
    geotest.set_param("mesh", "{e334f687-df71-4538-ad28-264e420210b8}")
    geotest.set_param("topography_object", "{ab3c2083-6ea8-4d31-9230-7aad3ec09525}")
    geotest.set_param("topography", "{a603a762-f6cb-4b21-afda-3160e725bf7d}")
    geotest.set_param("starting_model", 1e-04)
    geotest.set_param("inducing_field_inclination", 79.0)
    geotest.set_param("inducing_field_declination", 11.0)
    geotest.set_param("reference_model", 0.0)
    geotest.set_param("reference_inclination", 79.0)
    geotest.set_param("reference_declination", 11.0)

    return geotest.make()


def test_zero_reference_model(tmp_path):
    ws, params = setup_params(tmp_path)
    inversion_window = InversionWindow(ws, params)
    inversion_data = InversionData(ws, params, inversion_window.window)
    inversion_topography = InversionTopography(ws, params, inversion_window.window)
    inversion_mesh = InversionMesh(ws, params, inversion_data, inversion_topography)
    model = InversionModel(ws, params, inversion_mesh, "reference")
    incl = np.unique(ws.get_entity("reference_inclination")[0].values)
    decl = np.unique(ws.get_entity("reference_declination")[0].values)
    assert len(incl) == 1
    assert len(decl) == 1
    assert np.isclose(incl[0], 79.0)
    assert np.isclose(decl[0], 11.0)


def test_collection(tmp_path):
    ws, params = setup_params(tmp_path)
    inversion_window = InversionWindow(ws, params)
    inversion_data = InversionData(ws, params, inversion_window.window)
    inversion_topography = InversionTopography(ws, params, inversion_window.window)
    inversion_mesh = InversionMesh(ws, params, inversion_data, inversion_topography)
    active_cells = inversion_topography.active_cells(inversion_mesh)
    models = InversionModelCollection(ws, params, inversion_mesh)
    models.remove_air(active_cells)
    starting = InversionModel(ws, params, inversion_mesh, "starting")
    starting.remove_air(active_cells)
    np.testing.assert_allclose(models.starting, starting.model)


def test_initialize(tmp_path):

    ws, params = setup_params(tmp_path)
    inversion_window = InversionWindow(ws, params)
    inversion_data = InversionData(ws, params, inversion_window.window)
    inversion_topography = InversionTopography(ws, params, inversion_window.window)
    inversion_mesh = InversionMesh(ws, params, inversion_data, inversion_topography)
    starting_model = InversionModel(ws, params, inversion_mesh, "starting")
    assert len(starting_model.model) == 3 * inversion_mesh.nC
    assert len(np.unique(starting_model.model)) == 3


def test_model_from_object(tmp_path):
    # Test behaviour when loading model from Points object with non-matching mesh
    ws, params = setup_params(tmp_path)
    inversion_window = InversionWindow(ws, params)
    inversion_data = InversionData(ws, params, inversion_window.window)
    inversion_topography = InversionTopography(ws, params, inversion_window.window)
    inversion_mesh = InversionMesh(ws, params, inversion_data, inversion_topography)
    cc = inversion_mesh.mesh.cell_centers
    m0 = np.array([2.0, 3.0, 1.0])
    vals = (m0[0] * cc[:, 0]) + (m0[1] * cc[:, 1]) + (m0[2] * cc[:, 2])

    point_object = Points.create(ws, name=f"test_point", vertices=cc)
    point_object.add_data({"test_data": {"values": vals}})
    data_object = ws.get_entity("test_data")[0]
    params.lower_bound_object = point_object.uid
    params.lower_bound = data_object.uid
    lower_bound = InversionModel(ws, params, inversion_mesh, "lower_bound")
    nc = int(len(lower_bound.model) / 3)
    A = lower_bound.mesh.mesh.cell_centers
    b = lower_bound.model[:nc]
    from scipy.linalg import lstsq

    m = lstsq(A, b)[0]
    np.testing.assert_array_almost_equal(m, m0, decimal=1)


def test_permute_2_octree(tmp_path):

    ws, params = setup_params(tmp_path)
    params.lower_bound = 0.0
    inversion_window = InversionWindow(ws, params)
    inversion_data = InversionData(ws, params, inversion_window.window)
    inversion_topography = InversionTopography(ws, params, inversion_window.window)
    inversion_mesh = InversionMesh(ws, params, inversion_data, inversion_topography)
    lower_bound = InversionModel(ws, params, inversion_mesh, "lower_bound")
    cc = inversion_mesh.mesh.cell_centers
    center = np.mean(cc, axis=0)
    dx = inversion_mesh.mesh.h[0].min()
    dy = inversion_mesh.mesh.h[1].min()
    dz = inversion_mesh.mesh.h[2].min()
    xmin = center[0] - (5 * dx)
    xmax = center[0] + (5 * dx)
    ymin = center[1] - (5 * dy)
    ymax = center[1] + (5 * dy)
    zmin = center[2] - (5 * dz)
    zmax = center[2] + (5 * dz)
    xind = (cc[:, 0] > xmin) & (cc[:, 0] < xmax)
    yind = (cc[:, 1] > ymin) & (cc[:, 1] < ymax)
    zind = (cc[:, 2] > zmin) & (cc[:, 2] < zmax)
    ind = xind & yind & zind
    lower_bound.model[np.tile(ind, 3)] = 1
    lb_perm = lower_bound.permute_2_octree()

    locs_perm = params.mesh.centroids[lb_perm[: params.mesh.n_cells] == 1, :]
    origin = [float(params.mesh.origin[k]) for k in ["x", "y", "z"]]
    locs_perm_rot = rotate_xy(locs_perm, origin, -params.mesh.rotation)
    assert xmin <= locs_perm_rot[:, 0].min()
    assert xmax >= locs_perm_rot[:, 0].max()
    assert ymin <= locs_perm_rot[:, 1].min()
    assert ymax >= locs_perm_rot[:, 1].max()
    assert zmin <= locs_perm_rot[:, 2].min()
    assert zmax >= locs_perm_rot[:, 2].max()


def test_permute_2_treemesh(tmp_path):

    ws, params = setup_params(tmp_path)
    cc = params.mesh.centroids
    center = np.mean(cc, axis=0)
    dx = params.mesh.u_cell_size.min()
    dy = params.mesh.v_cell_size.min()
    dz = np.abs(params.mesh.w_cell_size.min())
    xmin = center[0] - (5 * dx)
    xmax = center[0] + (5 * dx)
    ymin = center[1] - (5 * dy)
    ymax = center[1] + (5 * dy)
    zmin = center[2] - (5 * dz)
    zmax = center[2] + (5 * dz)
    xind = (cc[:, 0] > xmin) & (cc[:, 0] < xmax)
    yind = (cc[:, 1] > ymin) & (cc[:, 1] < ymax)
    zind = (cc[:, 2] > zmin) & (cc[:, 2] < zmax)
    ind = xind & yind & zind
    model = np.zeros(params.mesh.n_cells, dtype=float)
    model[ind] = 1
    params.mesh.add_data({"test_model": {"values": model}})
    params.upper_bound = ws.get_entity("test_model")[0].uid

    inversion_window = InversionWindow(ws, params)
    inversion_data = InversionData(ws, params, inversion_window.window)
    inversion_topography = InversionTopography(ws, params, inversion_window.window)
    inversion_mesh = InversionMesh(ws, params, inversion_data, inversion_topography)
    upper_bound = InversionModel(ws, params, inversion_mesh, "upper_bound")
    locs = inversion_mesh.mesh.cell_centers
    locs_rot = rotate_xy(
        locs, inversion_mesh.rotation["origin"], inversion_mesh.rotation["angle"]
    )
    locs_rot = locs_rot[upper_bound.model[: inversion_mesh.mesh.nC] == 1, :]
    assert xmin <= locs_rot[:, 0].min()
    assert xmax >= locs_rot[:, 0].max()
    assert ymin <= locs_rot[:, 1].min()
    assert ymax >= locs_rot[:, 1].max()
    assert zmin <= locs_rot[:, 2].min()
    assert zmax >= locs_rot[:, 2].max()
