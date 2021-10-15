#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


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
from geoapps.io.MagneticVector import MagneticVectorParams
from geoapps.io.MagneticVector.constants import default_ui_json
from geoapps.utils import rotate_xy
from geoapps.utils.testing import Geoh5Tester

workspace = Workspace("./FlinFlon.geoh5")


def setup_params(path):

    geotest = Geoh5Tester(
        workspace, path, "test.geoh5", default_ui_json, MagneticVectorParams
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

    return geotest.make()


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
    cc = inversion_mesh.mesh.cell_centers[0].reshape(1, 3)
    point_object = Points.create(ws, name=f"test_point", vertices=cc)
    point_object.add_data({"test_data": {"values": np.array([3.0])}})
    data_object = ws.get_entity("test_data")[0]
    params.associations[data_object.uid] = point_object.uid
    params.lower_bound_object = point_object.uid
    params.lower_bound = data_object.uid
    lower_bound = InversionModel(ws, params, inversion_mesh, "lower_bound")
    assert np.all((lower_bound.model - 3) < 1e-10)
    assert len(lower_bound.model) == 3 * inversion_mesh.nC


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
    octree_mesh = ws.get_entity(params.mesh)[0]
    locs_perm = octree_mesh.centroids[lb_perm[: octree_mesh.n_cells] == 1, :]
    origin = [float(octree_mesh.origin[k]) for k in ["x", "y", "z"]]
    locs_perm_rot = rotate_xy(locs_perm, origin, -octree_mesh.rotation)
    assert xmin <= locs_perm_rot[:, 0].min()
    assert xmax >= locs_perm_rot[:, 0].max()
    assert ymin <= locs_perm_rot[:, 1].min()
    assert ymax >= locs_perm_rot[:, 1].max()
    assert zmin <= locs_perm_rot[:, 2].min()
    assert zmax >= locs_perm_rot[:, 2].max()


def test_permute_2_treemesh(tmp_path):

    ws, params = setup_params(tmp_path)
    octree_mesh = ws.get_entity(params.mesh)[0]
    cc = octree_mesh.centroids
    center = np.mean(cc, axis=0)
    dx = octree_mesh.u_cell_size.min()
    dy = octree_mesh.v_cell_size.min()
    dz = np.abs(octree_mesh.w_cell_size.min())
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
    model = np.zeros(3 * octree_mesh.n_cells, dtype=float)
    model[np.tile(ind, 3)] = 1
    octree_mesh.add_data({"test_model": {"values": model}})
    params.upper_bound = ws.get_entity("test_model")[0].uid
    params.associations[params.upper_bound] = octree_mesh.uid
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
