#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from copy import deepcopy

import numpy as np
from geoh5py.objects import Points

from geoapps.drivers.components import InversionMesh, InversionModel
from geoapps.io import InputFile
from geoapps.io.MVI import MVIParams
from geoapps.io.MVI.constants import default_ui_json
from geoapps.utils import rotate_xy

input_file = InputFile()
input_file.default(default_ui_json)
input_file.data["geoh5"] = "./FlinFlon.geoh5"
params = MVIParams.from_ifile(input_file)
params.mesh = "{e334f687-df71-4538-ad28-264e420210b8}"
params.starting_model = 1e-04
params.inducing_field_inclination = 79.0
params.inducing_field_declination = 11.0
ws = params.workspace


def test_initialize():
    inversion_mesh = InversionMesh(params, ws)
    starting_model = InversionModel(inversion_mesh, "starting", params, ws)
    assert len(starting_model.model) == 3 * inversion_mesh.nC
    assert len(np.unique(starting_model.model)) == 3


def test_model_from_object():
    # Test behaviour when loading model from Points object with non-matching mesh
    p = deepcopy(params)
    inversion_mesh = InversionMesh(params, ws)
    cc = inversion_mesh.mesh.cell_centers[0].reshape(1, 3)
    point_object = Points.create(ws, name=f"test_point", vertices=cc)
    point_object.add_data({"test_data": {"values": np.array([3])}})
    data_object = ws.get_entity("test_data")[0]
    params.associations[data_object.uid] = point_object.uid
    params.lower_bound_object = point_object.uid
    params.lower_bound = data_object.uid
    lower_bound = InversionModel(inversion_mesh, "lower_bound", params, ws)
    assert np.all((lower_bound.model - 3) < 1e-10)
    assert len(lower_bound.model) == inversion_mesh.nC


def test_permute_2_octree():

    params.lower_bound = 0.0
    inversion_mesh = InversionMesh(params, ws)
    lower_bound = InversionModel(inversion_mesh, "lower_bound", params, ws)
    cc = inversion_mesh.mesh.cell_centers
    center = np.mean(cc, axis=0)
    dx = inversion_mesh.mesh.hx.min()
    dy = inversion_mesh.mesh.hy.min()
    dz = inversion_mesh.mesh.hz.min()
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
    lower_bound.model[ind] = 1
    lb_perm = lower_bound.permute_2_octree()
    octree_mesh = ws.get_entity(params.mesh)[0]
    locs_perm = octree_mesh.centroids[lb_perm == 1, :]
    origin = [float(octree_mesh.origin[k]) for k in ["x", "y", "z"]]
    locs_perm_rot = rotate_xy(locs_perm, origin, -octree_mesh.rotation)
    assert xmin <= locs_perm_rot[:, 0].min()
    assert xmax >= locs_perm_rot[:, 0].max()
    assert ymin <= locs_perm_rot[:, 1].min()
    assert ymax >= locs_perm_rot[:, 1].max()
    assert zmin <= locs_perm_rot[:, 2].min()
    assert zmax >= locs_perm_rot[:, 2].max()


def test_permute_2_treemesh():
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
    model = np.zeros(octree_mesh.n_cells)
    model[ind] = 1
    octree_mesh.add_data({"test_model": {"values": model}})
    params.upper_bound = ws.get_entity("test_model")[0].uid
    params.associations[params.upper_bound] = octree_mesh.uid
    inversion_mesh = InversionMesh(params, ws)
    upper_bound = InversionModel(inversion_mesh, "upper_bound", params, ws)
    locs = inversion_mesh.mesh.cell_centers
    locs_rot = rotate_xy(
        locs, inversion_mesh.rotation["origin"], inversion_mesh.rotation["angle"]
    )
    locs_rot = locs_rot[upper_bound.model == 1, :]
    assert xmin <= locs_rot[:, 0].min()
    assert xmax >= locs_rot[:, 0].max()
    assert ymin <= locs_rot[:, 1].min()
    assert ymax >= locs_rot[:, 1].max()
    assert zmin <= locs_rot[:, 2].min()
    assert zmax >= locs_rot[:, 2].max()
