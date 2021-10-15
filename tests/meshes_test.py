#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import numpy as np
import pytest
from discretize import TreeMesh
from geoh5py.workspace import Workspace

from geoapps.drivers.components import InversionData, InversionMesh
from geoapps.io.MagneticVector import MagneticVectorParams
from geoapps.io.MagneticVector.constants import default_ui_json
from geoapps.utils.testing import Geoh5Tester

workspace = Workspace("./FlinFlon.geoh5")


def setup_params(tmp):
    geotest = Geoh5Tester(
        workspace, tmp, "test.geoh5", default_ui_json, MagneticVectorParams
    )
    geotest.set_param("mesh", "{e334f687-df71-4538-ad28-264e420210b8}")
    geotest.set_param("data_object", "{538a7eb1-2218-4bec-98cc-0a759aa0ef4f}")
    geotest.set_param("topography_object", "{ab3c2083-6ea8-4d31-9230-7aad3ec09525}")
    geotest.set_param("tmi_channel_bool", True)
    geotest.set_param("tmi_channel", "{44822654-b6ae-45b0-8886-2d845f80f422}")
    geotest.set_param("topography", "{a603a762-f6cb-4b21-afda-3160e725bf7d}")
    geotest.set_param("out_group", "MVIInversion")
    return geotest.make()


def test_initialize(tmp_path):

    ws, params = setup_params(tmp_path)
    inversion_mesh = InversionMesh(ws, params)
    assert isinstance(inversion_mesh.mesh, TreeMesh)
    assert inversion_mesh.rotation["angle"] == 20


def test_collect_mesh_params(tmp_path):
    ws, params = setup_params(tmp_path)
    locs = ws.get_entity(params.data_object)[0].centroids
    window = {"center": [np.mean(locs[:, 0]), np.mean(locs[:, 1])], "size": [100, 100]}
    data = InversionData(ws, params, window)
    inversion_mesh = InversionMesh(ws, params)
    octree_params = inversion_mesh.collect_mesh_params(params)
    assert "Refinement A" in octree_params.free_params_dict.keys()
    with pytest.raises(ValueError) as excinfo:
        params.u_cell_size = None
        octree_params = inversion_mesh.collect_mesh_params(params)
    assert "Cannot create OctreeParams" in str(excinfo.value)


def test_mesh_from_params(tmp_path):
    ws, params = setup_params(tmp_path)
    locs = ws.get_entity(params.data_object)[0].centroids
    window = {"center": [np.mean(locs[:, 0]), np.mean(locs[:, 1])], "size": [100, 100]}
    data = InversionData(ws, params, window)
    params.mesh_from_params = True
    params.mesh = None
    params.u_cell_size, params.v_cell_size, params.w_cell_size = 19, 25, 25
    inversion_mesh = InversionMesh(ws, params)
    assert all(inversion_mesh.mesh.h[0] == 19)
