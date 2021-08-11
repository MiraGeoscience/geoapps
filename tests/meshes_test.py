#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import numpy as np
from discretize import TreeMesh
from geoh5py.workspace import Workspace

from geoapps.drivers.components import InversionMesh
from geoapps.io.MVI import MVIParams
from geoapps.io.MVI.constants import default_ui_json
from geoapps.utils.testing import Geoh5Tester

workspace = Workspace("./FlinFlon.geoh5")


def setup_params(tmp):
    geotest = Geoh5Tester(workspace, tmp, "test.geoh5", default_ui_json, MVIParams)
    geotest.set_param("mesh", "{e334f687-df71-4538-ad28-264e420210b8}")
    return geotest.make()


def test_initialize(tmp_path):

    ws, params = setup_params(tmp_path)
    inversion_mesh = InversionMesh(ws, params)
    assert isinstance(inversion_mesh.mesh, TreeMesh)
    assert inversion_mesh.rotation["angle"] == 20


def test_original_cc(tmp_path):

    ws, params = setup_params(tmp_path)
    inversion_mesh = InversionMesh(ws, params)
    msh = ws.get_entity(params.mesh)[0]
    np.testing.assert_allclose(msh.centroids, inversion_mesh.original_cc())


def test_collect_mesh_params(tmp_path):
    ws, params = setup_params(tmp_path)
    inversion_mesh = InversionMesh(ws, params)
    inversion_mesh.collect_mesh_params(params)


# def test_mesh_from_params(tmp_path):
#     ws, params = setup_params(tmp_path)
#     inversion_mesh = InversionMesh(ws, params)
