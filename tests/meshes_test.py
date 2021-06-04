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

ws = Workspace("./FlinFlon.geoh5")
uipath = "../geoapps/drivers/example_ui_json/mvi_inversion_driver.ui.json"
params = MVIParams.from_path(uipath)


def test_initialize():
    inversion_mesh = InversionMesh(params, ws)
    assert isinstance(inversion_mesh.mesh, TreeMesh)
    assert inversion_mesh.rotation["angle"] == 20


def test_original_cc():
    inversion_mesh = InversionMesh(params, ws)
    msh = inversion_mesh.fetch("mesh")
    assert np.all(msh.centroids == inversion_mesh.original_cc())
