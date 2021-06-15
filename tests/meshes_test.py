#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import numpy as np
from discretize import TreeMesh

from geoapps.drivers.components import InversionMesh
from geoapps.io import InputFile
from geoapps.io.MVI import MVIParams
from geoapps.io.MVI.constants import default_ui_json

input_file = InputFile()
input_file.default(default_ui_json)
input_file.data["geoh5"] = "./FlinFlon.geoh5"
params = MVIParams.from_input_file(input_file)
params.mesh = "{e334f687-df71-4538-ad28-264e420210b8}"
ws = params.workspace


def test_initialize():
    mesh = InversionMesh(params, ws)
    assert isinstance(mesh.mesh, TreeMesh)
    assert mesh.rotation["angle"] == 20


def test_original_cc():
    mesh = InversionMesh(params, ws)
    msh = ws.get_entity(params.mesh)[0]
    np.testing.assert_allclose(msh.centroids, mesh.original_cc())
