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
    inversion_mesh = InversionMesh(params, ws)
    assert isinstance(inversion_mesh.mesh, TreeMesh)
    assert inversion_mesh.rotation["angle"] == 20


def test_original_cc():
    inversion_mesh = InversionMesh(params, ws)
    msh = ws.get_entity(params.mesh)[0]
    assert np.testing.assert_array_equal(msh.centroids, inversion_mesh.original_cc())
