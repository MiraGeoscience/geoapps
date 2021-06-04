#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from copy import deepcopy

import numpy as np
from geoh5py.objects import Points
from geoh5py.workspace import Workspace

from geoapps.drivers.components import InversionMesh, InversionModel
from geoapps.io.MVI import MVIParams

ws = Workspace("../assets/FlinFlon.geoh5")
params = MVIParams.from_path("../assets/mvi_inversion_driver.ui.json")


def test_initialize():
    inversion_mesh = InversionMesh(params, ws)
    starting_model = InversionModel(inversion_mesh, "starting", params, ws)
    assert len(starting_model.model) == 3 * inversion_mesh.nC
    assert len(np.unique(starting_model.model)) == 3


def test_model_from_object():
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
