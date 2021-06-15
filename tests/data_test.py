#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


import numpy as np
from geoh5py.objects import Grid2D, Points

from geoapps.drivers.components import InversionData, InversionMesh, get_topography
from geoapps.io import InputFile
from geoapps.io.MVI import MVIParams
from geoapps.io.MVI.constants import default_ui_json

input_file = InputFile()
input_file.default(default_ui_json)
input_file.data["geoh5"] = "./FlinFlon.geoh5"
params = MVIParams.from_input_file(input_file)
params.mesh = "{e334f687-df71-4538-ad28-264e420210b8}"
params.data_object = "{538a7eb1-2218-4bec-98cc-0a759aa0ef4f}"
params.tmi_channel = "{44822654-b6ae-45b0-8886-2d845f80f422}"
params.topography_object = "{ab3c2083-6ea8-4d31-9230-7aad3ec09525}"
params.topography = "{a603a762-f6cb-4b21-afda-3160e725bf7d}"
ws = params.workspace
mesh = InversionMesh(params, ws)
window = params.window()
topo = get_topography(ws, params, mesh, window)


def test_get_locs():

    locs = np.ones((10, 3), dtype=float)
    points_object = Points.create(ws, name="test-data", vertices=locs)
    data = InversionData(ws, params, mesh, topo, window)
    params.data_object = points_object.uid
    dlocs = data.get_locs(ws, params)
    np.testing.assert_allclose(locs, dlocs)

    xg, yg = np.meshgrid(np.arange(5) + 0.5, np.arange(5) + 0.5)
    locs = np.c_[xg.ravel(), yg.ravel(), np.zeros(25)]
    grid_object = Grid2D.create(
        ws,
        origin=[0, 0, 0],
        u_cell_size=1,
        v_cell_size=1,
        u_count=5,
        v_count=5,
        rotation=0.0,
        dip=0.0,
    )
    params.data_object = grid_object.uid
    dlocs = data.get_locs(ws, params)
    np.testing.assert_allclose(dlocs, locs)


def test_filter_locs():
    data = InversionData(ws, params, mesh, topo, window)
    data.mesh.rotation["angle"] = 0
    data.filter = None
    xg, yg = np.meshgrid(np.arange(5) + 0.5, np.arange(5) + 0.5)
    locs = np.c_[xg.ravel(), yg.ravel(), np.zeros(25)]
    test_window = {"center": [2.5, 2.5], "size": [1.0, 1.0]}
    tlocs = data.filter_locs(locs, window=test_window)
    assert np.all((tlocs[:, 0] < 3.5) & (tlocs[:, 1] < 3.5))
    assert np.all((tlocs[:, 0] < 3.5) & (tlocs[:, 1] < 3.5))

    data.filter = None
    tlocs = data.filter_locs(locs, resolution=1)
    assert len(tlocs) < len(locs)
