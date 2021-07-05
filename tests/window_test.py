#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import numpy as np
from geoh5py.objects import Points
from geoh5py.workspace import Workspace

from geoapps.drivers.components import InversionWindow
from geoapps.io.Gravity import GravityParams, default_ui_json
from geoapps.utils.testing import Geoh5Tester

workspace = Workspace("./FlinFlon.geoh5")


def test_initialize(tmp_path):

    # Test initialize from params
    tester = Geoh5Tester(
        workspace,
        tmp_path,
        "test.geoh5",
        ui=default_ui_json,
        params_class=GravityParams,
    )

    tester.set_param("window_center_x", 50.0)
    tester.set_param("window_center_y", 50.0)
    tester.set_param("window_width", 100.0)
    tester.set_param("window_height", 100.0)
    ws, params = tester.make()

    verts = np.array(
        [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [100.0, 100.0, 0.0]]
    )
    point_object = Points.create(ws, name=f"test-window", vertices=verts)
    params.data_object = point_object.uid

    win = InversionWindow(ws, params)
    assert np.all(win.window["center"] == [50.0, 50.0])
    assert np.all(win.window["size"] == [100.0, 100.0])

    win.window["center"] = [None, None]
    win.window["size"] = [None, None]

    assert win.is_empty()

    # Test initialize from None
    params.window_center_x = None
    params.window_center_y = None
    params.window_width = None
    params.window_height = None

    win = InversionWindow(ws, params)
    assert win.window["center"] == [50.0, 50.0]
    assert win.window["size"] == [100.0, 100.0]
