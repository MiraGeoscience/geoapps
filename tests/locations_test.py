#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import numpy as np
import pytest
from geoh5py.objects import Grid2D, Points
from geoh5py.workspace import Workspace

from geoapps.drivers.components import InversionMesh
from geoapps.drivers.components.locations import InversionLocations
from geoapps.io.MVI import MVIParams, default_ui_json
from geoapps.utils.testing import Geoh5Tester

workspace = Workspace("./FlinFlon.geoh5")


def setup_params(tmp):
    geotest = Geoh5Tester(workspace, tmp, "test.geoh5", default_ui_json, MVIParams)
    geotest.set_param("mesh", "{e334f687-df71-4538-ad28-264e420210b8}")
    return geotest.make()


def test_mask(tmp_path):
    ws, params = setup_params(tmp_path)
    mesh = InversionMesh(ws, params)
    window = params.window()
    locations = InversionLocations(ws, params, mesh, window)
    test_mask = [0, 1, 1, 0]
    locations.mask = test_mask
    assert isinstance(locations.mask, np.ndarray)
    assert locations.mask.dtype == bool
    test_mask = [0, 1, 2, 3]
    with pytest.raises(ValueError) as excinfo:
        locations.mask = test_mask
    assert "Badly formed" in str(excinfo.value)


def test_get_locs(tmp_path):
    ws, params = setup_params(tmp_path)
    mesh = InversionMesh(ws, params)
    window = params.window()
    locs = np.ones((10, 3), dtype=float)
    points_object = Points.create(ws, name="test-data", vertices=locs)
    locations = InversionLocations(ws, params, mesh, window)
    dlocs = locations.get_locs(points_object.uid)
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
    dlocs = locations.get_locs(grid_object.uid)
    np.testing.assert_allclose(dlocs, locs)


def test_filter(tmp_path):
    ws, params = setup_params(tmp_path)
    mesh = InversionMesh(ws, params)
    window = params.window()
    locations = InversionLocations(ws, params, mesh, window)
    test_data = np.array([0, 1, 2, 3, 4, 5])
    locations.mask = np.array([0, 0, 1, 1, 1, 0])
    filtered_data = locations.filter(test_data)
    assert np.all(filtered_data == [2, 3, 4])

    test_data = {"key": test_data}
    filtered_data = locations.filter(test_data)
    assert np.all(filtered_data["key"] == [2, 3, 4])


def test_rotate(tmp_path):
    # Basic test since rotate_xy already tested
    ws, params = setup_params(tmp_path)
    mesh = InversionMesh(ws, params)
    window = params.window()
    locations = InversionLocations(ws, params, mesh, window)
    test_locs = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    locs_rot = locations.rotate(test_locs)
    assert locs_rot.shape == test_locs.shape
