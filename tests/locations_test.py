#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from copy import deepcopy

import numpy as np
import pytest
from geoh5py.objects import Grid2D, Points
from geoh5py.workspace import Workspace

from geoapps.drivers.components import InversionMesh
from geoapps.drivers.components.locations import InversionLocations
from geoapps.io.MagneticVector import MagneticVectorParams, default_ui_json
from geoapps.utils.testing import Geoh5Tester

geoh5 = Workspace("./FlinFlon.geoh5")


def setup_params(tmp):
    geotest = Geoh5Tester(
        geoh5, tmp, "test.geoh5", deepcopy(default_ui_json), MagneticVectorParams
    )
    geotest.set_param("mesh", "{e334f687-df71-4538-ad28-264e420210b8}")
    geotest.set_param("topography_object", "{ab3c2083-6ea8-4d31-9230-7aad3ec09525}")
    return geotest.make()


def test_mask(tmp_path):
    ws, params = setup_params(tmp_path)
    window = params.window()
    locations = InversionLocations(ws, params, window)
    test_mask = [0, 1, 1, 0]
    locations.mask = test_mask
    assert isinstance(locations.mask, np.ndarray)
    assert locations.mask.dtype == bool
    test_mask = [0, 1, 2, 3]
    with pytest.raises(ValueError) as excinfo:
        locations.mask = test_mask
    assert "Badly formed" in str(excinfo.value)


def test_get_locations(tmp_path):
    ws, params = setup_params(tmp_path)
    window = params.window()
    locs = np.ones((10, 3), dtype=float)
    points_object = Points.create(ws, name="test-data", vertices=locs)
    locations = InversionLocations(ws, params, window)
    dlocs = locations.get_locations(points_object.uid)
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
    dlocs = locations.get_locations(grid_object.uid)
    np.testing.assert_allclose(dlocs, locs)


def test_filter(tmp_path):
    ws, params = setup_params(tmp_path)
    window = params.window()
    locations = InversionLocations(ws, params, window)
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
    window = params.window()
    locations = InversionLocations(ws, params, window)
    test_locs = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    locs_rot = locations.rotate(test_locs)
    assert locs_rot.shape == test_locs.shape


def test_z_from_topo(tmp_path):
    ws, params = setup_params(tmp_path)
    window = params.window()
    locations = InversionLocations(ws, params, window)
    locs = locations.set_z_from_topo(np.array([[315674, 6070832, 0]]))
    assert locs[0, 2] == 326

    params.topography = 320.0
    locations = InversionLocations(ws, params, window)
    locs = locations.set_z_from_topo(np.array([[315674, 6070832, 0]]))
    assert locs[0, 2] == 320.0
