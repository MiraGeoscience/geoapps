#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from copy import deepcopy

import numpy as np
import pytest
import SimPEG
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


def test_get_uncertainty_component():
    tparams = deepcopy(params)
    tparams.tmi_uncertainty = 1
    data = InversionData(ws, tparams, mesh, topo, window)
    unc = data.get_uncertainty_component("tmi")
    assert len(np.unique(unc)) == 1
    assert np.unique(unc)[0] == 1
    assert len(unc) == len(data.mask)

    tparams.tmi_uncertainty = None
    data = InversionData(ws, tparams, mesh, topo, window)
    unc = data.get_uncertainty_component("tmi")
    assert len(np.unique(unc)) == 1
    assert np.unique(unc)[0] == 1
    assert len(unc) == len(data.mask)


def test_parse_ignore_values():
    tparams = deepcopy(params)
    tparams.ignore_values = "<99"
    data = InversionData(ws, tparams, mesh, topo, window)
    val, type = data.parse_ignore_values()
    assert val == 99
    assert type == "<"

    tparams.ignore_values = ">99"
    data = InversionData(ws, tparams, mesh, topo, window)
    val, type = data.parse_ignore_values()
    assert val == 99
    assert type == ">"

    tparams.ignore_values = "99"
    data = InversionData(ws, tparams, mesh, topo, window)
    val, type = data.parse_ignore_values()
    assert val == 99
    assert type == "="


def test_set_infinity_uncertainties():
    data = InversionData(ws, params, mesh, topo, window)
    test_data = np.array([0, 1, 2, 3, 4, 5])
    test_unc = np.array([0.1] * 6)
    data.ignore_value = 3
    data.ignore_type = "="
    unc = data.set_infinity_uncertainties(test_unc, test_data)
    where_inf = np.where(np.isinf(unc))[0]
    assert len(where_inf) == 1
    assert where_inf == 3

    data.ignore_value = 3
    data.ignore_type = "<"
    unc = data.set_infinity_uncertainties(test_unc, test_data)
    where_inf = np.where(np.isinf(unc))[0]
    assert len(where_inf) == 4
    assert np.all(where_inf == [0, 1, 2, 3])

    data.ignore_value = 3
    data.ignore_type = ">"
    unc = data.set_infinity_uncertainties(test_unc, test_data)
    where_inf = np.where(np.isinf(unc))[0]
    assert len(where_inf) == 3
    assert np.all(where_inf == [3, 4, 5])

    data.ignore_value = None
    data.ignore_type = None
    unc = data.set_infinity_uncertainties(test_unc, test_data)
    assert np.all(test_unc == unc)


def test_get_locs():
    tparams = deepcopy(params)
    locs = np.ones((10, 3), dtype=float)
    points_object = Points.create(ws, name="test-data", vertices=locs)
    data = InversionData(ws, tparams, mesh, topo, window)
    tparams.data_object = points_object.uid
    dlocs = data.get_locs()
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
    tparams.data_object = grid_object.uid
    dlocs = data.get_locs()
    np.testing.assert_allclose(dlocs, locs)


def test_mask():
    data = InversionData(ws, params, mesh, topo, window)
    test_mask = [0, 1, 1, 0]
    data.mask = test_mask
    assert isinstance(data.mask, np.ndarray)
    assert data.mask.dtype == bool
    test_mask = [0, 1, 2, 3]
    with pytest.raises(ValueError) as excinfo:
        data.mask = test_mask
    assert "Badly formed" in str(excinfo.value)


def test_filter():
    data = InversionData(ws, params, mesh, topo, window)
    test_data = np.array([0, 1, 2, 3, 4, 5])
    data.mask = np.array([0, 0, 1, 1, 1, 0])
    filtered_data = data.filter(test_data)
    assert np.all(filtered_data == [2, 3, 4])

    test_data = {"key": test_data}
    filtered_data = data.filter(test_data)
    assert np.all(filtered_data["key"] == [2, 3, 4])


def test_rotate():
    # Basic test since rotate_xy already tested
    data = InversionData(ws, params, mesh, topo, window)
    test_locs = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    locs_rot = data.rotate(test_locs)
    assert locs_rot.shape == test_locs.shape


def test_displace():
    data = InversionData(ws, params, mesh, topo, window)
    test_locs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    test_offset = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    expected_locs = np.array([[2.0, 2.0, 3.0], [5.0, 5.0, 6.0], [8.0, 8.0, 9.0]])
    displaced_locs = data.displace(test_locs, test_offset)
    assert np.all(displaced_locs == expected_locs)

    test_offset = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    expected_locs = np.array([[1.0, 3.0, 3.0], [4.0, 6.0, 6.0], [7.0, 9.0, 9.0]])
    displaced_locs = data.displace(test_locs, test_offset)
    assert np.all(displaced_locs == expected_locs)

    test_offset = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    expected_locs = np.array([[1.0, 2.0, 4.0], [4.0, 5.0, 7.0], [7.0, 8.0, 10.0]])
    displaced_locs = data.displace(test_locs, test_offset)
    assert np.all(displaced_locs == expected_locs)


def test_drape():

    data = InversionData(ws, params, mesh, topo, window)

    # create radar object with z channel an set uid to data.radar
    test_locs = np.array([[1.0, 2.0, 1.0], [2.0, 1.0, 1.0], [8.0, 9.0, 1.0]])
    radar_ch = np.array([1.0, 2.0, 3.0])
    drape_object = Points.create(ws, name="test_drape", vertices=test_locs)
    test_drape = drape_object.add_data({"z": {"values": radar_ch}})
    data.radar = test_drape.uid

    # create topography and set data.topo
    xg, yg = np.meshgrid(np.linspace(0, 10, 11), np.linspace(0, 10, 11))
    x = xg.ravel()
    y = yg.ravel()
    z = np.ones(x.shape)
    z[(x > 5) & (y > 5)] = 2
    data.topo = np.c_[x, y, z]

    expected_locs = np.array([[1.0, 2.0, 2.0], [2.0, 1.0, 3.0], [8.0, 9.0, 5.0]])
    draped_locs = data.drape(test_locs)

    assert np.all(draped_locs == expected_locs)


def test_normalize():
    data = InversionData(ws, params, mesh, topo, window)
    data.data = {"tmi": np.array([1.0, 2.0, 3.0]), "gz": np.array([1.0, 2.0, 3.0])}
    data.components = list(data.data.keys())
    test_data = data.normalize(data.data)
    assert np.all(data.normalization == [1, -1])
    assert np.all(test_data["gz"] == (-1 * data.data["gz"]))


def test_get_survey():
    data = InversionData(ws, params, mesh, topo, window)
    survey = data.get_survey()
    assert isinstance(survey, SimPEG.potential_fields.magnetics.Survey)
