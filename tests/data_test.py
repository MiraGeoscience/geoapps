#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


import numpy as np
import pytest
import SimPEG
from geoh5py.objects import Grid2D, Points
from geoh5py.workspace import Workspace

from geoapps.drivers.components import InversionData, InversionMesh, InversionTopography
from geoapps.io import InputFile
from geoapps.io.MVI import MVIParams
from geoapps.io.MVI.constants import default_ui_json
from geoapps.utils.testing import Geoh5Tester

workspace = Workspace("./FlinFlon.geoh5")


def setup_params(tmp):
    geotest = Geoh5Tester(workspace, tmp, "test.geoh5", default_ui_json, MVIParams)
    geotest.set_param("mesh", "{e334f687-df71-4538-ad28-264e420210b8}")
    geotest.set_param("data_object", "{538a7eb1-2218-4bec-98cc-0a759aa0ef4f}")
    geotest.set_param("topography_object", "{ab3c2083-6ea8-4d31-9230-7aad3ec09525}")
    geotest.set_param("tmi_channel", "{44822654-b6ae-45b0-8886-2d845f80f422}")
    geotest.set_param("topography", "{a603a762-f6cb-4b21-afda-3160e725bf7d}")
    return geotest.make()


# input_file = InputFile()
# input_file.default(default_ui_json)
# input_file.data["geoh5"] = "./FlinFlon.geoh5"
# params = MVIParams.from_input_file(input_file)
# params.topography = "{a603a762-f6cb-4b21-afda-3160e725bf7d}"
# ws = params.workspace
# mesh = InversionMesh(params, ws)
# window = params.window()
# topo = get_topography(ws, params, mesh, window)


def test_get_uncertainty_component(tmp_path):
    ws, params = setup_params(tmp_path)
    mesh = InversionMesh(ws, params)
    window = params.window()
    topo = InversionTopography(ws, params, window)
    params.tmi_uncertainty = 1
    data = InversionData(ws, params, window)
    unc = data.get_uncertainty_component("tmi")
    assert len(np.unique(unc)) == 1
    assert np.unique(unc)[0] == 1
    assert len(unc) == len(data.mask)

    params.tmi_uncertainty = None
    data = InversionData(ws, params, window)
    unc = data.get_uncertainty_component("tmi")
    assert len(np.unique(unc)) == 1
    assert np.unique(unc)[0] == 1
    assert len(unc) == len(data.mask)


def test_parse_ignore_values(tmp_path):
    ws, params = setup_params(tmp_path)
    mesh = InversionMesh(ws, params)
    window = params.window()
    topo = InversionTopography(ws, params, window)
    params.ignore_values = "<99"
    data = InversionData(ws, params, window)
    val, type = data.parse_ignore_values()
    assert val == 99
    assert type == "<"

    params.ignore_values = ">99"
    data = InversionData(ws, params, window)
    val, type = data.parse_ignore_values()
    assert val == 99
    assert type == ">"

    params.ignore_values = "99"
    data = InversionData(ws, params, window)
    val, type = data.parse_ignore_values()
    assert val == 99
    assert type == "="


def test_set_infinity_uncertainties(tmp_path):
    ws, params = setup_params(tmp_path)
    mesh = InversionMesh(ws, params)
    window = params.window()
    topo = InversionTopography(ws, params, window)
    data = InversionData(ws, params, window)
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


def test_displace(tmp_path):
    ws, params = setup_params(tmp_path)
    mesh = InversionMesh(ws, params)
    window = params.window()
    topo = InversionTopography(ws, params, window)
    data = InversionData(ws, params, window)
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


def test_drape(tmp_path):
    ws, params = setup_params(tmp_path)
    window = params.window()
    topo = InversionTopography(ws, params, window)
    data = InversionData(ws, params, window)

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
    topo.locs = np.c_[x, y, z]

    expected_locs = np.array([[1.0, 2.0, 2.0], [2.0, 1.0, 3.0], [8.0, 9.0, 5.0]])
    draped_locs = data.drape(topo.locs, test_locs)

    assert np.all(draped_locs == expected_locs)


def test_normalize(tmp_path):
    ws, params = setup_params(tmp_path)
    mesh = InversionMesh(ws, params)
    window = params.window()
    topo = InversionTopography(ws, params, window)
    data = InversionData(ws, params, window)
    data.data = {"tmi": np.array([1.0, 2.0, 3.0]), "gz": np.array([1.0, 2.0, 3.0])}
    data.components = list(data.data.keys())
    test_data = data.normalize(data.data)
    assert np.all(data.normalizations == [1, -1])
    assert np.all(test_data["gz"] == (-1 * data.data["gz"]))


def test_get_survey(tmp_path):
    ws, params = setup_params(tmp_path)
    mesh = InversionMesh(ws, params)
    window = params.window()
    topo = InversionTopography(ws, params, window)
    data = InversionData(ws, params, window)
    survey = data.survey()
    assert isinstance(survey, SimPEG.potential_fields.magnetics.Survey)
