#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


import numpy as np
import SimPEG
from geoh5py.workspace import Workspace

from geoapps.drivers.components import InversionData
from geoapps.io.MagneticVector import MagneticVectorParams
from geoapps.io.MagneticVector.constants import default_ui_json
from geoapps.utils.testing import Geoh5Tester

workspace = Workspace("./FlinFlon.geoh5")


def setup_params(tmp):
    geotest = Geoh5Tester(
        workspace, tmp, "test.geoh5", default_ui_json, MagneticVectorParams
    )
    geotest.set_param("mesh", "{e334f687-df71-4538-ad28-264e420210b8}")
    geotest.set_param("data_object", "{538a7eb1-2218-4bec-98cc-0a759aa0ef4f}")
    geotest.set_param("topography_object", "{ab3c2083-6ea8-4d31-9230-7aad3ec09525}")
    geotest.set_param("tmi_channel_bool", True)
    geotest.set_param("tmi_channel", "{44822654-b6ae-45b0-8886-2d845f80f422}")
    geotest.set_param("topography", "{a603a762-f6cb-4b21-afda-3160e725bf7d}")
    geotest.set_param("out_group", "MVIInversion")
    return geotest.make()


def test_save_data(tmp_path):
    ws, params = setup_params(tmp_path)
    locs = ws.get_entity(params.data_object)[0].centroids
    window = {"center": [np.mean(locs[:, 0]), np.mean(locs[:, 1])], "size": [100, 100]}
    data = InversionData(ws, params, window)

    assert len(data.entity.vertices) > 0


def test_get_uncertainty_component(tmp_path):
    ws, params = setup_params(tmp_path)
    window = params.window()
    params.tmi_uncertainty = 1
    data = InversionData(ws, params, window)
    unc = data.get_uncertainty_component("tmi")
    assert len(np.unique(unc)) == 1
    assert np.unique(unc)[0] == 1
    assert len(unc) == len(data.mask)


def test_parse_ignore_values(tmp_path):
    ws, params = setup_params(tmp_path)
    window = params.window()
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
    window = params.window()
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
    window = params.window()
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
    data = InversionData(ws, params, window)
    test_locs = np.array([[1.0, 2.0, 1.0], [2.0, 1.0, 1.0], [8.0, 9.0, 1.0]])
    radar_ch = np.array([1.0, 2.0, 3.0])
    expected_locs = np.array([[1.0, 2.0, 2.0], [2.0, 1.0, 3.0], [8.0, 9.0, 4.0]])
    draped_locs = data.drape(test_locs, radar_ch)

    assert np.all(draped_locs == expected_locs)


def test_normalize(tmp_path):
    ws, params = setup_params(tmp_path)
    window = params.window()
    data = InversionData(ws, params, window)
    data.data = {"tmi": np.array([1.0, 2.0, 3.0]), "gz": np.array([1.0, 2.0, 3.0])}
    data.components = list(data.data.keys())
    test_data = data.normalize(data.data)
    assert list(data.normalizations.values()) == [1, -1]
    assert all(test_data["gz"] == (-1 * data.data["gz"]))


def test_get_survey(tmp_path):
    ws, params = setup_params(tmp_path)
    window = params.window()
    data = InversionData(ws, params, window)
    survey, _ = data.survey()
    assert isinstance(survey, SimPEG.potential_fields.magnetics.Survey)
