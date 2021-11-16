#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import numpy as np

from geoapps.io import UIJson
from geoapps.io.Gravity.constants import default_ui_json


def test_constructor():
    uijson = UIJson(default_ui_json)
    assert uijson.ui == default_ui_json


def test_group():
    uijson = UIJson(default_ui_json)
    window_group = uijson.group("Data window")
    check = [
        "window_center_x",
        "window_center_y",
        "window_width",
        "window_height",
        "window_azimuth",
    ]
    assert np.all(np.sort(check) == np.sort(list(window_group.keys())))


def test_collect():
    uijson = UIJson(default_ui_json)
    enabled_params = uijson.collect("enabled", value=True)
    assert all(["enabled" in v for v in enabled_params.values()])
    assert all([v["enabled"] for v in enabled_params.values()])


def test_data():
    uijson = UIJson(default_ui_json)
    data = uijson.data()
    assert data["starting_model"] is None
    assert data["tile_spatial"] == 1
    assert data["forward_only"] == False
    assert data["resolution"] == None


def test_group_enabled():
    uijson = UIJson(default_ui_json)
    assert not uijson.group_enabled("Data window")


def test_truth():
    uijson = UIJson(default_ui_json)
    assert not uijson.truth("detrend_order", "enabled")
    assert uijson.truth("max_chunk_size", "enabled")
    assert uijson.truth("chunk_by_rows", "enabled")
    assert not uijson.truth("chunk_by_rows", "optional")
