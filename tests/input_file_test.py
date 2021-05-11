#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import json
import os
from copy import deepcopy
from uuid import UUID

import numpy as np
import pytest

from geoapps.io import InputFile
from geoapps.io.MVI.constants import default_ui_json

######################  Setup  ###########################

d_u_j = deepcopy(default_ui_json)
input_dict = {k: v["default"] for k, v in d_u_j.items() if isinstance(v, dict)}
tmpfile = lambda path: os.path.join(path, "test.json")


def tmp_input_file(filepath, input_dict):
    with open(filepath, "w") as outfile:
        json.dump(input_dict, outfile)


######################  Tests  ###########################


def test_filepath_extension():

    with pytest.raises(IOError) as excinfo:
        InputFile("test.yaml")
    msg = "Input file must have 'ui.json' extension."
    assert str(excinfo.value) == msg


def test_default():

    ifile = InputFile("test.ui.json")
    ifile.default(default_ui_json)
    ifile.data


def test_dict_mapper():

    tdict = {"key1": {"key2": {"key3": "yargh"}}}
    ifile = InputFile("test.ui.json")
    f = lambda y, x: x[:-1] if x == "yargh" else x
    for k, v in tdict.items():
        v = ifile._dict_mapper(k, v, [f])
        tdict[k] = v
    assert tdict["key1"]["key2"]["key3"] == "yarg"


def test_stringify():

    tdict = {"test_n": None, "test_l": [1, 2], "test_i": np.inf}
    tdict.update({"test_i2": -np.inf, "choiceList": [1, 2]})
    tdict.update({"meshType": ["asdfas", "dafsdf"]})
    ifile = InputFile("test.ui.json")
    sdict = ifile._stringify(tdict)
    assert sdict["test_n"] == ""
    assert sdict["test_l"] == "1, 2"
    assert sdict["test_i"] == "inf"
    assert sdict["test_i2"] == "-inf"
    assert sdict["choiceList"] == [1, 2]
    assert sdict["meshType"] == ["asdfas", "dafsdf"]


def test_numify():

    tdict = {"test_i": "inf", "test_i2": "-inf", "test_n": ""}
    tdict.update({"test_l": "1, 2", "test_l2": "1,"})
    ifile = InputFile("test.ui.json")
    ndict = ifile._numify(tdict)
    assert ndict["test_i"] == np.inf
    assert ndict["test_i2"] == -np.inf
    assert ndict["test_n"] is None
    assert ndict["test_l"] == [1, 2]
    assert ndict["test_l2"] == [1]


def test_ui_2_py():

    tdict = {"run_command": "blah", "max_distance": "notgettinsaved"}
    tdict.update({"inversion_type": {"value": "mvi"}})
    tdict.update({"detrend_order": {"default": 0, "enabled": False, "value": "ohya"}})
    tdict.update(
        {"detrend_type": {"default": "all", "visible": False, "value": "ohno"}}
    )
    tdict.update({"topography": {"isValue": True, "property": "yep", "value": 2}})
    tdict.update({"topography2": {"isValue": False, "property": "yep", "value": 2}})
    tdict.update({"tmi_channel": {"value": "ldskfjsld"}})
    ifile = InputFile("test.ui.json")
    ifile._ui_2_py(tdict)
    assert ifile.data["run_command"] == "blah"
    assert ifile.data["inversion_type"] == "mvi"
    assert ifile.data["detrend_order"] == 0
    assert ifile.data["detrend_type"] == "all"
    assert ifile.data["topography"] == 2
    assert ifile.data["topography2"] == "yep"
    assert ifile.data["tmi_channel"] == "ldskfjsld"


def test_set_associations():

    o_uuid = "{538a7eb1-2218-4bec-98cc-0a759aa0ef4f}"
    tdict = {"obj": {"value": o_uuid}}
    f_uuid = "{44822654-b6ae-45b0-8886-2d845f80f422}"
    tdict.update(
        {"field": {"parent": "obj", "isValue": True, "property": "", "value": f_uuid}}
    )
    o_uuid2 = "{ab3c2083-6ea8-4d31-9230-7aad3ec09525}"
    tdict.update({"obj2": {"value": o_uuid2}})
    f_uuid2 = "{a603a762-f6cb-4b21-afda-3160e725bf7d}"
    tdict.update(
        {
            "field2": {
                "parent": "obj2",
                "isValue": False,
                "property": f_uuid2,
                "value": "",
            }
        }
    )
    ifile = InputFile("test.ui.json")
    ifile._set_associations(tdict)
    assert ifile.associations[UUID(f_uuid)] == UUID(o_uuid)
    assert ifile.associations[UUID(f_uuid2)] == UUID(o_uuid2)
    assert ifile.associations["field"] == "obj"
    assert ifile.associations["field2"] == "obj2"


def test_ui_json_io(tmp_path):

    ifile = InputFile(os.path.join(tmp_path, "test.ui.json"))
    assert not ifile.is_loaded
    assert not ifile.is_formatted
    ifile.write_ui_json(default_ui_json, default=True)
    ifile.read_ui_json(reformat=False)
    assert ifile.data == d_u_j
    assert ifile.is_loaded
    assert not ifile.is_formatted
    ifile.read_ui_json(reformat=True)
    assert ifile.is_formatted
    ifile.data["inversion_type"] = "gravity"
    ifile.data["forward_only"] = True
    ifile.write_ui_json(default_ui_json, default=False)
    ifile.read_ui_json(reformat=True)
    assert ifile.data["inversion_type"] == "mvi"
    assert ifile.data["forward_only"] == True
