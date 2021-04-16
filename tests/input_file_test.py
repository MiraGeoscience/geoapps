#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import json
import os

import pytest

from geoapps.io import InputFile
from geoapps.io.driver import create_work_path

######################  Setup  ###########################

input_dict = {
    "inversion_type": "mvi",
    "workspace": "Lovelace",
    "out_group": "yep",
    "data": {
        "name": "Garry",
        "channels": {
            "tmi": {
                "name": "Airborne_TMI",
                "uncertainties": [0.0, 13.58],
                "offsets": [0.0, 0.0, 0.0],
            },
        },
    },
    "mesh": "some_path",
    "topography": {
        "GA_object": {
            "name": "some_other_path",
        },
    },
}
tmpfile = lambda path: os.path.join(path, "test.json")


def tmp_input_file(filepath, input_dict):
    with open(filepath, "w") as outfile:
        json.dump(input_dict, outfile)


######################  Tests  ###########################


def test_filepath_extension():
    with pytest.raises(IOError) as excinfo:
        InputFile("test.yaml")
    msg = "Input file must have '.json' extension."
    assert str(excinfo.value) == msg


def test_create_work_path():
    fname = "../assets/test.json"
    wpath = create_work_path(fname)
    assert wpath == os.path.abspath("../assets") + os.path.sep


def test_load(tmp_path):
    filepath = tmpfile(tmp_path)
    tmp_input_file(filepath, input_dict)
    inputfile = InputFile(filepath)
    inputfile.load()
    assert "inversion_type" in inputfile.data.keys()
    assert "workspace" in inputfile.data.keys()
    assert "out_group" in inputfile.data.keys()
    assert "data" in inputfile.data.keys()
    assert "mesh" in inputfile.data.keys()
    assert "topography" in inputfile.data.keys()


def test_validate_parameters(tmp_path):
    filepath = tmpfile(tmp_path)
    idict = input_dict.copy()
    idict["inversion_method"] = "mvi"
    tmp_input_file(filepath, idict)
    inputfile = InputFile(filepath)
    with pytest.raises(KeyError) as excinfo:
        inputfile.load()
    msg = "'inversion_method is not a valid parameter name.'"
    assert str(excinfo.value) == msg


def test_validate_required_parameters(tmp_path):
    filepath = tmpfile(tmp_path)
    idict = input_dict.copy()
    idict = {"inversion_style": "voxel"}
    tmp_input_file(filepath, idict)
    inputfile = InputFile(filepath)
    with pytest.raises(ValueError) as excinfo:
        inputfile.load()
    assert "parameter(s): ('inversion_type', 'work" in str(excinfo.value)
