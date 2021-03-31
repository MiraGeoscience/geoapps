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


def test_create_work_path():
    fname = "../assets/test.json"
    inputfile = InputFile(fname)
    wpath = inputfile.create_work_path()
    assert wpath == os.path.abspath("../assets") + os.path.sep


def test_load(tmp_path):
    fname = os.path.join(tmp_path, "test.json")
    input_dict = {"inversion_type": "mvi", "core_cell_size": 2}
    with open(fname, "w") as outfile:
        json.dump(input_dict, outfile)
    inputfile = InputFile(fname)
    inputfile.load()
    assert list(inputfile.data.keys())[0] == "inversion_type"
    assert list(inputfile.data.values())[0] == "mvi"


def test_filepath_extension():
    with pytest.raises(IOError):
        InputFile("test.yaml")


def test_check_parameter_validity(tmp_path):
    fname = os.path.join(tmp_path, "test.json")
    input_dict = {"inversion_method": "mvi"}
    with open(fname, "w") as outfile:
        json.dump(input_dict, outfile)
    inputfile = InputFile(fname)
    with pytest.raises(ValueError):
        inputfile.load()


def test_check_required_parameters(tmp_path):
    fname = os.path.join(tmp_path, "test.json")
    input_dict = {"inversion_style": "voxel"}
    with open(fname, "w") as outfile:
        json.dump(input_dict, outfile)
    inputfile = InputFile(fname)
    with pytest.raises(ValueError):
        inputfile.load()
