#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import json
import os

import pytest

from geoapps.io import InputFile, Params

######### Convenience functions ##########

input_dict = {"inversion_type": "mvi", "core_cell_size": 2}
tmpfile = lambda path: os.path.join(path, "test.json")


def tmp_input_file(filepath, input_dict):
    with open(filepath, "w") as outfile:
        json.dump(input_dict, outfile)


def default_test_generator(tmp_path, default_value):
    filepath = tmpfile(tmp_path)
    tmp_input_file(filepath, input_dict)
    inputfile = InputFile(filepath)
    inputfile.load()
    params = Params(inputfile)

    def default_test():
        assert params.inversion_style == default_value

    return default_test


def param_test_generator(tmp_path, param, invalid_value, valid_value):
    idict = input_dict.copy()
    idict[param] = invalid_value
    filepath = tmpfile(tmp_path)
    tmp_input_file(filepath, idict)
    inputfile = InputFile(filepath)
    inputfile.load()
    with pytest.raises(ValueError) as excinfo:
        params = Params(inputfile)
    msg = f"Invalid {param} value.  Must be: {valid_value}"

    def param_test():
        assert str(excinfo.value) == msg

    return param_test


########### tests ###############


def test_filepath_extension():
    with pytest.raises(IOError) as excinfo:
        InputFile("test.yaml")
    msg = "Input file must have '.json' extension."
    assert str(excinfo.value) == msg


def test_create_work_path():
    fname = "../assets/test.json"
    inputfile = InputFile(fname)
    wpath = inputfile.create_work_path()
    assert wpath == os.path.abspath("../assets") + os.path.sep


def test_load(tmp_path):
    filepath = tmpfile(tmp_path)
    tmp_input_file(filepath, input_dict)
    inputfile = InputFile(filepath)
    inputfile.load()
    assert "inversion_type" in inputfile.data.keys()
    assert "core_cell_size" in inputfile.data.keys()
    assert "mvi" in inputfile.data.values()
    assert 2 in inputfile.data.values()
    assert len(inputfile.data.keys()) == 2


def test_validate_parameters(tmp_path):
    filepath = tmpfile(tmp_path)
    idict = input_dict.copy()
    idict["inversion_method"] = "mvi"
    tmp_input_file(filepath, idict)
    inputfile = InputFile(filepath)
    with pytest.raises(ValueError) as excinfo:
        inputfile.load()
    msg = f"Encountered an invalid input parameter: {'inversion_method'}."
    assert str(excinfo.value) == msg


def test_validate_required_parameters(tmp_path):
    filepath = tmpfile(tmp_path)
    idict = input_dict.copy()
    idict = {"inversion_style": "voxel"}
    tmp_input_file(filepath, idict)
    inputfile = InputFile(filepath)
    with pytest.raises(ValueError) as excinfo:
        inputfile.load()
    msg = f"Missing required parameter(s): ('inversion_type', 'core_cell_size')."
    assert str(excinfo.value) == msg


def test_default_inversion_style(tmp_path):
    dtest = default_test_generator(tmp_path, "voxel")
    dtest()


def test_validate_inversion_style(tmp_path):
    param = "inversion_style"
    i_value = "parametric"
    v_value = "voxel"
    ptest = param_test_generator(tmp_path, param, i_value, v_value)
    ptest()
