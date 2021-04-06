#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import json
import os

import numpy as np
import pytest

from geoapps.io import InputFile, Params
from geoapps.io.constants import (
    valid_parameter_keys,
    valid_parameter_shapes,
    valid_parameter_types,
    valid_parameter_values,
)
from geoapps.io.utils import (
    create_default_output_path,
    create_relative_output_path,
    create_work_path,
)

######################  Setup  ###########################


input_dict = {"inversion_type": "mvi", "core_cell_size": 2}
tmpfile = lambda path: os.path.join(path, "test.json")


def tmp_input_file(filepath, input_dict):
    with open(filepath, "w") as outfile:
        json.dump(input_dict, outfile)


def default_test_generator(tmp_path, param, default_value):
    filepath = tmpfile(tmp_path)
    tmp_input_file(filepath, input_dict)
    params = Params.from_path(filepath)

    assert params.__getattribute__(param) == default_value


def param_test_generator(tmp_path, param, invalid_value, validation_type):
    idict = input_dict.copy()
    filepath = tmpfile(tmp_path)
    tmp_input_file(filepath, idict)
    params = Params.from_path(filepath)
    with pytest.raises(ValueError) as excinfo:
        params.__setattr__(param, invalid_value)
    assert validation_type in str(excinfo.value)
    return params


######################  Tests  ###########################


def test_create_relative_output_path():
    dsep = os.path.sep
    outpath = create_work_path("../../some/project/file")
    path = create_relative_output_path("../assets/Inversion_.json", outpath)
    root = os.path.abspath("..")
    validate_path = os.path.join(root, "assets", "some", "project") + dsep
    assert path == validate_path


def test_create_default_output_path():
    dsep = os.path.sep
    path = create_default_output_path("../assets/Inversion_.json")
    root = os.path.abspath("..")
    validate_path = os.path.join(root, "assets", "SimPEG_PFInversion") + dsep
    assert path == validate_path


def test_create_work_path():
    wp = create_work_path("./inputfile.json")
    assert wp == os.path.abspath(".") + os.path.sep


def test_params_constructors(tmp_path):
    idict = input_dict.copy()
    filepath = tmpfile(tmp_path)
    tmp_input_file(filepath, idict)
    params = Params.from_path(filepath)
    assert params.inversion_type == "mvi"
    assert params.core_cell_size == 2
    assert params.inversion_style == "voxel"
    inputfile = InputFile(filepath)
    params = Params.from_ifile(inputfile)
    assert params.inversion_type == "mvi"
    assert params.core_cell_size == 2
    assert params.inversion_style == "voxel"


def test_override_default():
    params = Params("mvi", 2)
    params._override_default("forward_only", True)
    assert params.forward_only == True


def test_validate_inversion_type(tmp_path):
    param = "inversion_type"
    param_test_generator(tmp_path, param, "em", "value")
    filepath = tmpfile(tmp_path)
    tmp_input_file(filepath, {"core_cell_size": 2})
    with pytest.raises(ValueError) as excinfo:
        params = Params.from_path(filepath)
    assert "parameter(s): ('inversion_type',)." in str(excinfo.value)


def test_validate_core_cell_size(tmp_path):
    param = "core_cell_size"
    param_test_generator(tmp_path, param, "nope", "type")
    filepath = tmpfile(tmp_path)
    tmp_input_file(filepath, {"inversion_type": "mvi"})
    with pytest.raises(ValueError) as excinfo:
        params = Params.from_path(filepath)
    assert "parameter(s): ('core_cell_size',)." in str(excinfo.value)


def test_validate_inversion_style(tmp_path):
    param = "inversion_style"
    param_test_generator(tmp_path, param, "parametric", "value")
    default_test_generator(tmp_path, param, "voxel")


def test_validate_forward_only(tmp_path):
    param = "forward_only"
    param_test_generator(tmp_path, param, "true", "type")
    default_test_generator(tmp_path, param, False)


def test_validate_result_folder(tmp_path):
    param = "result_folder"
    param_test_generator(tmp_path, param, True, "type")
    path = os.path.join(tmp_path, "SimPEG_PFInversion") + os.path.sep
    default_test_generator(tmp_path, param, path)


def test_validate_inducing_field_aid(tmp_path):
    param = "inducing_field_aid"
    param_test_generator(tmp_path, param, "nope", "type")
    param_test_generator(tmp_path, param, [1.0, 2.0], "shape")
    params = Params("mvi", 2)
    params.inducing_field_aid = [1.0, 2.0, 3.0]
    assert type(params.inducing_field_aid) == np.ndarray
    with pytest.raises(ValueError) as excinfo:
        params.inducing_field_aid = [0, 1, 2]
    assert "greater than 0." in str(excinfo.value)
    default_test_generator(tmp_path, param, None)


def test_validate_resolution(tmp_path):
    param = "resolution"
    param_test_generator(tmp_path, param, "nope", "type")
    default_test_generator(tmp_path, param, 0)


def test_validate_window(tmp_path):
    param = "window"
    test_dict = {"center_x": 2, "center_y": 2, "width": 2, "height": 2}
    param_test_generator(tmp_path, param, 1, "type")
    param_test_generator(tmp_path, param, test_dict, "keys")
    default_test_generator(tmp_path, param, None)
    params = Params("mvi", 2)
    test_dict["azimuth"] = 2
    params.window = test_dict
    assert params.window["center"] == [2, 2]
    assert params.window["size"] == [2, 2]


def test_validate_workspace(tmp_path):
    param = "workspace"
    param_test_generator(tmp_path, param, 12234, "type")
    default_test_generator(tmp_path, param, None)
    idict = input_dict.copy()
    idict["data"] = {"type": "GA_object", "name": "test"}
    filepath = tmpfile(tmp_path)
    tmp_input_file(filepath, idict)
    with pytest.raises(ValueError) as excinfo:
        params = Params.from_path(filepath)
    assert "data type 'GA_object'." in str(excinfo.value)


def test_validate_data(tmp_path):
    idict = input_dict.copy()
    idict["data"] = {"type": "GA_object"}
    idict["workspace"] = "."
    filepath = tmpfile(tmp_path)
    tmp_input_file(filepath, idict)
    with pytest.raises(ValueError) as excinfo:
        params = Params.from_path(filepath)
    assert "Data 'type' and 'name'" in str(excinfo.value)


def test_validate_data_format(tmp_path):
    param = "data_format"
    param_test_generator(tmp_path, param, "blah", "value")
    default_test_generator(tmp_path, param, None)


def test_validate_data_name(tmp_path):
    param = "data_name"
    param_test_generator(tmp_path, param, 1234, "type")
    default_test_generator(tmp_path, param, None)


def test_validate_data_channels(tmp_path):
    param = "data_channels"
    test_dict = {"voltage": "yadda"}
    param_test_generator(tmp_path, param, 1, "type")
    param_test_generator(tmp_path, param, test_dict, "keys")
    default_test_generator(tmp_path, param, None)


def test_validate_ignore_values(tmp_path):
    param = "ignore_values"
    param_test_generator(tmp_path, param, 1234, "type")
    default_test_generator(tmp_path, param, None)


def test_validate_detrend(tmp_path):
    param = "detrend"
    param_test_generator(tmp_path, param, 1, "type")
    default_test_generator(tmp_path, param, None)
    idict = input_dict.copy()
    idict["detrend"] = {"corners": 3}
    filepath = tmpfile(tmp_path)
    tmp_input_file(filepath, idict)
    with pytest.raises(ValueError) as excinfo:
        params = Params.from_path(filepath)
    assert "Detrend order must be 0," in str(excinfo.value)


def test_validate_data_file(tmp_path):
    param = "data_file"
    param_test_generator(tmp_path, param, 1234, "type")
    default_test_generator(tmp_path, param, None)
    idict = input_dict.copy()
    idict["data"] = {"type": "ubc_mag", "name": "test"}
    filepath = tmpfile(tmp_path)
    tmp_input_file(filepath, idict)
    with pytest.raises(ValueError) as excinfo:
        params = Params.from_path(filepath)
    assert "for data types 'ubc_grav' and" in str(excinfo.value)
