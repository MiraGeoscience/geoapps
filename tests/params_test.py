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
    inputfile = InputFile(filepath)
    inputfile.load()
    params = Params.from_input_file(inputfile)

    assert params.__getattribute__(param) == default_value


def param_test_generator(tmp_path, param, invalid_value, validation_type, validations):
    idict = input_dict.copy()
    filepath = tmpfile(tmp_path)
    tmp_input_file(filepath, idict)
    inputfile = InputFile(filepath)
    inputfile.load()
    params = Params.from_input_file(inputfile)
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


def test_override_default():
    params = Params("mvi", 2)
    params._override_default("forward_only", True)
    assert params.forward_only == True


def test_default_inversion_style(tmp_path):
    default_test_generator(tmp_path, "inversion_style", "voxel")


def test_validate_inversion_style(tmp_path):
    param = "inversion_style"
    vpvals = valid_parameter_values[param]
    param_test_generator(tmp_path, param, "parametric", "value", vpvals)


def test_default_forward_only(tmp_path):
    default_test_generator(tmp_path, "forward_only", False)


def test_validate_forward_only(tmp_path):
    param = "forward_only"
    vptypes = valid_parameter_types[param]
    param_test_generator(tmp_path, param, "true", "type", vptypes)


def test_default_result_folder(tmp_path):
    path = os.path.join(tmp_path, "SimPEG_PFInversion") + os.path.sep
    default_test_generator(tmp_path, "result_folder", path)


def test_validate_result_folder(tmp_path):
    param = "result_folder"
    vptypes = valid_parameter_types[param]
    param_test_generator(tmp_path, param, True, "type", vptypes)


def test_default_inducing_field_aid(tmp_path):
    default_test_generator(tmp_path, "inducing_field_aid", None)


def test_validate_inducing_field_aid(tmp_path):
    param = "inducing_field_aid"
    vptypes = valid_parameter_types[param]
    vpshapes = valid_parameter_shapes[param]
    param_test_generator(tmp_path, param, "nope", "type", vptypes)
    param_test_generator(tmp_path, param, [1.0, 2.0], "shape", vpshapes)
    params = Params("mvi", 2)
    params.inducing_field_aid = [1.0, 2.0, 3.0]
    assert type(params.inducing_field_aid) == np.ndarray
    with pytest.raises(ValueError) as excinfo:
        params.inducing_field_aid = [0, 1, 2]
    assert "greater than 0." in str(excinfo.value)
