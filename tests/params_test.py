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
from geoapps.io.constants import valid_parameter_values
from geoapps.io.utils import (
    create_default_output_path,
    create_relative_output_path,
    create_work_path,
)

######### Convenience functions ##########

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
    params = Params(inputfile)

    def default_test():
        assert params.__getattribute__(param) == default_value

    return default_test


def param_test_generator(tmp_path, param, invalid_value):
    idict = input_dict.copy()
    idict[param] = invalid_value
    filepath = tmpfile(tmp_path)
    tmp_input_file(filepath, idict)
    inputfile = InputFile(filepath)
    inputfile.load()
    with pytest.raises(ValueError) as excinfo:
        params = Params(inputfile)
    vpvals = valid_parameter_values[param]
    nvals = len(vpvals)
    if nvals > 1:
        msg = f"Invalid {param} value. Must be one of: {*vpvals,}"
    elif nvals == 1:
        msg = f"Invalid {param} value.  Must be: {vpvals[0]}"

    def param_test():
        assert str(excinfo.value) == msg

    return param_test


########### tests ###############


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


def test_override_default(tmp_path):
    filepath = tmpfile(tmp_path)
    tmp_input_file(filepath, input_dict)
    inputfile = InputFile(filepath)
    inputfile.load()
    params = Params(inputfile)
    params._override_default("forward_only", True)
    assert params.forward_only


def test_default_inversion_style(tmp_path):
    dtest = default_test_generator(tmp_path, "inversion_style", "voxel")
    dtest()


def test_validate_inversion_style(tmp_path):
    ptest = param_test_generator(tmp_path, "inversion_style", "parametric")
    ptest()


def test_default_forward_only(tmp_path):
    dtest = default_test_generator(tmp_path, "forward_only", False)
    dtest()


def test_validate_forward_only(tmp_path):
    ptest = param_test_generator(tmp_path, "forward_only", "notgunndoit")
    ptest()


def test_default_result_folder(tmp_path):
    path = os.path.join(tmp_path, "SimPEG_PFInversion") + os.path.sep
    dtest = default_test_generator(tmp_path, "result_folder", path)
    dtest()
