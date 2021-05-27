#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import pytest

from geoapps.io import InputFile
from geoapps.io.MVI.constants import required_parameters, validations
from geoapps.io.validators import InputValidator

######################  Setup  ###########################

ifile = InputFile("test.ui.json")
ifile.data = {"mesh_from_params": True, "core_cell_size_x": 2}
validator = InputValidator(required_parameters, validations, input=ifile)

input_dict = {"inversion_type": "mvi", "core_cell_size": 2}
tmpfile = lambda path: os.path.join(path, "test.json")


def tmp_input_file(filepath, input_dict):
    with open(filepath, "w") as outfile:
        json.dump(input_dict, outfile)


######################  Tests  ###########################


def test_validate_parameter_val():
    param = "inversion_type"
    value = "em"
    validations = ["mvi", "gravity"]
    vtype = "value"
    with pytest.raises(ValueError) as excinfo:
        validator._validate_parameter_val(param, value, validations)
    msg = f"Invalid '{param}' {vtype}: '{value}'. Must be one of: 'mvi', 'gravity'."
    assert msg in str(excinfo.value)
    validations = ["mvi"]
    with pytest.raises(ValueError) as excinfo:
        validator._validate_parameter_val(param, value, validations)
    msg = f"Invalid '{param}' {vtype}: '{value}'. Must be: 'mvi'."
    assert msg in str(excinfo.value)


def test_validate_parameter_type():
    param = "max_distance"
    value = "notafloat"
    validations = [int, float]
    vtype = "type"
    with pytest.raises(TypeError) as excinfo:
        validator._validate_parameter_type(param, value, validations)
    msg = f"Invalid '{param}' {vtype}: 'str'. Must be one of: 'int', 'float'."
    assert msg in str(excinfo.value)
    validations = [int]
    with pytest.raises(TypeError) as excinfo:
        validator._validate_parameter_type(param, value, validations)
    msg = f"Invalid '{param}' {vtype}: 'str'. Must be: 'int'."
    assert msg in str(excinfo.value)
    param = "octree_levels_topo"
    value = ["1", 2, 3]
    validations = [int, float]
    with pytest.raises(TypeError) as excinfo:
        validator._validate_parameter_type(param, value, validations)
    msg = f"Invalid '{param}' {vtype}: 'str'. Must be one of: 'int', 'float'."
    assert msg in str(excinfo.value)
    param = "ignore_values"
    value = 342
    validations = [str]
    vtype = "type"
    with pytest.raises(TypeError) as excinfo:
        validator._validate_parameter_type(param, value, validations)
    msg = f"Invalid '{param}' {vtype}: 'int'. Must be: 'str'."
    assert msg in str(excinfo.value)


def test_validate_parameter_shape():
    param = "octree_levels_topo"
    value = [1, 2]
    validations = [(3,)]
    vtype = "shape"
    with pytest.raises(ValueError) as excinfo:
        validator._validate_parameter_shape(param, value, validations)
    msg = f"Invalid '{param}' {vtype}: '(2,)'. Must be: '(3,)'."
    assert msg in str(excinfo.value)


def test_validate_parameter_req():
    param = "topography"
    value = "sdetselkj"
    validations = ("topography_object",)
    vtype = "reqs"
    ifile.data["core_cell_size"] = None
    with pytest.raises(KeyError) as excinfo:
        validator._validate_parameter_req(param, value, validations)
    msg = f"Unsatisfied '{param}' requirement. Input file must contain "
    msg += f"'{validations[0]}' if '{param}' is provided."
    assert msg in str(excinfo.value)
    param = "mesh_from_params"
    value = True
    validations = (
        True,
        "core_cell_size",
    )
    vtype = "reqs"
    input_keys = ["mesh_from_param", "topography"]
    with pytest.raises(KeyError) as excinfo:
        validator._validate_parameter_req(param, value, validations)
    msg = f"Unsatisfied '{param}' requirement. Input file must contain "
    msg += f"'{validations[1]}' if '{param}' is '{str(value)}'."
    assert msg in str(excinfo.value)


def test_validate_parameter_uuid():
    param = "topography"
    value = "lskdfjsdlkfj"
    vtype = "uuid"
    with pytest.raises(ValueError) as excinfo:
        validator._validate_parameter_uuid(param, value)
    msg = f"Invalid '{param}' {vtype}: '{value}'. Must be a valid uuid string"
    assert msg in str(excinfo.value)


def test_isiterable():
    assert validator._isiterable("test") == False
    assert validator._isiterable(["test"]) == True
    assert validator._isiterable(["Hi", "there"]) == True
    assert validator._isiterable(1) == False
    assert validator._isiterable([1], checklen=True) == False
    assert validator._isiterable([1, 2]) == True
    assert validator._isiterable(1.0) == False
    assert validator._isiterable([1.0], checklen=True) == False
    assert validator._isiterable([1.0, 2.0]) == True
    assert validator._isiterable(True) == False
    assert validator._isiterable([True], checklen=True) == False
    assert validator._isiterable([True, True]) == True
