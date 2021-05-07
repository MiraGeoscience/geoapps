#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import pytest

from geoapps.io.validators import InputValidator, validations

######################  Setup  ###########################

validator = InputValidator()

input_dict = {"inversion_type": "mvi", "core_cell_size": 2}
tmpfile = lambda path: os.path.join(path, "test.json")


def tmp_input_file(filepath, input_dict):
    with open(filepath, "w") as outfile:
        json.dump(input_dict, outfile)


######################  Tests  ###########################


def test_param_validation_msg():
    msg = validator._param_validation_msg("inversion_type", "em", "value", ["mvi"])
    assert msg == "Invalid 'inversion_type' value: 'em'. Must be: 'mvi'."
    msg = validator._param_validation_msg(
        "inversion_type", "EM", "value", ["mvi", "grav", "mag"]
    )
    assert (
        msg
        == f"Invalid 'inversion_type' value: 'EM'. Must be one of: 'mvi', 'grav', 'mag'."
    )
    msg = validator._param_validation_msg("inversion_type", 3, "type", [str])
    assert (
        msg
        == f"Invalid 'inversion_type' type: '<class 'int'>'. Must be: '<class 'str'>'."
    )
    msg = validator._param_validation_msg(
        "inversion_type", "sldkfj", "type", [int, float]
    )
    assert (
        msg
        == f"Invalid 'inversion_type' type: '<class 'str'>'. Must be one of: '<class 'int'>', '<class 'float'>'."
    )
    msg = validator._param_validation_msg("inducing_field_aid", [1, 2], "shape", (3,))
    assert msg == f"Invalid 'inducing_field_aid' shape: (2,). Must be: (3,)."
    msg = validator._param_validation_msg("inducing_field_aid", [1, 2], "shape", (3, 3))
    assert msg == f"Invalid 'inducing_field_aid' shape: (2,). Must be: (3, 3)."
    msg = validator._param_validation_msg(
        "max_distance", 100, "reqs", ("core_cell_size",)
    )
    assert (
        msg
        == f"Unsatisfied 'max_distance' requirement. Input file must contain 'core_cell_size' if 'max_distance' is provided."
    )
    msg = validator._param_validation_msg(
        "mesh_from_params", True, "reqs", (True, "core_cell_size")
    )
    assert (
        msg
        == f"Unsatisfied 'mesh_from_params' requirement. Input file must contain 'core_cell_size' if 'mesh_from_params' is 'True'."
    )


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
