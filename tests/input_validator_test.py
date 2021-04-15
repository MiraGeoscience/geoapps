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
    msg = validator._param_validation_msg("inversion_type", "value", ["mvi"])
    assert msg == "Invalid inversion_type value. Must be one of: ('mvi',)."
    msg = validator._param_validation_msg(
        "inversion_type", "value", ["mvi", "grav", "mag"]
    )
    assert (
        msg == f"Invalid inversion_type value. Must be one of: ('mvi', 'grav', 'mag')."
    )
    msg = validator._param_validation_msg("inversion_type", "type", [str])
    assert msg == f"Invalid inversion_type type. Must be one of: (<class 'str'>,)."
    msg = validator._param_validation_msg("inversion_type", "type", [int, float])
    assert (
        msg
        == f"Invalid inversion_type type. Must be one of: (<class 'int'>, <class 'float'>)."
    )
    msg = validator._param_validation_msg("inducing_field_aid", "shape", (3,))
    assert msg == f"Invalid inducing_field_aid shape. Must be: (3,)."
    msg = validator._param_validation_msg("inducing_field_aid", "shape", (3, 3))
    assert msg == f"Invalid inducing_field_aid shape. Must be: (3, 3)."


def test_isiterable():
    assert validator._isiterable("test") == False
    assert validator._isiterable(["test"]) == True
    assert validator._isiterable(["Hi", "there"]) == True
    assert validator._isiterable(1) == False
    assert validator._isiterable([1]) == True
    assert validator._isiterable([1, 2]) == True
    assert validator._isiterable(1.0) == False
    assert validator._isiterable([1.0]) == True
    assert validator._isiterable([1.0, 2.0]) == True
    assert validator._isiterable(True) == False
    assert validator._isiterable([True]) == True
    assert validator._isiterable([True, True]) == True
