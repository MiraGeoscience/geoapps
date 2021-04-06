#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import numpy as np

from .constants import (
    required_parameters,
    valid_parameter_shapes,
    valid_parameter_types,
    valid_parameter_values,
    valid_parameters,
)


class InputValidator:
    def __init__(self, input=None):
        self.input = input

    @property
    def input(self):
        return self._input

    @input.setter
    def input(self, val):
        if val is not None:
            self.validate_input(val)
        self._input = val

    def validate_input(self, input):
        """ Validates input params and contents/type/shape of values. """
        self._validate_required_parameters(input)
        for k, v in input.items():
            self.validate(k, v)

    def validate(self, param, value):
        """ Validates parameter values, types, and shapes. """

        if value is None:
            if param in required_parameters:
                raise ValueError(f"{param} is a required parameter. Cannot be None.")
            else:
                return

        checkval = True if param in valid_parameter_values else False
        checktype = True if param in valid_parameter_types else False
        checkshape = True if param in valid_parameter_shapes else False

        if param not in valid_parameters:
            msg = f"Encountered an invalid input parameter: {param}."
            raise ValueError(msg)
        if checkval:
            self._validate_parameter_val(param, value)
        if checktype:
            self._validate_parameter_type(param, value)
        if checkshape:
            self._validate_parameter_shape(param, value)

    def _validate_parameter_val(self, param, value):
        """ Raise ValueError if parameter value is invalid.  """
        vpvals = valid_parameter_values[param]
        if value not in vpvals:
            msg = self._param_validation_msg(param, "value", vpvals)
            raise ValueError(msg)

    def _validate_parameter_type(self, param, value):
        """ Raise ValueError if parameter type is invalid. """
        vptypes = valid_parameter_types[param]
        tnames = [t.__name__ for t in vptypes]
        msg = self._param_validation_msg(param, "type", tnames)

        if self._isiterable(value):
            if not all(type(v) in vptypes for v in value):
                raise ValueError(msg)
        elif type(value) not in vptypes:
            raise ValueError(msg)

    def _validate_parameter_shape(self, param, value):
        """ Raise ValueError if parameter shape is invalid. """
        vpshape = valid_parameter_shapes[param]
        snames = [s.__str__() for s in vpshape]
        msg = self._param_validation_msg(param, "shape", snames)

        if np.array(value).shape != vpshape:
            raise ValueError(msg)

    def _param_validation_msg(self, param, validation_type, validations):
        """ Generate ValueError message for parameter validation. """

        if validation_type == "shape":
            msg = f"Invalid {param} {validation_type}. Must be: {validations}."
        else:
            if self._isiterable(validations):
                msg = f"Invalid {param} {validation_type}. Must be one of: {*validations,}."
            else:
                msg = f"Invalid {param} {validation_type}. Must be: {validations[0]}."

        return msg

    def _validate_required_parameters(self, input):
        """ Ensures that all required input file keys are present."""
        missing = []
        for param in required_parameters:
            if param not in input.keys():
                missing.append(param)
        if missing:
            raise ValueError(f"Missing required parameter(s): {*missing,}.")

    def _isiterable(self, v):
        if (hasattr(v, "__iter__")) & (not isinstance(v, str)):
            return True if len(v) > 1 else False
        else:
            return False
