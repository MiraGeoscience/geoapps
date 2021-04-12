#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import numpy as np

from .constants import (
    required_parameters,
    valid_parameter_keys,
    valid_parameter_shapes,
    valid_parameter_types,
    valid_parameter_values,
    valid_parameters,
)


class InputValidator:
    """
    Validations for inversion parameters.

    Attributes
    ----------
    input : dict, optional
        Input file contents parsed to dict

    Methods
    -------
    validate_input(input)
        Validates input params and contents/type/shape/keys of values.
    validate(param value)
        Validates parameter values, types, shapes, and keys.

    """

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
        """Validates input params and contents/type/shape/keys of values.

        Calls validate method on individual key/value pairs in input, and
        handles validations requiring knowledge of other parameters.

        Parameters
        ----------
        input : dict
            Input file contents parsed to dict.

        Raises
        ------
        ValueError, TypeError whenever an input parameter fails one of it's
        value/type/shape/key validations.
        """

        self._validate_required_parameters(input)
        if "data" in input.keys():
            require_keys = "type" not in input["data"].keys()
            require_keys |= "name" not in input["data"].keys()
            if require_keys:
                raise ValueError("Data 'type' and 'name' must not be empty.")
            if input["data"]["type"] == "GA_object":
                if "workspace" not in input.keys():
                    msg = "Input file must contain a 'workspace' path "
                    msg += "for data type 'GA_object'."
                    raise ValueError(msg)
            if input["data"]["type"] in ["ubc_grav", "ubc_mag"]:
                if "data_file" not in input.keys():
                    msg = "Input file must contain an 'input_file' path"
                    msg += " for data types 'ubc_grav' and 'ubc_mag'."
                    raise ValueError(msg)
        elif "input_mesh" in input.keys():
            if "save_to_geoh5" not in input.keys():
                msg = "Input file must contain a 'save_to_geoh5' path if"
                msg += " 'input_mesh' provided."
                raise ValueError(msg)
            if "input_mesh_file" not in input.keys():
                msg = "Input file must contain a 'input_mesh_file' path if"
                msg += " 'input_mesh' provided."
                raise ValueError(msg)
        elif "forward_only" in input.keys():
            if input["forward_only"] == True:
                if "reference_model" not in input.keys():
                    msg = "A reference model/value must be provided for "
                    msg += "forward modeling"
                    raise ValueError(msg)
        elif "reference_model" in input.keys():
            if "value" in input["reference_model"].keys():
                v = input["reference_model"]["value"]
                v = np.r_[v]
                if v.shape[0] not in [1, 3]:
                    msg = "Start model needs to be a scalar or 3 component vector"
                    raise ValueError(msg)
        elif "save_to_geoh5" in input.keys():
            if "out_group" not in input.keys():
                msg = "Input file must contain a 'out_group' string if"
                msg += "'save_to_geoh5' provided."
                raise ValueError(msg)

        for k, v in input.items():
            self.validate(k, v)

    def validate(self, param, value):
        """Validates parameter values, types, shapes, and keys.

        Wraps val, type, shape, keys validate methods depending on whether
        a validation is stored in .constants.valid_parameter_values,
        .constants.valid_parameter_types, .constants.valid_parameter_shapes,
        and .constants.valid_parameter_keys.

        Parameters
        ----------
        param : str
            Parameter for driving inversion.
        value : anything
            Value attached to param.

        Raises
        ------
        ValueError, TypeError
            Whenever an input parameter fails one of it's
            value/type/shape/key validations.
        """

        if value is None:
            if param in required_parameters:
                raise ValueError(f"{param} is a required parameter. Cannot be None.")
            else:
                return

        checkval = True if param in valid_parameter_values else False
        checktype = True if param in valid_parameter_types else False
        checkshape = True if param in valid_parameter_shapes else False
        checkkeys = True if param in valid_parameter_keys else False

        if param not in valid_parameters:
            msg = f"Encountered an invalid input parameter: {param}."
            raise ValueError(msg)
        if checkval:
            self._validate_parameter_val(param, value)
        if checktype:
            self._validate_parameter_type(param, value)
        if checkshape:
            self._validate_parameter_shape(param, value)
        if checkkeys:
            self._validate_parameter_keys(param, value)

    def _validate_parameter_val(self, param, value):
        """ Raise ValueError if parameter value is invalid.  """
        vpvals = valid_parameter_values[param]
        if value not in vpvals:
            msg = self._param_validation_msg(param, "value", vpvals)
            raise ValueError(msg)

    def _validate_parameter_type(self, param, value):
        """ Raise TypeError if parameter type is invalid. """
        vptypes = valid_parameter_types[param]
        tnames = [t.__name__ for t in vptypes]
        msg = self._param_validation_msg(param, "type", tnames)
        if self._isiterable(value):
            value = np.array(value).flatten().tolist()
            if not all(type(v) in vptypes for v in value):
                raise TypeError(msg)
        elif type(value) not in vptypes:
            raise TypeError(msg)

    def _validate_parameter_shape(self, param, value):
        """ Raise ValueError if parameter shape is invalid. """
        vpshape = valid_parameter_shapes[param]
        snames = [s.__str__() for s in vpshape]
        msg = self._param_validation_msg(param, "shape", snames)

        if np.array(value).shape != vpshape:
            raise ValueError(msg)

    def _validate_parameter_keys(self, param, value):
        """ Raise ValueError if dict type parameter has invalid keys. """
        vpkeys = valid_parameter_keys[param]
        msg = self._param_validation_msg(param, "keys", vpkeys)

        if not all(k in vpkeys for k in value.keys()):
            raise ValueError(msg)

    def _param_validation_msg(self, param, validation_type, validations):
        """Generate an error message for parameter validation.

        Parameters
        ----------
        param: str
            parameter name as stored in input file
        validation_type: str
            name of validation type.  One of: 'value', 'type', 'shape', 'keys'.
        validations: any
            valid input content.

        Returns
        -------
        string
            Message to be printed with the raised exception.
        """

        if validation_type == "keys":
            msg = (
                f"Invalid {param} {validation_type}. Must be one of : {*validations,}."
            )
        elif validation_type == "shape":
            msg = f"Invalid {param} {validation_type}. Must be: {validations}."
        else:
            if self._isiterable(validations):
                msg = f"Invalid {param} {validation_type}. Must be one of: {*validations,}."
            else:
                msg = f"Invalid {param} {validation_type}. Must be: {validations[0]}."

        return msg

    def _validate_required_parameters(self, input):
        """
        Ensures that all required input file keys are present.

        Parameters
        ----------
        input : dict
            Input file contents parsed to dict.

        Raises
        ------
        ValueError
            If a required parameter (stored ing constants.required_parameters)
            is missing from the input file contents.

        """
        missing = []
        for param in required_parameters:
            if param not in input.keys():
                missing.append(param)
        if missing:
            raise ValueError(f"Missing required parameter(s): {*missing,}.")

    def _isiterable(self, v):
        """
        Checks if object is iterable

        Returns
        -------
        bool
            True if object has __iter__ attribute but is not string or dict type.
        """
        only_array_like = (not isinstance(v, str)) & (not isinstance(v, dict))
        if (hasattr(v, "__iter__")) & only_array_like:
            return True
        else:
            return False
