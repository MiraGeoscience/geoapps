#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from typing import Any, Dict, List, Tuple
from uuid import UUID

import numpy as np

from .constants import required_parameters, validations


class InputValidator:
    """
    Validations for inversion parameters.

    Attributes
    ----------
    input : Input file contents parsed to dict.

    Methods
    -------
    validate_input(input)
        Validates input params and contents/type/shape/keys of values.
    validate(param value)
        Validates parameter values, types, shapes, and keys.

    """

    def __init__(self, input: Dict[str, Any] = None):
        self.input = input
        self.current_validation = None

    @property
    def input(self):
        return self._input

    @input.setter
    def input(self, val):
        if val is not None:
            self.validate_input(val)
        self._input = val

    def validate_uuid(self, workspace, uuid_str):
        """ Check whether a string is a valid uuid and addresses an object in the workspace. """
        try:
            obj_uuid = UUID(uuid_str)
        except ValueError:
            raise ValueError(f"Badly formed hexadecimal UUID string: {uuid_str}")

        obj = workspace.get_entity(obj_uuid)
        if obj[0] is None:
            raise IndexError(
                f"UUID address {uuid_str} does not exist in the workspace."
            )
        else:
            return obj_uuid

    def validate_input(self, input: Dict[str, Any]) -> None:
        """
        Validates input params and contents/type/shape/requirements of values.

        Calls validate method on individual key/value pairs in input, and
        handles validations requiring knowledge of other parameters.

        Parameters
        ----------
        input : Input file contents parsed to dict.

        Raises
        ------
        ValueError, TypeError KeyError whenever an input parameter fails one of
        it's value/type/shape/requirement validations.
        """

        self._validate_required_parameters(input)

        for k, v in input.items():
            if k not in validations.keys():
                raise KeyError(f"{k} is not a valid parameter name.")
            else:
                self.validate(k, v, validations[k], list(input.keys()))

    def validate(
        self, param: str, value: Any, validations: Any, input_keys: List[str] = None
    ) -> None:
        """
        Validates parameter values, types, shapes, and requirements.

        Wraps val, type, shape, reqs validate methods depending on whether
        a validation is stored in .constants.validations dictionary

        Parameters
        ----------
        param : Parameter for driving inversion.
        value : Value attached to param.
        validations : validation dictionary from .constants.validations with
            keys for val, type, shape or reqs validation types

        Raises
        ------
        ValueError, TypeError, KeyError
            Whenever an input parameter fails one of it's
            value/type/shape/reqs validations.
        """

        if isinstance(value, dict):
            for k, v in value.items():
                if k not in validations.keys():
                    exclusions = ["values", "types", "shapes", "reqs"]
                    vkeys = [k for k in validations.keys() if k not in exclusions]
                    msg = f"Invalid {param} keys: {k}. Must be one of {*vkeys,}."
                    raise KeyError(msg)
                self.validate(k, v, validations[k], input_keys)

        if value is None:
            if param in required_parameters:
                raise ValueError(f"{param} is a required parameter. Cannot be None.")
            else:
                return

        if "values" in validations.keys():
            self._validate_parameter_val(param, value, validations["values"])
        if "types" in validations.keys():
            self._validate_parameter_type(param, value, validations["types"])
        if "shapes" in validations.keys():
            self._validate_parameter_shape(param, value, validations["shapes"])
        if ("reqs" in validations.keys()) & (input_keys is not None):
            for v in validations["reqs"]:
                self._validate_parameter_reqs(param, value, v, input_keys)

    def _validate_parameter_val(self, param: str, value: Any, vvals: Any) -> None:
        """ Raise ValueError if parameter value is invalid.  """
        if value not in vvals:
            msg = self._param_validation_msg(param, "value", vvals)
            raise ValueError(msg)

    def _validate_parameter_type(self, param: str, value: type, vtypes: Any) -> None:
        """ Raise TypeError if parameter type is invalid. """
        if self._isiterable(value):
            value = np.array(value).flatten().tolist()
            if not all(type(v) in vtypes for v in value):
                tnames = [t.__name__ for t in vtypes]
                msg = self._param_validation_msg(param, "type", tnames)
                raise TypeError(msg)
        elif type(value) not in vtypes:
            tnames = [t.__name__ for t in vtypes]
            msg = self._param_validation_msg(param, "type", tnames)
            raise TypeError(msg)

    def _validate_parameter_shape(
        self, param: str, value: Tuple[int], vshape: Any
    ) -> None:
        """ Raise ValueError if parameter shape is invalid. """
        if np.array(value).shape != vshape:
            msg = self._param_validation_msg(param, "shape", vshape)
            raise ValueError(msg)

    def _validate_parameter_reqs(
        self, param: str, value: Any, vreqs: tuple, input_keys: List[str]
    ) -> None:
        """ Raise a KeyError if parameter requirement is not met. """
        if len(vreqs) > 1:
            if (value == vreqs[0]) & (vreqs[1] not in input_keys):
                msg = self._param_validation_msg(param, "reqs", vreqs)
                raise KeyError(msg)
        else:
            if vreqs[0] not in input_keys:
                msg = self._param_validation_msg(param, "reqs", vreqs)
                raise KeyError(msg)

    def _param_validation_msg(
        self, param: str, validation_type: str, validations: Any
    ) -> str:
        """Generate an error message for parameter validation.

        Parameters
        ----------
        param: parameter name as stored in input file
        validation_type: name of validation type.  One of: 'value', 'type', 'shape', 'reqs'.
        validations: valid input content.

        Returns
        -------
        Message to be printed with the raised exception.
        """

        if validation_type == "shape":
            msg = f"Invalid {param} {validation_type}. Must be: {validations}."

        elif validation_type == "reqs":
            if len(validations) > 1:
                msg = (
                    f"Invalid {param} requirement. Input file must contain "
                    f"'{validations[1]}' if '{param}' is '{validations[0]}'."
                )
            else:
                msg = (
                    f"Invalid {param} requirement. Input file must contain "
                    f"'{validations[0]}' if '{param}' is provided."
                )

        else:
            if self._isiterable(validations, checklen=True):
                msg = f"Invalid {param} {validation_type}. Must be one of: {*validations,}."
            else:
                msg = f"Invalid {param} {validation_type}. Must be: {validations[0]}."

        return msg

    def _validate_required_parameters(self, input: Dict[str, Any]) -> None:
        """
        Ensures that all required input file keys are present.

        Parameters
        ----------
        input : Input file contents parsed to dict.

        Raises
        ------
        ValueError
            If a required parameter (stored in constants.required_parameters)
            is missing from the input file contents.

        """
        missing = []
        for param in required_parameters:
            if param not in input.keys():
                missing.append(param)
        if missing:
            raise ValueError(f"Missing required parameter(s): {*missing,}.")

    def _isiterable(self, v: Any, checklen: bool = False) -> bool:
        """
        Checks if object is iterable.

        Parameters
        ----------
        v : Object to check for iterableness.
        checklen : Restrict objects with __iter__ method to len > 1.

        Returns
        -------
        True if object has __iter__ attribute but is not string or dict type.
        """
        only_array_like = (not isinstance(v, str)) & (not isinstance(v, dict))
        if (hasattr(v, "__iter__")) & only_array_like:
            return False if (checklen and (len(v) == 1)) else True
        else:
            return False
