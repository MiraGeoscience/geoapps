#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from typing import Any, Dict, List, Tuple, Union
from uuid import UUID

import numpy as np
from geoh5py.workspace import Workspace


class InputValidator:
    """
    Validations for driver parameters.

    Attributes
    ----------
    requirements : List of required parameters.
    validations : Validations dictionary with matching set of input parameter keys.
    workspace (optional) : Workspace instance needed to validate uuid types.
    input (optional) : Input file contents parsed to dict.

    Methods
    -------
    validate_uuid(uuid)
        validates string as a valid uuid.
    validate_input(input)
        Validates input params and contents/type/shape/keys of values.
    validate(param value)
        Validates parameter values, types, shapes, and keys.

    """

    def __init__(
        self,
        requirements: List[str],
        validations: Dict[str, Any],
        workspace: Workspace = None,
        input=None,
    ):
        self.requirements = requirements
        self.validations = validations
        self.workspace = workspace
        self.input = input

    @property
    def requirements(self):
        return self._requirements

    @requirements.setter
    def requirements(self, val):
        self._requirements = val

    @property
    def validations(self):
        return self._validations

    @validations.setter
    def validations(self, val):
        self._validations = val

    @property
    def input(self):
        return self._input

    @input.setter
    def input(self, val):
        self._input = val
        if val is not None:
            self.validate_input(val)

    def validate_input(self, input) -> None:
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

        self._validate_requirements(input.data)

        for k, v in input.data.items():
            if k not in self.validations.keys():
                raise KeyError(f"{k} is not a valid parameter name.")
            else:
                self.validate(
                    k, v, self.validations[k], self.workspace, input.associations
                )

    def validate(
        self,
        param: str,
        value: Any,
        pvalidations: Dict[str, List[Any]],
        workspace: Workspace = None,
        associations: Dict[Union[str, UUID], Union[str, UUID]] = None,
    ) -> None:
        """
        Validates parameter values, types, shapes, and requirements.

        Wraps val, type, shape, reqs validate methods and applies each method according to what
        is stored in the pvalidations dictionary.  If value is a dictionary type validate() will
        recurse until value is not a dictionary and check that the data keys exist in the
        pvalidations key set.

        Parameters
        ----------
        param : Parameter name.
        value : Value attached to param.
        pvalidations : validation dictionary mapping validation type to its validators.

        Raises
        ------
        ValueError, TypeError, KeyError
            Whenever an input parameter fails one of it's
            value/type/shape/reqs validations.
        ValueError
            If param value is None, but is a required parameter.
        KeyError
            If value is a dictionary and any of it's keys do not exist in pvalidators.
        """

        if isinstance(value, dict):
            for k, v in value.items():
                if k not in pvalidations.keys():
                    exclusions = ["values", "types", "shapes", "reqs", "uuid"]
                    vkeys = [k for k in pvalidations.keys() if k not in exclusions]
                    msg = self.iterable_validation_msg(param, "keys", k, vkeys)
                    raise KeyError(msg)
                self.validate(k, v, pvalidations[k], workspace, associations)

        if value is None:
            if param in self.requirements:
                raise ValueError(f"{param} is a required parameter. Cannot be None.")
            else:
                return

        if "values" in pvalidations.keys():
            self._validate_parameter_val(param, value, pvalidations["values"])
        if "types" in pvalidations.keys():
            self._validate_parameter_type(param, value, pvalidations["types"])
        if "shapes" in pvalidations.keys():
            self._validate_parameter_shape(param, value, pvalidations["shapes"])
        if ("reqs" in pvalidations.keys()) & (self.input is not None):
            for req in pvalidations["reqs"]:
                self._validate_parameter_req(param, value, req)
        if "uuid" in pvalidations.keys():
            ws = self.workspace if workspace is None else workspace
            try:
                child_uuid = UUID(value) if isinstance(value, str) else value
                parent = associations[child_uuid]
            except:
                parent = None
            self._validate_parameter_uuid(param, value, ws, parent)

    def _validate_parameter_val(
        self, param: str, value: Any, vvals: List[Union[float, str]]
    ) -> None:
        """ Raise ValueError if parameter value is invalid.  """
        if value not in vvals:
            msg = self._iterable_validation_msg(param, "value", value, vvals)
            raise ValueError(msg)

    def _validate_parameter_type(
        self, param: str, value: Any, vtypes: List[type]
    ) -> None:
        """ Raise TypeError if parameter type is invalid. """
        isiter = self._isiterable(value)
        value = np.array(value).flatten().tolist()[0] if isiter else value
        if type(value) not in vtypes:
            tnames = [t.__name__ for t in vtypes]
            ptype = type(value).__name__
            msg = self._iterable_validation_msg(param, "type", ptype, tnames)
            raise TypeError(msg)

    def _validate_parameter_shape(
        self, param: str, value: Any, vshape: List[Tuple[int]]
    ) -> None:
        """ Raise ValueError if parameter shape is invalid. """
        pshape = np.array(value).shape
        if pshape != vshape:
            msg = self._iterable_validation_msg(param, "shape", pshape, vshape)
            raise ValueError(msg)

    def _validate_parameter_req(self, param: str, value: Any, req: tuple) -> None:
        """ Raise a KeyError if parameter requirement is not satisfied. """

        hasval = len(req) > 1  # req[0] contains value for which param req[1] must exist
        preq = req[1] if hasval else req[0]
        val = req[0] if hasval else None

        if hasval:
            if value != req[0]:
                return

        if preq in self.input.data.keys():
            noreq = True if self.input.data[preq] == None else False
        else:
            noreq = True

        if noreq:
            msg = self._req_validation_msg(param, preq, val)
            raise KeyError(msg)

    def _req_validation_msg(self, param, preq, val=None):
        """ Generate unsatisfied parameter requirement message. """

        msg = f"Unsatisfied '{param}' requirement. Input file must contain "
        if val is not None:
            msg += f"'{preq}' if '{param}' is '{str(val)}'."
        else:
            msg += f"'{preq}' if '{param}' is provided."
        return msg

    def _validate_parameter_uuid(
        self, param: str, value: str, workspace: Workspace = None, parent: UUID = None
    ) -> None:
        """ Check whether a string is a valid uuid and addresses an object in the workspace. """

        msg = self._general_validation_msg(param, "uuid", value)
        try:
            obj_uuid = UUID(value)
        except ValueError:
            msg += " Must be a valid uuid string."
            raise ValueError(msg)

        if workspace is not None:
            obj = workspace.get_entity(obj_uuid)
            if obj[0] is None:
                msg += f" Address does not exist in workspace: {workspace}."
                raise IndexError(msg)

        if parent is not None:
            parent_obj = workspace.get_entity(parent)[0]
            if obj_uuid not in [c.uid for c in parent_obj.children]:
                msg += f" Object must be a child of {parent}."
                raise IndexError(msg)

    def _general_validation_msg(self, param: str, type: str, value: Any) -> str:
        """ Generate base error message: "Invalid '{param}' {type}: {value}.". """
        return f"Invalid '{param}' {type}: '{value}'."

    def _iterable_validation_msg(
        self, param: str, type: str, value: Any, validations: List[Any]
    ) -> str:
        """ Append possibly iterable validations: "Must be (one of): {validations}.". """

        msg = self._general_validation_msg(param, type, value)
        if self._isiterable(validations, checklen=True):
            vstr = "'" + "', '".join(str(k) for k in validations) + "'"
            msg += f" Must be one of: {vstr}."
        else:
            msg += f" Must be: '{validations[0]}'."

        return msg

    def _validate_requirements(
        self, input: Dict[str, Any], requirements: List[str] = None
    ) -> None:
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
        reqs = self.requirements if requirements is None else requirements

        missing = []
        for param in reqs:
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
