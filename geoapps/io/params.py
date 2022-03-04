#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any
from uuid import UUID

from geoh5py.groups import PropertyGroup
from geoh5py.shared import Entity
from geoh5py.ui_json import InputFile, InputValidation
from geoh5py.workspace import Workspace

validations = {
    "geoh5": {
        "required": True,
        "types": [str, Workspace],
    },
}


class Params:
    """
    Stores input parameters to drive an inversion.

    Attributes
    ----------
    geoh5 :
        Path to geoh5 file results workspace object.
    workpath :
        Path to working directory.
    validator :
        Parameter validation class instance.

    Methods
    -------
    is_uuid(p) :
        Returns True if string is valid uuid.
    parent(child_id) :
        Returns parent id for provided child id.
    active() :
        Returns parameters that are not None.
    default(default_ui, param) :
        return default value for param stored in default_ui.

    """

    _geoh5: Workspace = None
    _validator: InputValidation = None
    _ifile: InputFile = None
    _run_command = None
    _run_command_boolean = None
    _conda_environment = None
    _conda_environment_boolean = None
    _title = None
    _monitoring_directory = None
    _free_param_keys: list = None

    def __init__(self, input_file, default=True, validate=True, validator_opts={}):

        self.workpath = "."
        self.input_file = input_file
        self.default = default
        self.validate = validate
        self.validator_opts = validator_opts
        self.geoh5 = None

    def update(self, params_dict: dict[str, Any], validate=True):
        """Update parameters with dictionary contents."""

        original_validate_state = self.validate
        self.validate = validate

        # Pull out workspace data for validations and forward_only for defaults.

        if "geoh5" in params_dict.keys():
            if params_dict["geoh5"] is not None:
                setattr(self, "geoh5", params_dict["geoh5"])

        for key, value in params_dict.items():

            if " " in key:
                continue  # ignores grouped parameter names

            if key not in self.default_ui_json.keys():
                continue  # ignores keys not in default_ui_json

            if isinstance(value, (Entity, PropertyGroup)):
                setattr(self, key, value.uid)
            else:
                setattr(self, key, value)

        self.validate = original_validate_state

    @property
    def workpath(self):
        return os.path.abspath(self._workpath)

    @workpath.setter
    def workpath(self, val):
        self._workpath = val

    @property
    def validations(self):
        """Encoded parameter validator type and associated validations."""
        return self._validations

    def to_dict(self, ui_json_format=False):
        """Return params and values dictionary."""
        params_dict = {
            k: getattr(self, k) for k in self.param_names if hasattr(self, k)
        }
        if ui_json_format:
            self.input_file.data = params_dict
            return self.input_file.ui_json

        return params_dict

    def active_set(self):
        """Return list of parameters with non-null entries."""
        return [k for k, v in self.to_dict().items() if v is not None]

    def is_uuid(self, p: str) -> bool:
        """Return true if string contains valid UUID."""
        if isinstance(p, str):
            private_attr = self.__getattribue__("_" + p)
            return True if isinstance(private_attr, UUID) else False
        else:
            pass

    @property
    def validator(self) -> InputValidator:
        if getattr(self, "_validator", None) is None:
            self._validator = InputValidation(
                ui_json=self.default_ui_json, validations=validations
            )
        return self._validator

    @validator.setter
    def validator(self, validator: InputValidation):
        assert isinstance(
            validator, InputValidation
        ), f"Input value must be of class {InputValidation}"
        self._validator = validator

    @property
    def geoh5(self):
        return self._geoh5

    @geoh5.setter
    def geoh5(self, val):
        if val is None:
            self._geoh5 = val
            return
        self.setter_validator(
            "geoh5", val, fun=lambda x: Workspace(x) if isinstance(val, str) else x
        )

        self.validator.geoh5 = self.geoh5

    @property
    def run_command(self):
        return self._run_command

    @run_command.setter
    def run_command(self, val):
        self.setter_validator("run_command", val)

    @property
    def run_command_boolean(self):
        return self._run_command_boolean

    @run_command_boolean.setter
    def run_command_boolean(self, val):
        self.setter_validator("run_command_boolean", val)

    @property
    def monitoring_directory(self):
        return self._monitoring_directory

    @monitoring_directory.setter
    def monitoring_directory(self, val):
        self.setter_validator("monitoring_directory", val)

    @property
    def conda_environment(self):
        return self._conda_environment

    @conda_environment.setter
    def conda_environment(self, val):
        self.setter_validator("conda_environment", val)

    @property
    def conda_environment_boolean(self):
        return self._conda_environment_boolean

    @conda_environment_boolean.setter
    def conda_environment_boolean(self, val):
        self.setter_validator("conda_environment_boolean", val)

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, val):
        self.setter_validator("title", val)

    @property
    def input_file(self):
        return self._input_file

    @input_file.setter
    def input_file(self, ifile):
        if ifile is None:
            self._input_file = None
            return
        self._input_file = ifile

    def _uuid_promoter(self, x):
        return UUID(x) if isinstance(x, str) else x

    def setter_validator(self, key: str, value, fun=lambda x: x):

        if value is None:
            setattr(self, f"_{key}", value)
            return

        if self.validate:
            if "association" in self.validations[key]:
                validations = deepcopy(self.validations[key])
                validations["association"] = getattr(
                    self, self.validations[key]["association"]
                )
            else:
                validations = self.validations[key]

            self.validator.validate(key, value, validations)
        value = fun(value)
        setattr(self, f"_{key}", value)

    def write_input_file(
        self,
        ui_json: dict = None,
        default: bool = False,
        name: str = None,
        path: str = None,
    ):
        """Write out a ui.json with the current state of parameters"""
        if default:
            self.input_file.validation_options["disabled"] = True
            self.input_file.data = self.defaults
        else:
            self.input_file.data = self.to_dict()

        self.input_file.write_ui_json(name=name, path=path)
