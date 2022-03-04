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
from geoh5py.ui_json.constants import base_validations, default_ui_json, ui_validations
from geoh5py.workspace import Workspace


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

    _validator: InputValidation = None
    _validations = base_validations
    _input_file: InputFile = None
    _monitoring_directory = None
    _free_param_keys: list = None
    _defaults = None
    _ui_json = None
    validate = True

    def __init__(
        self,
        input_file=None,
        defaults=None,
        ui_json=None,
        validate=True,
        validator_options={},
        workpath=".",
        **kwargs,
    ):
        self._monitoring_directory: str = None
        self._workspace_geoh5: str = None
        self._geoh5 = None
        self._run_command: str = None
        self._run_command_boolean: bool = None
        self._conda_environment: str = None
        self._conda_environment_boolean: bool = None

        self.workpath = workpath
        self.input_file = input_file
        self.ui_json = ui_json
        self.defaults = defaults
        self.validate = validate
        self.validator_options = validator_options

        self._initialize(kwargs)

    def _initialize(self, **kwargs):
        """Custom actions to initialize the class and deal with input values."""
        # Set data on inputfile
        if self.input_file is None:
            self.input_file = InputFile(
                ui_json=self.ui_json,
                data=self.defaults,
                validations=self.validations,
                validation_options={"disabled": True},
            )
        self.update(self.input_file.data, validate=False)
        self.param_names = list(self.input_file.data.keys())
        self.input_file.validation_options["disabled"] = False

        # Apply user input
        if any(kwargs):
            kwargs = InputFile.numify(kwargs)
            self.update(kwargs)

    @property
    def defaults(self):
        """
        Dictionary of default parameters and values. Also used to reset the
        order or the ui_json structure.
        """
        return self._defaults

    @defaults.setter
    def defaults(self, values: dict[str, Any] | None):
        if not isinstance(values, (type(None), dict)):
            raise ValueError("Input 'defaults' must be of type dict or None.")

        if self._ui_json is not None:
            for key in values:
                if key not in self.default_ui_json:
                    raise ValueError(
                        f"Input 'defaults' contains unrecognized '{key}'  parameter "
                        "that is not present in the default_ui_json."
                    )

        self._defaults = values

    @property
    def default_ui_json(self):
        """The default ui_json structure"""
        if getattr(self, "_default_ui_json", None) is None:
            self.default_ui_json = deepcopy(default_ui_json)

        return self._default_ui_json

    @default_ui_json.setter
    def default_ui_json(self, ui_json: dict[str, Any] | None):
        if not isinstance(ui_json, (dict, type(None))):
            raise ValueError("Input 'ui_json' must be of type dict.")

        if self.defaults is not None:
            ui_json = {k: ui_json[k] for k in self.defaults}

        self._default_ui_json = InputFile.numify(params_dict)

    def update(self, params_dict: dict[str, Any], validate=True):
        """Update parameters with dictionary contents."""
        original_validate_state = self.validate
        self.validate = validate

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
    def validations(self) -> dict[str, Any]:
        if getattr(self, "_validations", None) is None:
            self._validations = self.input_file.validations
        return self._validations

    @validations.setter
    def validations(self, validations: dict[str, Any]):
        assert isinstance(
            validations, dict
        ), f"Input value must be a dictionary of validations."
        self._validations = validations

    @property
    def validator(self) -> InputValidator:
        if getattr(self, "_validator", None) is None:
            self._validator = self.input_file.validators
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
        if getattr(self, "_input_file", None) is None:
            self.input_file = InputFile(
                ui_json=self.default_ui_json,
                validations=self.validations,
                validator_options=self.validator_options,
            )
        return self._input_file

    @input_file.setter
    def input_file(self, ifile):
        if ifile is None:
            self._input_file = None
            return
        self.validator = self.input_file.validators
        self.validations = self.input_file.validations
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
