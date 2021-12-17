#  Copyright (c) 2021 Mira Geoscience Ltd.
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

from geoh5py.shared import Entity
from geoh5py.workspace import Workspace

from .input_file import InputFile
from .validators import InputValidator

required_parameters = ["geoh5"]
validations = {
    "geoh5": {
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
    associations :
        Stores parent/child relationships.

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

    associations: dict[str | UUID, str | UUID] = None
    _geoh5: Workspace = None
    _validator: InputValidator = None
    _ifile: InputFile = None
    _run_command = None
    _run_command_boolean = None
    _conda_environment = None
    _conda_environment_boolean = None
    _title = None
    _monitoring_directory = None
    _free_param_keys: list = None

    def __init__(
        self, input_file=None, default=True, validate=True, validator_opts={}, **kwargs
    ):

        self.workpath = "."
        self.input_file = input_file
        self.default = default
        self.validate = validate
        self.validator_opts = validator_opts
        self.geoh5 = None

    def update(self, params_dict: Dict[str, Any], validate=True):
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

            if isinstance(value, dict):
                field = "value"
                if "isValue" in value.keys():
                    if not value["isValue"]:
                        field = "property"
                setattr(self, key, value[field])
            else:
                if isinstance(value, Entity):
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
    def required_parameters(self):
        """Parameters required on initialization."""
        return self._required_parameters

    @property
    def validations(self):
        """Encoded parameter validator type and associated validations."""
        return self._validations

    def to_dict(self, ui_json: dict = None, ui_json_format=True):
        """Return params and values dictionary."""
        if ui_json_format:
            if ui_json is None:
                ui_json = deepcopy(self.default_ui_json)

            for k in self.param_names:
                if k not in ui_json.keys() or not hasattr(self, k):
                    continue
                new_val = getattr(self, k)
                if isinstance(ui_json[k], dict):
                    field = "value"
                    if "isValue" in ui_json[k].keys():
                        if isinstance(new_val, UUID) or new_val is None:
                            ui_json[k]["isValue"] = False
                            field = "property"
                        else:
                            ui_json[k]["isValue"] = True

                    if ui_json[k][field] != new_val:
                        ui_json[k]["enabled"] = True
                        ui_json[k][field] = new_val

                else:
                    ui_json[k] = new_val

            return ui_json

        else:
            return {k: getattr(self, k) for k in self.param_names if hasattr(self, k)}

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

    def parent(self, child_id: str | UUID) -> str | UUID:
        """Returns parent id of provided child id."""
        return self.associations[child_id]

    @property
    def validator(self) -> InputValidator:

        if getattr(self, "_validator", None) is None:
            self._validator = InputValidator(required_parameters, validations)
        return self._validator

    @validator.setter
    def validator(self, validator: InputValidator):
        assert isinstance(
            validator, InputValidator
        ), f"Input value must be of class {InputValidator}"
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
        self.associations = ifile.associations
        self.workpath = ifile.workpath
        self._input_file = ifile

    def setter_validator(self, key: str, value, fun=lambda x: x):

        if value is None:
            setattr(self, f"_{key}", value)
            return

        if self.validate:
            self.validator.validate(
                key, value, self.validations[key], self.geoh5, self.associations
            )
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

        if name is not None:
            if ".ui.json" not in name:
                name += ".ui.json"
        else:
            name = f"{self.__class__.__name__}.ui.json"

        if ui_json is None:
            ui_json = self.default_ui_json

        if default:
            ifile = InputFile()
        else:
            ifile = InputFile.from_dict(self.to_dict(ui_json=ui_json))

        if path is not None:
            if not os.path.exists(path):
                raise ValueError(f"Provided path {path} does not exist.")
            ifile.workpath = path

        ifile.write_ui_json(ui_json, name=name, default=default)

    def get_associations(self, params_dict: dict[str, Any]):
        associations = InputFile.get_associations(self.default_ui_json)
        uuid_associations = {}
        for k, v in associations.items():
            if all([p in params_dict.keys() for p in [k, v]]):
                child = params_dict[k]
                parent = params_dict[v]
                child = child.uid if isinstance(child, Entity) else child
                parent = parent.uid if isinstance(parent, Entity) else parent
                if all([InputFile.is_uuid(p) for p in [child, parent]]):
                    uuid_associations[child] = parent
        associations.update(uuid_associations)
        return associations
