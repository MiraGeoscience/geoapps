# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024-2025 Mira Geoscience Ltd.                                '
#                                                                              '
#  This file is part of geoapps.                                               '
#                                                                              '
#  geoapps is distributed under the terms and conditions of the MIT License    '
#  (see LICENSE file at the root of this source code package).                 '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID

from geoh5py.shared.utils import fetch_active_workspace, str2uuid, uuid2entity
from geoh5py.ui_json import InputFile, InputValidation, utils
from geoh5py.workspace import Workspace

import geoapps


class BaseParams:
    """
    Stores input parameters to drive a ui.json application.

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

    _defaults = None
    _default_ui_json = None
    _free_parameter_keys: list[str] | None = None
    _free_parameter_identifier: str | None = None
    _input_file: InputFile = None
    _monitoring_directory = None
    _ui_json = None
    _validate = True
    _validations = None
    _validation_options = None
    _validator: InputValidation = None
    _version = geoapps.__version__

    def __init__(
        self,
        input_file=None,
        validate=True,
        validation_options=None,
        workpath=None,
        **kwargs,
    ):
        self._monitoring_directory: str = None
        self._workspace_geoh5: str = None
        self._geoh5 = None
        self._run_command: str = None
        self._title = None
        self._conda_environment: str = None
        self._conda_environment_boolean: bool = None
        self._generate_sweep: bool = False
        self._workspace = None
        self._run_command_boolean: bool = False
        self.workpath = workpath
        self.input_file = input_file
        self.validate = validate
        self.validation_options = validation_options

        self._initialize(**kwargs)

    def _initialize(self, **kwargs):
        """Custom actions to initialize the class and deal with input values."""
        # Set data on inputfile
        original_validate_state = self.validate
        self.validate = False

        if self._input_file is None:
            self.input_file = InputFile(
                ui_json=self._default_ui_json,
                validations=self.validations,
                validate=False,
            )
        self.update(self.input_file.data)
        self.validate = original_validate_state
        self.param_names = list(self.input_file.data.keys())

        # Apply user input
        if any(kwargs):
            self.update(kwargs)

    @property
    def version(self):
        return self._version

    @version.setter
    def version(self, val):
        self._version = val

    @property
    def defaults(self):
        """
        Dictionary of default values.
        """
        return self._defaults

    @property
    def ui_json(self):
        """The default ui_json structure stored on InputFile."""
        if getattr(self, "_ui_json", None) is None and self.input_file is not None:
            self._ui_json = self.input_file.ui_json

        return self._ui_json

    def update(self, params_dict: dict[str, Any]):
        """Update parameters with dictionary contents."""
        params_dict = self.input_file.numify(params_dict)
        if params_dict.get("geoh5", None) is not None and self.input_file.geoh5 is None:
            self.geoh5 = params_dict.pop("geoh5")

        with fetch_active_workspace(self.geoh5):
            params_dict = self.input_file.promote(params_dict)  # pylint: disable=W0212

            for key, value in params_dict.items():
                if key not in self.ui_json.keys():
                    continue  # ignores keys not in default_ui_json

                setattr(self, key, value)

            # Set all parameters belonging to groupOptional disabled.
            for key in utils.find_all(self.ui_json, "groupOptional"):
                if key in params_dict and params_dict[key] is None:
                    for elem in utils.collect(
                        self.ui_json, "group", self.ui_json[key]["group"]
                    ):
                        setattr(self, elem, None)

    @property
    def workpath(self):
        """
        Working directory.
        """
        if (
            getattr(self, "_workpath", None) is None
            and getattr(self, "_geoh5", None) is not None
        ):
            self._workpath = Path(self.geoh5.h5file).parent
        return self._workpath

    @workpath.setter
    def workpath(self, val: str | None):
        if val is not None:
            if not Path(val).is_dir():
                raise ValueError("Provided 'workpath' is not a valid path.")
            val = Path(val).resolve()

        self._workpath = val

    @property
    def validation_options(self):
        """Optional validation directives."""
        if self._validation_options is None:
            return {}

        return self._validation_options

    @validation_options.setter
    def validation_options(self, value: dict | None):
        if not isinstance(value, (dict, type(None))):
            raise UserWarning(
                "Input 'validation_options' must a dictionary of options or None."
            )

        self._validation_options = value

    @property
    def validate(self):
        """Return True if validation is enabled."""
        return self._validate

    @validate.setter
    def validate(self, value: bool):
        """Set validation state."""
        if not isinstance(value, bool):
            raise UserWarning("Input 'validate' must be a boolean.")

        self._validate = value

        if self.input_file is not None:
            self.input_file.validate = value

    def to_dict(self, ui_json_format=False):
        """Return params and values dictionary."""

        if ui_json_format:
            return self.input_file.stringify(
                self.input_file.demote(self.input_file.ui_json)
            )

        return {k: getattr(self, k) for k in self.param_names if hasattr(self, k)}

    def active_set(self):
        """Return list of parameters with non-null entries."""
        return [k for k, v in self.to_dict().items() if v is not None]

    def is_uuid(self, p: str) -> bool:
        """Return true if string contains valid UUID."""
        if isinstance(p, str):
            private_attr = self.__getattribute__("_" + p)
            return True if isinstance(private_attr, UUID) else False
        else:
            pass

    @property
    def free_parameter_dict(self):
        """
        Extract groups of free parameters from the ui_json dictionary that match
        the 'free_parameter_identifier' and 'free_parameter_keys'.
        """
        free_parameter_dict = {}
        if (
            getattr(self, "_free_parameter_keys", None) is not None
            and getattr(self, "_free_parameter_identifier", None) is not None
            and getattr(self, "_ui_json", None) is not None
        ):
            ui_groups = list(
                {
                    form["group"]
                    for form in utils.collect(self.ui_json, "group").values()
                    if form["group"] is not None
                }
            )
            ui_groups.sort()
            for group in ui_groups:
                if self._free_parameter_identifier in group.lower():
                    # TODO Create a geoh5py validation for "allof" -> ["object", "levels", "type", "distance"]
                    free_parameter_dict[group] = {}
                    forms = utils.collect(self.ui_json, "group", group)
                    for label, key in zip(
                        forms, self._free_parameter_keys, strict=False
                    ):
                        if key not in label.lower():
                            raise ValueError(
                                f"Malformed input refinement group {group}. "
                                f"Must contain forms for all of {self._free_parameter_keys} in this order."
                            )
                        free_parameter_dict[group][key] = label

        return free_parameter_dict

    @property
    def free_parameter_keys(self) -> list | None:
        """
        String identifiers for free form parameters.
        """
        return self._free_parameter_keys

    @property
    def validations(self) -> dict[str, Any]:
        """Encoded parameter validator type and associated validations."""
        if getattr(self, "_validations", None) is None:
            self._validations = self.input_file.validations
        return self._validations

    @validations.setter
    def validations(self, validations: dict[str, Any]):
        assert isinstance(validations, dict), (
            "Input value must be a dictionary of validations."
        )
        self._validations = validations

    @property
    def geoh5(self):
        return self._geoh5

    @geoh5.setter
    def geoh5(self, val):
        if val is None:
            self._geoh5 = val
            return

        self.setter_validator(
            "geoh5",
            val,
            fun=lambda x: Workspace(x) if isinstance(val, (str, Path)) else x,
        )

    @property
    def run_command(self):
        return self._run_command

    @run_command.setter
    def run_command(self, val):
        self.setter_validator("run_command", val)

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
    def workspace_geoh5(self):
        """Source geoh5 file."""
        return self._workspace_geoh5

    @workspace_geoh5.setter
    def workspace_geoh5(self, val):
        self.setter_validator("workspace_geoh5", val)

    @property
    def input_file(self) -> InputFile | None:
        """
        An InputFile class holding the associated ui_json and validations.
        """
        return self._input_file

    @input_file.setter
    def input_file(self, ifile: InputFile | None):
        if not isinstance(ifile, (type(None), InputFile)):
            raise TypeError(
                f"Value for 'input_file' must be {InputFile} or None. "
                f"Provided {ifile} of type{type(ifile)}"
            )

        if ifile is not None:
            self.validator = ifile.validators
            self.validations = ifile.validations

        self._input_file = ifile

    def _uuid_promoter(self, uid):
        if self.geoh5 is None:
            return uid

        uid = str2uuid(uid)
        entity = uuid2entity(uid, self.geoh5)
        if entity is not None:
            return entity
        return uid

    def setter_validator(self, key: str, value, fun=lambda x: x):
        if value is None:
            setattr(self, f"_{key}", value)
            return

        value = fun(value)

        if value != self.input_file.data[key]:
            self.input_file.set_data_value(key, value)

        setattr(self, f"_{key}", value)

    def write_input_file(
        self,
        name: str | None = None,
        path: str | Path | None = None,
        validate: bool = True,
    ) -> str:
        """Write out a ui.json with the current state of parameters"""
        original_validate_state = self.input_file.validate
        self.input_file.validate = validate
        self.input_file.data = self.to_dict()
        self.input_file.validate = original_validate_state
        return self.input_file.write_ui_json(name=name, path=path)
