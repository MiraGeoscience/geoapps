#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Any
from uuid import UUID

from geoh5py.workspace import Workspace

from .input_file import InputFile
from .validators import InputValidator

required_parameters = ["workspace", "geoh5"]
validations = {
    "workspace": {
        "types": [str, Workspace],
    },
    "geoh5": {
        "types": [str, Workspace],
    },
}


class Params:
    """
    Stores input parameters to drive an inversion.

    Attributes
    ----------
    workspace :
        Path to geoh5 file workspace object.
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

    Constructors
    ------------
    from_input_file(input_file)
        Construct Params object from InputFile instance.
    from_path(path)
        Construct Params object from path to input file (wraps from_input_file constructor).

    """

    _default_ui_json = {}
    associations: dict[str | UUID, str | UUID] = None
    _workspace: Workspace = None
    _output_geoh5: str = None
    _validator: InputValidator = None
    _ifile: InputFile = None

    def __init__(self, validate=True, **kwargs):
        self.update(self.default_ui_json)
        if kwargs:
            self.update(kwargs)
            if validate:
                ifile = InputFile.from_dict(self.to_dict(), self.validator)
            else:
                ifile = InputFile.from_dict(self.to_dict())

            cls = self.from_input_file(ifile)
            self.__dict__.update(cls.__dict__)

    @classmethod
    def from_input_file(
        cls, input_file: InputFile, workspace: Workspace = None
    ) -> Params:
        """Construct Params object from InputFile instance.

        Parameters
        ----------
        input_file : InputFile
            class instance to handle loading input file
        """

        p = cls()
        p._input_file = input_file
        p.workpath = input_file.workpath
        p.associations = input_file.associations

        for v in ["geoh5", "workspace"]:
            if v in input_file.data.keys():
                if input_file.data[v] is not None:
                    p.workspace = Workspace(input_file.data[v])
                    p.geoh5 = Workspace(input_file.data[v])
            input_file.data.pop(v, None)

        if workspace is not None:
            p.workspace = (
                Workspace(workspace) if isinstance(workspace, str) else workspace
            )

        p.update(input_file.data)

        return p

    @classmethod
    def from_path(cls, file_path: str, workspace: Workspace = None) -> Params:
        """
        Construct Params object from path to input file.

        Parameters
        ----------
        file_path : str
            path to input file.
        """
        p = cls()
        input_file = InputFile(file_path, p.validator, workspace)
        p = cls.from_input_file(input_file, workspace)

        return p

    @property
    def default_ui_json(self):
        """Dictionary of default values structured in ANALYST ui.json format"""
        return self._default_ui_json

    @property
    def required_parameters(self):
        """Parameters required on initialization."""
        return self._required_parameters

    @property
    def validations(self):
        """Encoded parameter validator type and associated validations."""
        return self._validations

    def update(self, params_dict: Dict[str, Any], default: bool = False):
        """Update parameters with dictionary contents."""
        for key, value in params_dict.items():
            try:
                if isinstance(value, dict):
                    field = "value"
                    if default:
                        field = "default"
                    elif "isValue" in value.values():
                        if not value["isValue"]:
                            field = "property"
                    setattr(self, key, value[field])
                else:
                    setattr(self, key, value)
            except AttributeError:
                warnings.warn(
                    f"Skipping dictionary entry: {key}.  Not a valid attribute."
                )
                continue

    def _init_params(
        self,
        inputfile: InputFile,
        required_parameters: list[str] = required_parameters,
        validations: dict[str, Any] = validations,
        workspace: Workspace = None,
    ) -> None:
        """Overrides default parameter values with input file values."""

        self.workspace = workspace
        if getattr(self, "workspace", None) is None:
            self.workspace = Workspace(inputfile.data["geoh5"])

        if inputfile.data["geoh5"] is None:
            self.output_geoh5 = self.workspace
        else:
            self.geoh5 = Workspace(inputfile.data["geoh5"])

        self.validator.workspace = self.workspace
        self.validator.input = inputfile  # Triggers validations

        for param, value in inputfile.data.items():
            try:
                if param in ["workspace", "geoh5"]:
                    continue
                self.__setattr__(param, value)
            except KeyError:
                continue

    def to_dict(self, ui_json_format=True):
        """Return params and values dictionary."""

        if ui_json_format:
            ui_json = deepcopy(self._default_ui_json)
            for k in self.param_names:
                if isinstance(ui_json[k], dict):
                    field = "value"
                    if "isValue" in ui_json[k].keys():
                        if ui_json[k]["isValue"] is False:
                            field = "property"
                    ui_json[k][field] = getattr(self, k)
                else:
                    ui_json[k] = getattr(self, k)

            return ui_json

        else:
            return {k: getattr(self, k) for k in self.param_names}

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

    def default(self, default_ui: dict[str, Any], param: str) -> Any:
        """Return default value of parameter stored in default_ui_json."""
        return default_ui[param]["default"]

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
    def workspace(self):
        return self._workspace

    @workspace.setter
    def workspace(self, val):
        if val is None:
            self._workspace = val
            return
        self.setter_validator(
            "workspace", val, fun=lambda x: Workspace(x) if isinstance(val, str) else x
        )

    @property
    def geoh5(self):
        return self._geoh5

    @geoh5.setter
    def geoh5(self, val):
        if val is None:
            self._geoh5 = val
            return
        self.setter_validator("geoh5", val)

    @property
    def input_file(self):
        return self._input_file

    def setter_validator(self, key: str, value, fun=lambda x: x):
        if value is None:
            setattr(self, f"_{key}", value)
            return

        self.validator.validate(
            key, value, self.validations[key], self.workspace, self.associations
        )
        setattr(self, f"_{key}", fun(value))

    def write_input_file(self, name: str = None):
        """Write out a ui.json with the current state of parameters"""

        ifile = InputFile.from_dict(
            self.to_dict(), self.required_parameters, self.validations
        )
        ifile.write_ui_json(self.default_ui_json, default=False, name=name)
        # if getattr(self, "input_file", None) is not None:
        #     input_dict = self.default_ui_json
        #     if self.input_file.input_dict is not None:
        #         input_dict = self.input_file.input_dict
        #
        #     params = {}
        #     for key in self.input_file.data.keys():
        #         try:
        #             value = getattr(self, key)
        #             if hasattr(value, "h5file"):
        #                 value = value.h5file
        #             params[key] = value
        #         except KeyError:
        #             continue
        #
        #     self.input_file.workpath = self.workpath
        #     self.input_file.write_ui_json(
        #         input_dict,
        #         name=name,
        #         param_dict=params,
        #         workspace=self.workspace.h5file,
        #     )
