#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import os
from copy import deepcopy
from typing import Any, Dict, List, Union
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
    associations: Dict[Union[str, UUID], Union[str, UUID]] = None
    _workspace: Workspace = None
    _output_geoh5: str = None
    _validator: InputValidator = None
    _ifile: InputFile = None

    def __init__(self, **kwargs):

        self.workpath: str = os.path.abspath(".")
        ui = deepcopy(self.default_ui_json)

        if kwargs.keys():
            ui.update({k: v for k, v in kwargs if k in ui})
            ifile = InputFile()
            ifile.input_from_dict(ui)
            self._init_params(ifile)
        else:
            self._set_defaults(ui)

    @property
    def default_ui_json(self):
        """Dictionary of default values structured in ANALYST ui.json format"""
        return self._default_ui_json

    @classmethod
    def from_input_file(cls, input_file: InputFile, **kwargs) -> "Params":
        """Construct Params object from InputFile instance.

        Parameters
        ----------
        input_file : InputFile
            class instance to handle loading input file
        """
        if not input_file.is_loaded:
            input_file.read_ui_json()

        p = cls()
        p._ifile = input_file
        p.workpath = input_file.workpath
        p.associations = input_file.associations
        p._init_params(input_file)

        return p

    @classmethod
    def from_path(cls, file_path: str, **kwargs) -> "Params":
        """
        Construct Params object from path to input file.

        Parameters
        ----------
        file_path : str
            path to input file.
        """

        input_file = InputFile(file_path)
        p = cls.from_input_file(input_file)

        return p

    @classmethod
    def from_dict(cls, ui_json: dict) -> "Params":
        p = cls()
        p.init_from_dict(ui_json)
        return p

    def init_from_dict(self, ui_json: dict) -> None:
        """
        Construct Params object from a dictionary.

        Parameters
        ----------
        ui_json: Dictionary of parameters store in ui.json format
        """
        self._input_file = InputFile()
        self._input_file.input_from_dict(ui_json, required_parameters, validations)
        self.workpath = self._input_file.workpath
        self.associations = self._input_file.associations
        self._init_params(self._input_file)

    def _set_defaults(self, default_ui: Dict[str, Any]) -> None:
        """ Populate parameters with default values stored in default_ui. """
        for key, value in default_ui.items():
            try:
                if isinstance(default_ui[key], dict):
                    self.__setattr__(key, value["default"])
                else:
                    self.__setattr__(key, value)
            except KeyError:

                continue

    def _init_params(
        self,
        inputfile: InputFile,
        required_parameters: List[str] = required_parameters,
        validations: Dict[str, Any] = validations,
    ) -> None:
        """ Overrides default parameter values with input file values. """

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

    def is_uuid(self, p: str) -> bool:
        """ Return true if string contains valid UUID. """
        if isinstance(p, str):
            private_attr = self.__getattribue__("_" + p)
            return True if isinstance(private_attr, UUID) else False
        else:
            pass

    def parent(self, child_id: Union[str, UUID]) -> Union[str, UUID]:
        """ Returns parent id of provided child id. """
        return self.associations[child_id]

    def active_set(self) -> List[str]:
        """ Retrieve active parameter set (value not None). """
        return [k[1:] for k, v in self.__dict__.items() if v is not None]

    def default(self, default_ui: Dict[str, Any], param: str) -> Any:
        """ Return default value of parameter stored in default_ui_json. """
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
        p = "workspace"
        self.validator.validate(p, val, validations[p])
        if isinstance(val, str):
            self._workspace = Workspace(val)
        else:
            self._workspace = val

    @property
    def geoh5(self):
        return self._geoh5

    @geoh5.setter
    def geoh5(self, val):
        if val is None:
            self._geoh5 = val
            return
        p = "geoh5"
        self.validator.validate(p, val, validations[p])
        self._geoh5 = val

    @property
    def input_file(self):
        return self._input_file

    def write_input_file(self, name: str = None):
        """Write out a ui.json with the current state of parameters"""
        if getattr(self, "input_file", None) is not None:
            input_dict = self.default_ui_json
            if self.input_file.input_dict is not None:
                input_dict = self.input_file.input_dict

            params = {}
            for key in self.input_file.data.keys():
                try:
                    value = getattr(self, key)
                    if hasattr(value, "h5file"):
                        value = value.h5file
                    params[key] = value
                except KeyError:
                    continue

            self.input_file.write_ui_json(
                input_dict,
                name=name,
                param_dict=params,
                workspace=self.workspace.h5file,
            )
