#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import os
from typing import Any, Dict, List, Union
from uuid import UUID

from geoh5py.workspace import Workspace

from .input_file import InputFile
from .validators import InputValidator

required_parameters = ["workspace", "output_geoh5"]
validations = {
    "workspace": {
        "types": [str, Workspace],
    },
    "output_geoh5": {
        "types": [str, Workspace],
    },
}


class Params:
    """
    Stores input parameters to drive an inversion.

    Attributes
    ----------
    workspace :
        Path to geoh5py workspace object.
    output_geoh5:
        Path to geoh5py results workspace object.
    workpath :
        Path to working directory.
    validator :
        Parameter validation class instance.
    associations :
        Stores parent/child relationships.

    Methods
    -------
    is_uuid(p)
        Returns True if string is valid uuid.
    parent(child_id)
        Returns parent id for provided child id
    active()
        Returns parameters that are not None
    default(default_ui, param)
        return default value for param stored in default_ui

    Constructors
    ------------
    from_ifile(ifile)
        Construct Params object from InputFile instance.
    from_path(path)
        Construct Params object from path to input file (wraps from_ifile constructor).

    """

    def __init__(self):

        self.workspace = None
        self.output_geoh5 = None
        self.workpath = os.path.abspath(".")
        self.validator = InputValidator(required_parameters, validations)
        self.associations = {}
        self._ifile = None

    @classmethod
    def from_ifile(cls, ifile: InputFile) -> None:
        """Construct Params object from InputFile instance.

        Parameters
        ----------
        ifile : InputFile
            class instance to handle loading input file
        """
        if not ifile.is_loaded:
            ifile.read_ui_json()

        p = cls()
        p._ifile = ifile
        p.workpath = ifile.workpath
        p.associations = ifile.associations
        p._init_params(ifile)

        return p

    @classmethod
    def from_path(cls, filepath: str) -> None:
        """
        Construct Params object from path to input file.

        Parameters
        ----------
        filepath : str
            path to input file.
        """
        ifile = InputFile(filepath)
        p = cls.from_ifile(ifile)
        return p

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

    def active(self) -> List[str]:
        """ Retrieve active parameter set (value not None). """
        return [k[1:] for k, v in self.__dict__.items() if v is not None]

    def default(self, default_ui: Dict[str, Any], param: str) -> Any:
        """ Return default value of parameter stored in default_ui_json. """
        return default_ui[param]["default"]

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
    def output_geoh5(self):
        return self._output_geoh5

    @output_geoh5.setter
    def output_geoh5(self, val):
        if val is None:
            self._output_geoh5 = val
            return
        p = "output_geoh5"
        self.validator.validate(p, val, validations[p])
        self._output_geoh5 = val

    def _set_defaults(self, default_ui: Dict[str, Any]) -> None:
        """ Populate parameters with default values stored in default_ui. """
        for a in self.__dict__.keys():
            if a in ["_ifile", "validations", "validator", "workpath", "associations"]:
                continue
            self.__setattr__(a, default_ui[a[1:]]["default"])

    def _init_params(self, inputfile: InputFile) -> None:
        """ Overrides default parameter values with input file values. """

        self.workspace = inputfile.data["workspace_geoh5"]
        self.output_geoh5 = inputfile.data["geoh5"]
        for param, value in inputfile.data.items():
            if param in ["workspace", "output_geoh5"]:
                continue
            self.__setattr__(param, value)
