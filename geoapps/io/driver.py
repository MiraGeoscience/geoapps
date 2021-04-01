#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import json

from .constants import valid_parameter_values, valid_parameters
from .utils import (
    create_default_output_path,
    create_relative_output_path,
    create_work_path,
)


class InputFile:

    _valid_parameters = valid_parameters
    _required_parameters = ["inversion_type", "core_cell_size"]
    _valid_extensions = ["json"]

    def __init__(self, filepath):
        self.filepath = filepath
        self.workpath = create_work_path(filepath)
        self.data = None
        self.itype = None

    @property
    def filepath(self):
        return self._filepath

    @filepath.setter
    def filepath(self, f):
        if f.split(".")[-1] not in self._valid_extensions:
            raise OSError("Input file must have '.json' extension.")
        else:
            self._filepath = f

    def load(self):
        """ Loads input file contents to dictionary. """
        with open(self.filepath) as f:
            self.data = json.load(f)
            self._validate_required_parameters()
            self._validate_parameters()
            self.itype = self.data["inversion_type"]

    def _validate_parameters(self):
        """ Ensures that all the input files keys are accepted strings."""
        for k in self.data.keys():
            if k not in self._valid_parameters:
                raise ValueError(f"Encountered an invalid input parameter: {k}.")

    def _validate_required_parameters(self):
        """ Ensures that all required input file keys are present."""
        missing = []
        for param in self._required_parameters:
            if param not in self.data.keys():
                missing.append(param)
        if missing:
            raise ValueError(f"Missing required parameter(s): {*missing,}.")


class Params:

    _valid_parameter_values = valid_parameter_values

    def __init__(self, inputfile):
        self.itype = inputfile.itype
        self.workpath = inputfile.workpath
        self._inversion_style = "voxel"
        self._forward_only = False
        self._result_folder = create_default_output_path(inputfile.filepath)
        self._init_params(inputfile)

    @property
    def itype(self):
        return self._itype

    @itype.setter
    def itype(self, val):
        self._validate_parameter_value("inversion_type", val)
        self._itype = val

    @property
    def inversion_style(self):
        return self._inversion_style

    @inversion_style.setter
    def inversion_style(self, val):
        self._validate_parameter_value("inversion_style", val)
        self._inversion_style = val

    @property
    def forward_only(self):
        return self._forward_only

    @forward_only.setter
    def forward_only(self, val):
        self._validate_parameter_value("forward_only", val)
        self._forward_only = val

    @property
    def result_folder(self):
        return self._result_folder

    @result_folder.setter
    def result_folder(self, val):
        self._validate_parameter_value("result_folder", val)
        self._result_folder = val

    def _override_default(self, param, value):
        """ Override parameter default value. """
        self.__setattr__(param, value)

    def _init_params(self, inputfile):
        """ Overrides default parameter values with input file values. """
        for param, val in inputfile.data.items():
            if param == "result_folder":
                self.result_folder = create_relative_output_path(
                    inputfile.workpath, val
                )
            else:
                self._override_default(param, val)

    def _validate_parameter_value(self, param, val):
        """ Validates parameter values against accepted values. """
        vpvals = self._valid_parameter_values[param]
        nvals = len(vpvals)
        if nvals > 1:
            msg = f"Invalid {param} value. Must be one of: {*vpvals,}"
        elif nvals == 1:
            msg = f"Invalid {param} value.  Must be: {vpvals[0]}"
        if val not in vpvals:
            raise ValueError(msg)
