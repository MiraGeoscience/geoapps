#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import json
import os

from .constants import valid_parameter_values, valid_parameters


class InputFile:

    _valid_parameters = valid_parameters
    _required_parameters = ["inversion_type", "core_cell_size"]
    _valid_extensions = ["json"]

    def __init__(self, filepath):
        self.filepath = filepath
        self.workpath = self.create_work_path()
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

    def create_work_path(self):
        """ Creates absolute path to input file. """
        dsep = os.path.sep
        workpath = os.path.dirname(os.path.abspath(self.filepath)) + dsep

        return workpath

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
        self._load_params_from_input_file(inputfile)

    @property
    def itype(self):
        return self._itype

    @property
    def inversion_style(self):
        return self._inversion_style

    @itype.setter
    def itype(self, val):
        self._validate_parameter_value("inversion_type", val)
        self._itype = val

    @inversion_style.setter
    def inversion_style(self, val):
        self._validate_parameter_value("inversion_style", val)
        self._inversion_style = val

    def _load_params_from_input_file(self, inputfile):
        """ Overrides default parameter values input file values. """
        if "inversion_style" in inputfile.data.keys():
            self.inversion_style = inputfile.data["inversion_style"]

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
