#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import json
import os

import numpy as np

from .utils import (
    create_default_output_path,
    create_relative_output_path,
    create_work_path,
)
from .validators import InputValidator


class InputFile:

    _valid_extensions = ["json"]

    def __init__(self, filepath):
        self.filepath = filepath
        self.workpath = create_work_path(filepath)
        self.data = None
        self.validator = None
        self.isloaded = False

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
            self.validator = InputValidator(self.data)
        self.isloaded = True


class Params:
    def __init__(self, inversion_type, core_cell_size, workpath=os.path.abspath(".")):
        self.validator = InputValidator()
        self.inversion_type = inversion_type
        self.core_cell_size = core_cell_size
        self.workpath = workpath
        self.inversion_style = "voxel"
        self.forward_only = False
        self.result_folder = os.path.join(workpath, "SimPEG_PFInversion")
        self.inducing_field_aid = None
        self.resolution = 0
        self.window = None
        self.workspace = None
        self.data_format = None
        self.data_name = None
        self.data_channels = None
        self.ignore_values = None
        self.detrend = None
        self.data_file = None

    @classmethod
    def from_ifile(cls, ifile):
        """ Construct Params object from InputFile instance. """
        if not ifile.isloaded:
            ifile.load()
        inversion_type = ifile.data["inversion_type"]
        core_cell_size = ifile.data["core_cell_size"]
        p = cls(inversion_type, core_cell_size, ifile.workpath)
        p._init_params(ifile)
        return p

    @classmethod
    def from_path(cls, filepath):
        """ Construct Params object from .json input file path. """
        ifile = InputFile(filepath)
        p = cls.from_ifile(ifile)
        return p

    @property
    def inversion_type(self):
        return self._inversion_type

    @inversion_type.setter
    def inversion_type(self, val):
        self.validator.validate("inversion_type", val)
        self._inversion_type = val

    @property
    def core_cell_size(self):
        return self._core_cell_size

    @core_cell_size.setter
    def core_cell_size(self, val):
        self.validator.validate("core_cell_size", val)
        self._core_cell_size = val

    @property
    def inversion_style(self):
        return self._inversion_style

    @inversion_style.setter
    def inversion_style(self, val):
        self.validator.validate("inversion_style", val)
        self._inversion_style = val

    @property
    def forward_only(self):
        return self._forward_only

    @forward_only.setter
    def forward_only(self, val):
        self.validator.validate("forward_only", val)
        self._forward_only = val

    @property
    def result_folder(self):
        return self._result_folder

    @result_folder.setter
    def result_folder(self, val):
        self.validator.validate("result_folder", val)
        sep = os.path.sep
        if val.split(sep)[-1]:
            val += sep
        self._result_folder = val

    @property
    def inducing_field_aid(self):
        return self._inducing_field_aid

    @inducing_field_aid.setter
    def inducing_field_aid(self, val):
        if val is None:
            self._inducing_field_aid = val
            return
        self.validator.validate("inducing_field_aid", val)
        if val[0] <= 0:
            raise ValueError("inducing_field_aid[0] must be greater than 0.")
        self._inducing_field_aid = np.array(val)

    @property
    def resolution(self):
        return self._resolution

    @resolution.setter
    def resolution(self, val):
        self.validator.validate("resolution", val)
        self._resolution = val

    @property
    def window(self):
        return self._window

    @window.setter
    def window(self, val):
        if val is None:
            self._window = val
            return
        self.validator.validate("window", val)
        if "center" not in val.keys():
            val["center"] = [val["center_x"], val["center_y"]]
        if "size" not in val.keys():
            val["size"] = [val["width"], val["height"]]
        self._window = val

    @property
    def data_format(self):
        return self._data_format

    @data_format.setter
    def data_format(self, val):
        if val is None:
            self._data_format = val
            return
        self.validator.validate("data_format", val)
        self._data_format = val

    @property
    def data_name(self):
        return self._data_name

    @data_name.setter
    def data_name(self, val):
        if val is None:
            self._data_name = val
            return
        self.validator.validate("data_name", val)
        self._data_name = val

    @property
    def data_channels(self):
        return self._data_channels

    @data_channels.setter
    def data_channels(self, val):
        if val is None:
            self._data_channels = val
            return
        self.validator.validate("data_channels", val)
        # for v in val.keys():
        #     valid_keys = ["tmi"]
        #     if v not in valid_keys:
        #         raise ValueError(f"Invalid key {v}. Must be one of {*valid_keys,}")
        self._data_channels = val

    @property
    def workspace(self):
        return self._workspace

    @workspace.setter
    def workspace(self, val):
        if val is None:
            self._workspace = val
            return
        self.validator.validate("workspace", val)
        self._workspace = val

    @property
    def ignore_values(self):
        return self._ignore_values

    @ignore_values.setter
    def ignore_values(self, val):
        if val is None:
            self._ignore_values = val
            return
        self.validator.validate("ignore_values", val)
        self._ignore_values = val

    @property
    def detrend(self):
        return self._detrend

    @detrend.setter
    def detrend(self, val):
        if val is None:
            self._detrend = val
            return
        self.validator.validate("detrend", val)
        for v in val.values():
            if v not in [0, 1, 2]:
                raise ValueError("Detrend order must be 0, 1, or 2.")
        self._detrend = val

    @property
    def data_file(self):
        return self._data_file

    @data_file.setter
    def data_file(self, val):
        if val is None:
            self._data_file = val
            return
        self.validator.validate("data_file", val)
        self._data_file = val

    def _override_default(self, param, value):
        """ Override parameter default value. """
        self.__setattr__(param, value)

    def _init_params(self, inputfile):
        """ Overrides default parameter values with input file values. """
        for param, value in inputfile.data.items():
            if param == "result_folder":
                op = create_relative_output_path(inputfile.workpath, value)
                self._override_default(param, op)
            elif param == "data":
                self._override_default("data_format", value["type"])
                self._override_default("data_name", value["name"])
                if "channels" in value.keys():
                    self._override_default("data_channels", value["channels"])
            else:
                self._override_default(param, value)
