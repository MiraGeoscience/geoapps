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
        self.new_uncert = None
        self.input_mesh = None
        self.save_to_geoh5 = None
        self.inversion_mesh_type = "TREE"
        self.shift_mesh_z0 = None
        self.topography = None
        self.receivers_offset = None
        self.chi_factor = 1
        self.model_norms = [2, 2, 2, 2]
        self.max_iterations = 10
        self.max_cg_iterations = 30
        self.tol_cg = 1e-4
        self.max_global_iterations = 100

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
        req_keys = ["center_x", "center_y", "width", "height", "azimuth"]
        if not all(k in val.keys() for k in req_keys):
            msg = "Input parameter 'window' dictionary must contain "
            msg += f"all of {*req_keys,}."
            raise ValueError(msg)
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

    @property
    def new_uncert(self):
        return self._new_uncert

    @new_uncert.setter
    def new_uncert(self, val):
        if val is None:
            self._new_uncert = val
            return
        self.validator.validate("new_uncert", val)
        if (val[0] < 0) | (val[0] > 1):
            msg = "Uncertainty percent (new_uncert[0]) must be between 0 and 1."
            raise ValueError(msg)
        if val[1] < 0:
            msg = "Uncertainty floor (new_uncert[1]) must be greater than 0."
            raise ValueError(msg)
        self._new_uncert = val

    @property
    def input_mesh(self):
        return self._input_mesh

    @input_mesh.setter
    def input_mesh(self, val):
        self.validator.validate("input_mesh", val)
        self._input_mesh = val

    @property
    def save_to_geoh5(self):
        return self._save_to_geoh5

    @save_to_geoh5.setter
    def save_to_geoh5(self, val):
        self.validator.validate("save_to_geoh5", val)
        self._save_to_geoh5 = val

    @property
    def inversion_mesh_type(self):
        return self._inversion_mesh_type

    @inversion_mesh_type.setter
    def inversion_mesh_type(self, val):
        self.validator.validate("inversion_mesh_type", val)
        self._inversion_mesh_type = val

    @property
    def shift_mesh_z0(self):
        return self._shift_mesh_z0

    @shift_mesh_z0.setter
    def shift_mesh_z0(self, val):
        self.validator.validate("shift_mesh_z0", val)
        self._shift_mesh_z0 = val

    @property
    def topography(self):
        return self._topography

    @topography.setter
    def topography(self, val):
        self.validator.validate("topography", val)
        self._topography = val

    @property
    def receivers_offset(self):
        return self._receivers_offset

    @receivers_offset.setter
    def receivers_offset(self, val):
        self.validator.validate("receivers_offset", val)
        self._receivers_offset = val

    @property
    def chi_factor(self):
        return self._chi_factor

    @chi_factor.setter
    def chi_factor(self, val):
        self.validator.validate("chi_factor", val)
        if val <= 0:
            raise ValueError("Invalid chi_factor. Must be between 0 and 1.")
        self._chi_factor = val

    @property
    def model_norms(self):
        return self._model_norms

    @model_norms.setter
    def model_norms(self, val):
        self.validator.validate("model_norms", val)
        if len(val) % 4 != 0:
            raise ValueError("Invalid 'model_norms' length.  Must be a multiple of 4.")
        self._model_norms = val

    @property
    def max_iterations(self):
        return self._max_iterations

    @max_iterations.setter
    def max_iterations(self, val):
        self.validator.validate("max_iterations", val)
        if val <= 0:
            raise ValueError("Invalid 'max_iterations' value.  Must be > 0.")
        self._max_iterations = val

    @property
    def max_cg_iterations(self):
        return self._max_cg_iterations

    @max_cg_iterations.setter
    def max_cg_iterations(self, val):
        self.validator.validate("max_cg_iterations", val)
        if val <= 0:
            raise ValueError("Invalid 'max_cg_iterations' value.  Must be > 0.")
        self._max_cg_iterations = val

    @property
    def tol_cg(self):
        return self._tol_cg

    @tol_cg.setter
    def tol_cg(self, val):
        self.validator.validate("tol_cg", val)
        self._tol_cg = val

    @property
    def max_global_iterations(self):
        return self._max_global_iterations

    @max_global_iterations.setter
    def max_global_iterations(self, val):
        self.validator.validate("max_global_iterations", val)
        if val <= 0:
            raise ValueError("Invalid 'max_global_iterations' value.  Must be > 0.")
        self._max_global_iterations = val

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
            elif param == "model_norms":
                if "max_iterations" not in inputfile.data.keys():
                    if not np.all(np.r_[value] == 2):
                        self._override_default("max_iterations", 40)
            else:
                self._override_default(param, value)
