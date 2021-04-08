#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import json
import multiprocessing
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
        self.data = None
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
        self.model_norms = [2] * 4
        self.max_iterations = 10
        self.max_cg_iterations = 30
        self.tol_cg = 1e-4
        self.max_global_iterations = 100
        self.gradient_type = "total"
        self.initial_beta = None
        self.initial_beta_ratio = 1e2
        self.n_cpu = multiprocessing.cpu_count() / 2
        self.max_ram = 2
        self.padding_distance = [[0, 0]] * 3
        self.octree_levels_topo = [0, 1]
        self.octree_levels_obs = [5, 5]
        self.octree_levels_padding = [2, 2]
        self.alphas = [1] * 12
        self.reference_model = None
        self.starting_model = None
        self.lower_bound = -np.inf
        self.upper_bound = np.inf
        self.max_distance = np.inf
        self.max_chunk_size = 128
        self.chunk_by_rows = False
        self.output_tile_files = False
        self.no_data_value = 0
        self.parallelized = True
        self.out_group = None

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
    def data(self):
        return self._data

    @data.setter
    def data(self, val):
        if val is None:
            self._data = val
            return
        self.validator.validate("data", val)
        self._data = val

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

    @property
    def gradient_type(self):
        return self._gradient_type

    @gradient_type.setter
    def gradient_type(self, val):
        self.validator.validate("gradient_type", val)
        self._gradient_type = val

    @property
    def initial_beta(self):
        return self._initial_beta

    @initial_beta.setter
    def initial_beta(self, val):
        self.validator.validate("initial_beta", val)
        self._initial_beta = val

    @property
    def initial_beta_ratio(self):
        return self._initial_beta_ratio

    @initial_beta_ratio.setter
    def initial_beta_ratio(self, val):
        self.validator.validate("initial_beta_ratio", val)
        self._initial_beta_ratio = val

    @property
    def n_cpu(self):
        return self._n_cpu

    @n_cpu.setter
    def n_cpu(self, val):
        self.validator.validate("n_cpu", val)
        self._n_cpu = val

    @property
    def max_ram(self):
        return self._max_ram

    @max_ram.setter
    def max_ram(self, val):
        self.validator.validate("max_ram", val)
        self._max_ram = val

    @property
    def padding_distance(self):
        return self._padding_distance

    @padding_distance.setter
    def padding_distance(self, val):
        self.validator.validate("padding_distance", val)
        self._padding_distance = val

    @property
    def octree_levels_topo(self):
        return self._octree_levels_topo

    @octree_levels_topo.setter
    def octree_levels_topo(self, val):
        self.validator.validate("octree_levels_topo", val)
        self._octree_levels_topo = val

    @property
    def octree_levels_obs(self):
        return self._octree_levels_obs

    @octree_levels_obs.setter
    def octree_levels_obs(self, val):
        self.validator.validate("octree_levels_obs", val)
        self._octree_levels_obs = val

    @property
    def octree_levels_padding(self):
        return self._octree_levels_padding

    @octree_levels_padding.setter
    def octree_levels_padding(self, val):
        self.validator.validate("octree_levels_padding", val)
        self._octree_levels_padding = val

    @property
    def alphas(self):
        return self._alphas

    @alphas.setter
    def alphas(self, val):
        self.validator.validate("alphas", val)
        if len(val) == 4:
            val = val * 3
        if len(val) not in [4, 12]:
            msg = "Input parameter 'alphas' must be a list"
            msg += " of length 4 or 12"
            raise ValueError(msg)
        self._alphas = val

    @property
    def reference_model(self):
        return self._reference_model

    @reference_model.setter
    def reference_model(self, val):
        self.validator.validate("reference_model", val)
        self._reference_model = val

    @property
    def starting_model(self):
        return self._starting_model

    @starting_model.setter
    def starting_model(self, val):
        self.validator.validate("starting_model", val)
        self._starting_model = val

    @property
    def lower_bound(self):
        return self._lower_bound

    @lower_bound.setter
    def lower_bound(self, val):
        self.validator.validate("lower_bound", val)
        self._lower_bound = val

    @property
    def upper_bound(self):
        return self._upper_bound

    @upper_bound.setter
    def upper_bound(self, val):
        self.validator.validate("upper_bound", val)
        self._upper_bound = val

    @property
    def max_distance(self):
        return self._max_distance

    @max_distance.setter
    def max_distance(self, val):
        self.validator.validate("max_distance", val)
        self._max_distance = val

    @property
    def max_chunk_size(self):
        return self._max_chunk_size

    @max_chunk_size.setter
    def max_chunk_size(self, val):
        self.validator.validate("max_chunk_size", val)
        self._max_chunk_size = val

    @property
    def chunk_by_rows(self):
        return self._chunk_by_rows

    @chunk_by_rows.setter
    def chunk_by_rows(self, val):
        self.validator.validate("chunk_by_rows", val)
        self._chunk_by_rows = val

    @property
    def output_tile_files(self):
        return self._output_tile_files

    @output_tile_files.setter
    def output_tile_files(self, val):
        self.validator.validate("output_tile_files", val)
        self._output_tile_files = val

    @property
    def no_data_value(self):
        return self._no_data_value

    @no_data_value.setter
    def no_data_value(self, val):
        self.validator.validate("no_data_value", val)
        self._no_data_value = val

    @property
    def parallelized(self):
        return self._parallelized

    @parallelized.setter
    def parallelized(self, val):
        self.validator.validate("parallelized", val)
        self._parallelized = val

    @property
    def out_group(self):
        return self._out_group

    @out_group.setter
    def out_group(self, val):
        self.validator.validate("out_group", val)
        self._out_group = val

    def _override_default(self, param, value):
        """ Override parameter default value. """
        self.__setattr__(param, value)

    def _init_params(self, inputfile):
        """ Overrides default parameter values with input file values. """
        for param, value in inputfile.data.items():
            if param == "result_folder":
                op = create_relative_output_path(inputfile.workpath, value)
                self._override_default(param, op)
            elif param == "model_norms":
                if "max_iterations" not in inputfile.data.keys():
                    if not np.all(np.r_[value] == 2):
                        if "max_iterations" in inputfile.data.keys():
                            inputfile.data["max_iterations"] = 40
                        self._override_default("max_iterations", 40)
            else:
                self._override_default(param, value)
