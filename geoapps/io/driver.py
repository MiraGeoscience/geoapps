#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import json
import os
from typing import Any

import numpy as np

from .constants import default_ui_json, defaults, validations
from .validators import InputValidator

### Utils ###


def create_work_path(filepath: str) -> str:
    """ Returns absolute path of root directory of possible relative file path. """
    dsep = os.path.sep
    work_path = os.path.dirname(os.path.abspath(filepath)) + dsep
    return work_path


def create_relative_path(filepath: str, folder: str) -> str:
    """ Creates a relative path to folder from common path path elements in filepath. """
    dsep = os.path.sep
    work_path = create_work_path(filepath)
    root = os.path.commonprefix([folder, work_path])
    output_path = work_path + os.path.relpath(folder, root) + dsep
    return output_path


class InputFile:
    """
    Handles loading input file containing inversion parameters.

    Attributes
    ----------
    filepath : path to input file.
    workpath : path to working directory.
    data : input file contents parsed to dictionary.
    isloaded : True if load() method called to populate the 'data' attribute.

    Methods
    -------
    load()
        Loads and validates input file contents to dictionary.

    """

    _valid_extensions = ["json"]

    def __init__(self, filepath):
        self.filepath = filepath
        self.workpath = create_work_path(filepath)
        self.data = {}
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

    def write_ui_json(self):

        with open(self.filepath, "w") as f:
            json.dump(self._stringify(default_ui_json), f, indent=4)

    def read_ui_json(self):

        with open(self.filepath) as f:
            data = self._numify(json.load(f))
            self._ui_2_py(data)

        # validator = InputValidator(self.data)
        self.isloaded = True

    def _ui_2_py(self, d):
        for param, fields in d.items():
            if not isinstance(fields, dict):
                continue
            if "enabled" in fields.keys():
                if fields["enabled"] == False:
                    self.data[param] = None
                    continue
            if "visible" in fields.keys():
                if fields["visible"] == False:
                    self.data[param] = None
                    continue

            if param == "mesh":
                self.data[param] = fields["value"]
            else:
                if "meshType" in fields.keys():
                    continue
                elif "parent" in fields.keys():
                    if "isValue" in fields.keys():
                        fkey = "value" if fields["isValue"] else "property"
                    else:
                        fkey = "value"

                    self.data[param] = {d[fields["parent"]]["value"]: fields[fkey]}
                else:
                    self.data[param] = fields["value"]

    def _stringify(self, d):
        """ Convert inf, none, and list types to strings within a dictionary """

        # map [...] to "[...]"
        excl = ["choiceList", "meshType"]
        l2s = lambda k, v: str(v)[1:-1] if isinstance(v, list) & (k not in excl) else v
        n2s = lambda k, v: "" if v is None else v  # map None to ""

        def i2s(k, v):  # map np.inf to "inf"
            if not isinstance(v, (int, float)):
                return v
            else:
                return str(v) if not np.isfinite(v) else v

        for k, v in d.items():
            v = self._dict_mapper(k, v, [l2s, n2s, i2s])
            d[k] = v

        return d

    def _numify(self, d):
        """ Convert inf, none and list strings to numerical types within a dictionary """

        def s2l(k, v):  # map "[...]" to [...]
            if isinstance(v, str):
                if v in ["inf", "-inf"]:
                    return v
                else:
                    try:
                        return [float(n) for n in v.split(",")]
                    except ValueError:
                        return v
            else:
                return v

        s2n = lambda k, v: None if v == "" else v  # map "" to None
        s2i = (
            lambda k, v: float(v) if v in ["inf", "-inf"] else v
        )  # map "inf" to np.inf

        for k, v in d.items():
            v = self._dict_mapper(k, v, [s2l, s2n, s2i])
            d[k] = v

        return d

    def _dict_mapper(self, key, val, string_funcs):
        """ Recurses through nested dictionary and applies mapping funcs to all values """
        if isinstance(val, dict):
            for k, v in val.items():
                val[k] = self._dict_mapper(k, v, string_funcs)
            return val
        else:
            for f in string_funcs:
                val = f(key, val)
            return val

    # def load(self) -> None:
    #     """ Loads and validates input file contents to dictionary. """
    #     with open(self.filepath) as f:
    #         self.data = json.load(f)
    #         validator = InputValidator(self.data)
    #     self.isloaded = True


class Params:
    """
    Stores input parameters to drive an inversion.

    Attributes
    ----------
    validator : InputValidator
        class instance to handle parameter validation.
    inversion_type : str

    Constructors
    ------------
    from_ifile(ifile)
        Construct Params object from InputFile instance.
    from_path(path)
        Construct Params object from path to input file (wraps from_ifile constructor).

    """

    def __init__(
        self,
        inversion_type: str,
        workspace: str,
        out_group: str,
        data: dict,
        mesh: str,
        topography: dict,
        workpath: str = os.path.abspath("."),
    ):
        """
        Parameters
        ----------
        inversion_type : str
            Type of inversion. Must be one of: 'gravity', 'magnetics', 'mvi', 'mvic'.
        core_cell_size : int, float
            Core cell size for base mesh.
        workpath : str, optional
            Path to working folder (default is current directory).
        """
        self.validator = InputValidator()
        self.inversion_type = inversion_type
        self.workspace = workspace
        self.out_group = out_group
        self.data = data
        self.mesh = mesh
        self.topography = topography
        self.workpath = workpath
        self.inversion_style = defaults["inversion_style"]
        self.forward_only = defaults["forward_only"]
        self.inducing_field_aid = defaults["inducing_field_aid"]
        self.resolution = defaults["resolution"]
        self.window = defaults["window"]
        self.ignore_values = defaults["ignore_values"]
        self.detrend = defaults["detrend"]
        self.new_uncert = defaults["new_uncert"]
        self.output_geoh5 = defaults["output_geoh5"]
        self.receivers_offset = defaults["receivers_offset"]
        self.chi_factor = defaults["chi_factor"]
        self.model_norms = defaults["model_norms"]
        self.max_iterations = defaults["max_iterations"]
        self.max_cg_iterations = defaults["max_cg_iterations"]
        self.tol_cg = defaults["tol_cg"]
        self.max_global_iterations = defaults["max_global_iterations"]
        self.gradient_type = defaults["gradient_type"]
        self.initial_beta = defaults["initial_beta"]
        self.initial_beta_ratio = defaults["initial_beta_ratio"]
        self.n_cpu = defaults["n_cpu"]
        self.max_ram = defaults["max_ram"]
        self.core_cell_size = defaults["core_cell_size"]
        self.depth_core = defaults["depth_core"]
        self.padding_distance = defaults["padding_distance"]
        self.octree_levels_topo = defaults["octree_levels_topo"]
        self.octree_levels_obs = defaults["octree_levels_obs"]
        self.octree_levels_padding = defaults["octree_levels_padding"]
        self.alphas = defaults["alphas"]
        self.reference_model = defaults["reference_model"]
        self.reference_inclination = defaults["reference_inclination"]
        self.reference_declination = defaults["reference_declination"]
        self.starting_model = defaults["starting_model"]
        self.starting_inclination = defaults["starting_inclination"]
        self.starting_declination = defaults["starting_declination"]
        self.lower_bound = defaults["lower_bound"]
        self.upper_bound = defaults["upper_bound"]
        self.max_distance = defaults["max_distance"]
        self.max_chunk_size = defaults["max_chunk_size"]
        self.chunk_by_rows = defaults["chunk_by_rows"]
        self.output_tile_files = defaults["output_tile_files"]
        self.no_data_value = defaults["no_data_value"]
        self.parallelized = defaults["parallelized"]

    @classmethod
    def from_ifile(cls, ifile: InputFile) -> None:
        """Construct Params object from InputFile instance.

        Parameters
        ----------
        ifile : InputFile
            class instance to handle loading input file
        """
        if not ifile.isloaded:
            ifile.load()
        inversion_type = ifile.data["inversion_type"]
        workspace = ifile.data["workspace"]
        out_group = ifile.data["out_group"]
        data = ifile.data["data"]
        mesh = ifile.data["mesh"]
        topography = ifile.data["topography"]
        p = cls(
            inversion_type, workspace, out_group, data, mesh, topography, ifile.workpath
        )
        p._init_params(ifile)
        return p

    @classmethod
    def from_path(cls, filepath: str) -> None:
        """Construct Params object from path to input file.

        Parameters
        ----------
        filepath : str
            path to input file.
        """
        ifile = InputFile(filepath)
        p = cls.from_ifile(ifile)
        return p

    @property
    def inversion_type(self):
        return self._inversion_type

    @inversion_type.setter
    def inversion_type(self, val):
        if val is None:
            self._inversion_type = val
            return
        p = "inversion_type"
        self.validator.validate(p, val, validations[p])
        self._inversion_type = val

    @property
    def core_cell_size(self):
        return self._core_cell_size

    @core_cell_size.setter
    def core_cell_size(self, val):
        p = "core_cell_size"
        self.validator.validate(p, val, validations[p])
        self._core_cell_size = val

    @property
    def inversion_style(self):
        return self._inversion_style

    @inversion_style.setter
    def inversion_style(self, val):
        p = "inversion_style"
        self.validator.validate(p, val, validations[p])
        self._inversion_style = val

    @property
    def forward_only(self):
        return self._forward_only

    @forward_only.setter
    def forward_only(self, val):
        p = "forward_only"
        self.validator.validate(p, val, validations[p])
        self._forward_only = val

    @property
    def inducing_field_aid(self):
        return self._inducing_field_aid

    @inducing_field_aid.setter
    def inducing_field_aid(self, val):
        if val is None:
            self._inducing_field_aid = val
            return
        p = "inducing_field_aid"
        self.validator.validate(p, val, validations[p])
        if val[0] <= 0:
            raise ValueError("inducing_field_aid[0] must be greater than 0.")
        self._inducing_field_aid = np.array(val)

    @property
    def resolution(self):
        return self._resolution

    @resolution.setter
    def resolution(self, val):
        p = "resolution"
        self.validator.validate(p, val, validations[p])
        self._resolution = val

    @property
    def window(self):
        return self._window

    @window.setter
    def window(self, val):
        if val is None:
            self._window = val
            return
        p = "window"
        self.validator.validate(p, val, validations[p])
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
        p = "data"
        self.validator.validate(p, val, validations[p])
        if "name" not in val.keys():
            raise KeyError("Missing required key: 'name', for input parameter 'data'.")
        self._data = val

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

    @property
    def ignore_values(self):
        return self._ignore_values

    @ignore_values.setter
    def ignore_values(self, val):
        if val is None:
            self._ignore_values = val
            return
        p = "ignore_values"
        self.validator.validate(p, val, validations[p])
        self._ignore_values = val

    @property
    def detrend(self):
        return self._detrend

    @detrend.setter
    def detrend(self, val):
        if val is None:
            self._detrend = val
            return
        p = "detrend"
        self.validator.validate(p, val, validations[p])
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
        p = "data_file"
        self.validator.validate(p, val, validations[p])
        self._data_file = val

    @property
    def new_uncert(self):
        return self._new_uncert

    @new_uncert.setter
    def new_uncert(self, val):
        if val is None:
            self._new_uncert = val
            return
        p = "new_uncert"
        self.validator.validate(p, val, validations[p])
        if (val[0] < 0) | (val[0] > 1):
            msg = "Uncertainty percent (new_uncert[0]) must be between 0 and 1."
            raise ValueError(msg)
        if val[1] < 0:
            msg = "Uncertainty floor (new_uncert[1]) must be greater than 0."
            raise ValueError(msg)
        self._new_uncert = val

    @property
    def mesh(self):
        return self._mesh

    @mesh.setter
    def mesh(self, val):
        if val is None:
            self._mesh = val
            return
        p = "mesh"
        self.validator.validate(p, val, validations[p])
        self._mesh = val

    @property
    def save_to_geoh5(self):
        return self._save_to_geoh5

    @save_to_geoh5.setter
    def save_to_geoh5(self, val):
        p = "save_to_geoh5"
        self.validator.validate(p, val, validations[p])
        self._save_to_geoh5 = val

    @property
    def topography(self):
        return self._topography

    @topography.setter
    def topography(self, val):
        if val is None:
            self._topography = val
            return
        p = "topography"
        self.validator.validate(p, val, validations[p])
        self._topography = val

    @property
    def receivers_offset(self):
        return self._receivers_offset

    @receivers_offset.setter
    def receivers_offset(self, val):
        if val is None:
            self._receivers_offset = val
            return
        p = "receivers_offset"
        self.validator.validate(p, val, validations[p])
        self._receivers_offset = val

    @property
    def chi_factor(self):
        return self._chi_factor

    @chi_factor.setter
    def chi_factor(self, val):
        p = "chi_factor"
        self.validator.validate(p, val, validations[p])
        if val <= 0:
            raise ValueError("Invalid chi_factor. Must be between 0 and 1.")
        self._chi_factor = val

    @property
    def model_norms(self):
        return self._model_norms

    @model_norms.setter
    def model_norms(self, val):
        p = "model_norms"
        self.validator.validate(p, val, validations[p])
        if len(val) % 4 != 0:
            raise ValueError("Invalid 'model_norms' length.  Must be a multiple of 4.")
        self._model_norms = val

    @property
    def max_iterations(self):
        return self._max_iterations

    @max_iterations.setter
    def max_iterations(self, val):
        p = "max_iterations"
        self.validator.validate(p, val, validations[p])
        if val <= 0:
            raise ValueError("Invalid 'max_iterations' value.  Must be > 0.")
        self._max_iterations = val

    @property
    def max_cg_iterations(self):
        return self._max_cg_iterations

    @max_cg_iterations.setter
    def max_cg_iterations(self, val):
        p = "max_cg_iterations"
        self.validator.validate(p, val, validations[p])
        if val <= 0:
            raise ValueError("Invalid 'max_cg_iterations' value.  Must be > 0.")
        self._max_cg_iterations = val

    @property
    def tol_cg(self):
        return self._tol_cg

    @tol_cg.setter
    def tol_cg(self, val):
        p = "tol_cg"
        self.validator.validate(p, val, validations[p])
        self._tol_cg = val

    @property
    def max_global_iterations(self):
        return self._max_global_iterations

    @max_global_iterations.setter
    def max_global_iterations(self, val):
        p = "max_global_iterations"
        self.validator.validate(p, val, validations[p])
        if val <= 0:
            raise ValueError("Invalid 'max_global_iterations' value.  Must be > 0.")
        self._max_global_iterations = val

    @property
    def gradient_type(self):
        return self._gradient_type

    @gradient_type.setter
    def gradient_type(self, val):
        p = "gradient_type"
        self.validator.validate(p, val, validations[p])
        self._gradient_type = val

    @property
    def initial_beta(self):
        return self._initial_beta

    @initial_beta.setter
    def initial_beta(self, val):
        if val is None:
            self._initial_beta = val
            return
        p = "initial_beta"
        self.validator.validate(p, val, validations[p])
        self._initial_beta = val

    @property
    def initial_beta_ratio(self):
        return self._initial_beta_ratio

    @initial_beta_ratio.setter
    def initial_beta_ratio(self, val):
        p = "initial_beta_ratio"
        self.validator.validate(p, val, validations[p])
        self._initial_beta_ratio = val

    @property
    def n_cpu(self):
        return self._n_cpu

    @n_cpu.setter
    def n_cpu(self, val):
        if val is None:
            self._n_cpu = val
            return
        p = "n_cpu"
        self.validator.validate(p, val, validations[p])
        self._n_cpu = val

    @property
    def max_ram(self):
        return self._max_ram

    @max_ram.setter
    def max_ram(self, val):
        p = "max_ram"
        self.validator.validate(p, val, validations[p])
        self._max_ram = val

    @property
    def depth_core(self):
        return self._depth_core

    @depth_core.setter
    def depth_core(self, val):
        if val is None:
            self._depth_core = val
            return
        p = "depth_core"
        self.validator.validate(p, val, validations[p])
        self._depth_core = val

    @property
    def padding_distance(self):
        return self._padding_distance

    @padding_distance.setter
    def padding_distance(self, val):
        p = "padding_distance"
        self.validator.validate(p, val, validations[p])
        self._padding_distance = val

    @property
    def octree_levels_topo(self):
        return self._octree_levels_topo

    @octree_levels_topo.setter
    def octree_levels_topo(self, val):
        p = "octree_levels_topo"
        self.validator.validate(p, val, validations[p])
        self._octree_levels_topo = val

    @property
    def octree_levels_obs(self):
        return self._octree_levels_obs

    @octree_levels_obs.setter
    def octree_levels_obs(self, val):
        p = "octree_levels_obs"
        self.validator.validate(p, val, validations[p])
        self._octree_levels_obs = val

    @property
    def octree_levels_padding(self):
        return self._octree_levels_padding

    @octree_levels_padding.setter
    def octree_levels_padding(self, val):
        p = "octree_levels_padding"
        self.validator.validate(p, val, validations[p])
        self._octree_levels_padding = val

    @property
    def alphas(self):
        return self._alphas

    @alphas.setter
    def alphas(self, val):
        p = "alphas"
        self.validator.validate(p, val, validations[p])
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
        if val is None:
            self._reference_model = val
            return
        p = "reference_model"
        self.validator.validate(p, val, validations[p])
        self._reference_model = val

    @property
    def reference_inclination(self):
        return self._reference_inclination

    @reference_inclination.setter
    def reference_inclination(self, val):
        if val is None:
            self._reference_inclination = val
            return
        p = "reference_inclination"
        self.validator.validate(p, val, validations[p])
        self._reference_inclination = val

    @property
    def reference_declination(self):
        return self._reference_declination

    @reference_declination.setter
    def reference_declination(self, val):
        if val is None:
            self._reference_declination = val
            return
        p = "reference_declination"
        self.validator.validate(p, val, validations[p])
        self._reference_declination = val

    @property
    def starting_model(self):
        return self._starting_model

    @starting_model.setter
    def starting_model(self, val):
        if val is None:
            self._starting_model = val
            return
        p = "starting_model"
        self.validator.validate(p, val, validations[p])
        self._starting_model = val

    @property
    def starting_inclination(self):
        return self._starting_inclination

    @starting_inclination.setter
    def starting_inclination(self, val):
        if val is None:
            self._starting_inclination = val
            return
        p = "starting_inclination"
        self.validator.validate(p, val, validations[p])
        self._starting_inclination = val

    @property
    def starting_declination(self):
        return self._starting_declination

    @starting_declination.setter
    def starting_declination(self, val):
        if val is None:
            self._starting_declination = val
            return
        p = "starting_declination"
        self.validator.validate(p, val, validations[p])
        self._starting_declination = val

    @property
    def lower_bound(self):
        return self._lower_bound

    @lower_bound.setter
    def lower_bound(self, val):
        p = "lower_bound"
        self.validator.validate(p, val, validations[p])
        self._lower_bound = val

    @property
    def upper_bound(self):
        return self._upper_bound

    @upper_bound.setter
    def upper_bound(self, val):
        p = "upper_bound"
        self.validator.validate(p, val, validations[p])
        self._upper_bound = val

    @property
    def max_distance(self):
        return self._max_distance

    @max_distance.setter
    def max_distance(self, val):
        p = "max_distance"
        self.validator.validate(p, val, validations[p])
        self._max_distance = val

    @property
    def max_chunk_size(self):
        return self._max_chunk_size

    @max_chunk_size.setter
    def max_chunk_size(self, val):
        p = "max_chunk_size"
        self.validator.validate(p, val, validations[p])
        self._max_chunk_size = val

    @property
    def chunk_by_rows(self):
        return self._chunk_by_rows

    @chunk_by_rows.setter
    def chunk_by_rows(self, val):
        p = "chunk_by_rows"
        self.validator.validate(p, val, validations[p])
        self._chunk_by_rows = val

    @property
    def output_tile_files(self):
        return self._output_tile_files

    @output_tile_files.setter
    def output_tile_files(self, val):
        p = "output_tile_files"
        self.validator.validate(p, val, validations[p])
        self._output_tile_files = val

    @property
    def no_data_value(self):
        return self._no_data_value

    @no_data_value.setter
    def no_data_value(self, val):
        p = "no_data_value"
        self.validator.validate(p, val, validations[p])
        self._no_data_value = val

    @property
    def parallelized(self):
        return self._parallelized

    @parallelized.setter
    def parallelized(self, val):
        p = "parallelized"
        self.validator.validate(p, val, validations[p])
        self._parallelized = val

    @property
    def out_group(self):
        return self._out_group

    @out_group.setter
    def out_group(self, val):
        if val is None:
            self._out_group = val
            return
        p = "out_group"
        self.validator.validate(p, val, validations[p])
        self._out_group = val

    def _override_default(self, param: str, value: Any) -> None:
        """ Override parameter default value. """
        self.__setattr__(param, value)

    def _init_params(self, inputfile: InputFile) -> None:
        """ Overrides default parameter values with input file values. """
        for param, value in inputfile.data.items():
            if param == "model_norms":
                if "max_iterations" not in inputfile.data.keys():
                    if not np.all(np.r_[value] == 2):
                        if "max_iterations" in inputfile.data.keys():
                            inputfile.data["max_iterations"] = 40
                        self._override_default("max_iterations", 40)
                self._override_default(param, value)
            else:
                self._override_default(param, value)
