#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import os
from copy import deepcopy
from uuid import UUID

import numpy as np
from geoh5py.groups import ContainerGroup
from geoh5py.workspace import Workspace

from ..input_file import InputFile
from ..params import Params
from ..validators import InputValidator
from .constants import required_parameters, validations


class InversionParams(Params):

    _ga_group = None

    def __init__(
        self, input_file=None, default=True, validate=True, validator_opts={}, **kwargs
    ):

        self.forward_only: bool = None
        self.topography_object: UUID = None
        self.topography: UUID | float = None
        self.data_object: UUID = None
        self.starting_model_object: UUID = None
        self.starting_model: UUID | float = None
        self.tile_spatial = None
        self.z_from_topo: bool = None
        self.receivers_radar_drape = None
        self.receivers_offset_x: float = None
        self.receivers_offset_y: float = None
        self.receivers_offset_z: float = None
        self.gps_receivers_offset = None
        self.ignore_values: str = None
        self.resolution: float = None
        self.detrend_order: int = None
        self.detrend_type: str = None
        self.max_chunk_size: int = None
        self.chunk_by_rows: bool = None
        self.output_tile_files: bool = None
        self.mesh = None
        self.u_cell_size: float = None
        self.v_cell_size: float = None
        self.w_cell_size: float = None
        self.octree_levels_topo: list[int] = None
        self.octree_levels_obs: list[int] = None
        self.depth_core: float = None
        self.max_distance: float = None
        self.horizontal_padding: float = None
        self.vertical_padding: float = None
        self.window_azimuth: float = None
        self.window_center_x: float = None
        self.window_center_y: float = None
        self.window_height: float = None
        self.window_width: float = None
        self.inversion_style: str = None
        self.chi_factor: float = None
        self.sens_wts_threshold: float = None
        self.every_iteration_bool: bool = None
        self.f_min_change: float = None
        self.minGNiter: float = None
        self.beta_tol: float = None
        self.prctile: float = None
        self.coolingRate: float = None
        self.coolEps_q: bool = None
        self.coolEpsFact: float = None
        self.beta_search: bool = None
        self.starting_chi_factor: float = None
        self.max_iterations: int = None
        self.max_line_search_iterations: int = None
        self.max_cg_iterations: int = None
        self.max_global_iterations: int = None
        self.initial_beta: float = None
        self.initial_beta_ratio: float = None
        self.tol_cg: float = None
        self.alpha_s: float = None
        self.alpha_x: float = None
        self.alpha_y: float = None
        self.alpha_z: float = None
        self.s_norm: float = None
        self.x_norm: float = None
        self.y_norm: float = None
        self.z_norm: float = None
        self.reference_model_object: UUID = None
        self.reference_model = None
        self.gradient_type: str = None
        self.lower_bound_object: UUID = None
        self.lower_bound = None
        self.upper_bound_object: UUID = None
        self.upper_bound = None
        self.parallelized: bool = None
        self.n_cpu: int = None
        self.max_ram: float = None
        self.out_group = None
        self.no_data_value: float = None
        self.monitoring_directory: str = None
        self.workspace_geoh5: str = None
        self.geoh5 = None
        self.run_command: str = None
        self.run_command_boolean: bool = None
        self.conda_environment: str = None
        self.conda_environment_boolean: bool = None
        self.distributed_workers = None
        super().__init__(input_file, default, validate, validator_opts, **kwargs)

        self._initialize(kwargs)

    def _initialize(self, params_dict):

        # Collect params_dict from superposition of kwargs onto input_file.data
        # and determine forward_only state.
        fwd = False
        if self.input_file:
            params_dict = dict(self.input_file.data, **params_dict)
        if "forward_only" in params_dict.keys():
            fwd = params_dict["forward_only"]

        # Use forward_only state to determine defaults and default_ui_json.
        self.defaults = self._forward_defaults if fwd else self._inversion_defaults
        self.default_ui_json.update(
            self.forward_ui_json if fwd else self.inversion_ui_json
        )
        self.default_ui_json = {
            k: self.default_ui_json[k] for k in self.defaults.keys()
        }
        self.param_names = list(self.defaults.keys())

        # Superimpose params_dict onto defaults.
        if self.default:
            params_dict = dict(self.defaults, **params_dict)

        # Validate.
        if self.validate:
            self.geoh5 = params_dict["geoh5"]
            self.associations = self.get_associations(params_dict)
            self.validator: InputValidator = InputValidator(
                self._required_parameters,
                self._validations,
                self.geoh5,
                **self.validator_opts,
            )
            self.validator.validate_chunk(params_dict, self.associations)

        # Set params attributes from validated input.
        self.update(params_dict, validate=False)

    def uncertainty(self, component: str) -> float:
        """Returns uncertainty for chosen data component."""
        return getattr(self, "_".join([component, "uncertainty"]), None)

    def channel(self, component: str) -> UUID:
        """Returns channel uuid for chosen data component."""
        return getattr(self, "_".join([component, "channel"]), None)

    def cell_size(self):
        """Returns core cell size in all 3 dimensions."""
        return [self.u_cell_size, self.v_cell_size, self.w_cell_size]

    def padding_distance(self):
        """Returns padding distance in all 3 dimensions."""
        return [
            self.padding_distance_x,
            self.padding_distance_y,
            self.padding_distance_z,
        ]

    def components(self) -> list[str]:
        """Retrieve component names used to index channel and uncertainty data."""
        comps = []
        channels = np.unique(
            [
                k.lstrip("_").split("_")[0]
                for k in self.__dict__.keys()
                if "channel" in k
            ]
        )
        for c in channels:
            use_ch = False
            if getattr(self, f"{c}_channel", None) is not None:
                use_ch = True
            if getattr(self, f"{c}_channel_bool", None) is True:
                use_ch = True
            if use_ch:
                comps.append(c)

        return comps

    def window(self) -> dict[str, float]:
        """Returns window dictionary"""
        win = {
            "azimuth": self.window_azimuth,
            "center_x": self.window_center_x,
            "center_y": self.window_center_y,
            "width": self.window_width,
            "height": self.window_height,
            "center": [self.window_center_x, self.window_center_y],
            "size": [self.window_width, self.window_height],
        }
        check_keys = ["center_x", "center_y", "width", "height"]
        no_data = all([v is None for k, v in win.items() if k in check_keys])
        return None if no_data else win

    def offset(self) -> tuple[list[float], UUID]:
        """Returns offset components as list and drape data."""
        offsets = [
            self.receivers_offset_x,
            self.receivers_offset_y,
            self.receivers_offset_z,
        ]
        is_offset = any([(k != 0) for k in offsets])
        offsets = offsets if is_offset else None
        r = self.receivers_radar_drape
        if isinstance(r, (str, UUID)):
            r = UUID(r) if isinstance(r, str) else r
            radar = self.geoh5.get_entity(r)
            radar = radar[0].values if radar else None
        else:
            radar = None
        return offsets, radar

    def model_norms(self) -> list[float]:
        """Returns model norm components as a list."""
        return [
            self.s_norm,
            self.x_norm,
            self.y_norm,
            self.z_norm,
        ]

    @property
    def forward_only(self):
        return self._forward_only

    @forward_only.setter
    def forward_only(self, val):
        if val is None:
            self._forward_only = val
            return

        p = "forward_only"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._forward_only = val

    @property
    def topography_object(self):
        return self._topography_object

    @topography_object.setter
    def topography_object(self, val):
        if val is None:
            self._topography_object = val
            return
        p = "topography_object"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._topography_object = UUID(val) if isinstance(val, str) else val

    @property
    def topography(self):
        return self._topography

    @topography.setter
    def topography(self, val):
        if val is None:
            self._topography = val
            return
        p = "topography"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._topography = UUID(val) if isinstance(val, str) else val

    @property
    def data_object(self):
        return self._data_object

    @data_object.setter
    def data_object(self, val):
        if val is None:
            self._data_object = val
            return
        p = "data_object"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._data_object = UUID(val) if isinstance(val, str) else val

    @property
    def starting_model_object(self):
        return self._starting_model_object

    @starting_model_object.setter
    def starting_model_object(self, val):
        if val is None:
            self._starting_model_object = val
            return
        p = "starting_model_object"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._starting_model_object = UUID(val) if isinstance(val, str) else val

    @property
    def starting_model(self):
        return self._starting_model

    @starting_model.setter
    def starting_model(self, val):
        if val is None:
            self._starting_model = val
            return
        p = "starting_model"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._starting_model = UUID(val) if isinstance(val, str) else val

    @property
    def tile_spatial(self):
        return self._tile_spatial

    @tile_spatial.setter
    def tile_spatial(self, val):
        if val is None:
            self._tile_spatial = val
            return
        p = "tile_spatial"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._tile_spatial = UUID(val) if isinstance(val, str) else val

    @property
    def z_from_topo(self):
        return self._z_from_topo

    @z_from_topo.setter
    def z_from_topo(self, val):
        if val is None:
            self._z_from_topo = val
            return
        p = "z_from_topo"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._z_from_topo = val

    @property
    def receivers_radar_drape(self):
        return self._receivers_radar_drape

    @receivers_radar_drape.setter
    def receivers_radar_drape(self, val):
        if val is None:
            self._receivers_radar_drape = val
            return
        p = "receivers_radar_drape"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._receivers_radar_drape = UUID(val) if isinstance(val, str) else val

    @property
    def receivers_offset_x(self):
        return self._receivers_offset_x

    @receivers_offset_x.setter
    def receivers_offset_x(self, val):
        if val is None:
            self._receivers_offset_x = val
            return
        p = "receivers_offset_x"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._receivers_offset_x = val

    @property
    def receivers_offset_y(self):
        return self._receivers_offset_y

    @receivers_offset_y.setter
    def receivers_offset_y(self, val):
        if val is None:
            self._receivers_offset_y = val
            return
        p = "receivers_offset_y"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._receivers_offset_y = val

    @property
    def receivers_offset_z(self):
        return self._receivers_offset_z

    @receivers_offset_z.setter
    def receivers_offset_z(self, val):
        if val is None:
            self._receivers_offset_z = val
            return
        p = "receivers_offset_z"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._receivers_offset_z = val

    @property
    def gps_receivers_offset(self):
        return self._gps_receivers_offset

    @gps_receivers_offset.setter
    def gps_receivers_offset(self, val):
        if val is None:
            self._gps_receivers_offset = val
            return
        p = "gps_receivers_offset"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._gps_receivers_offset = UUID(val) if isinstance(val, str) else val

    @property
    def ignore_values(self):
        return self._ignore_values

    @ignore_values.setter
    def ignore_values(self, val):
        if val is None:
            self._ignore_values = val
            return
        p = "ignore_values"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._ignore_values = val

    @property
    def resolution(self):
        return self._resolution

    @resolution.setter
    def resolution(self, val):
        if val is None:
            self._resolution = val
            return
        p = "resolution"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._resolution = val

    @property
    def detrend_order(self):
        return self._detrend_order

    @detrend_order.setter
    def detrend_order(self, val):
        if val is None:
            self._detrend_order = val
            return
        p = "detrend_order"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._detrend_order = val

    @property
    def detrend_type(self):
        return self._detrend_type

    @detrend_type.setter
    def detrend_type(self, val):
        if val is None:
            self._detrend_type = val
            return
        p = "detrend_type"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._detrend_type = val

    @property
    def max_chunk_size(self):
        return self._max_chunk_size

    @max_chunk_size.setter
    def max_chunk_size(self, val):
        if val is None:
            self._max_chunk_size = val
            return
        p = "max_chunk_size"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._max_chunk_size = val

    @property
    def chunk_by_rows(self):
        return self._chunk_by_rows

    @chunk_by_rows.setter
    def chunk_by_rows(self, val):
        if val is None:
            self._chunk_by_rows = val
            return
        p = "chunk_by_rows"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._chunk_by_rows = val

    @property
    def output_tile_files(self):
        return self._output_tile_files

    @output_tile_files.setter
    def output_tile_files(self, val):
        if val is None:
            self._output_tile_files = val
            return
        p = "output_tile_files"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._output_tile_files = val

    @property
    def mesh(self):
        return self._mesh

    @mesh.setter
    def mesh(self, val):
        if val is None:
            self._mesh = val
            return
        p = "mesh"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._mesh = UUID(val) if isinstance(val, str) else val

    @property
    def u_cell_size(self):
        return self._u_cell_size

    @u_cell_size.setter
    def u_cell_size(self, val):
        if val is None:
            self._u_cell_size = val
            return
        p = "u_cell_size"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._u_cell_size = val

    @property
    def v_cell_size(self):
        return self._v_cell_size

    @v_cell_size.setter
    def v_cell_size(self, val):
        if val is None:
            self._v_cell_size = val
            return
        p = "v_cell_size"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._v_cell_size = val

    @property
    def w_cell_size(self):
        return self._w_cell_size

    @w_cell_size.setter
    def w_cell_size(self, val):
        if val is None:
            self._w_cell_size = val
            return
        p = "w_cell_size"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._w_cell_size = val

    @property
    def octree_levels_topo(self):
        return self._octree_levels_topo

    @octree_levels_topo.setter
    def octree_levels_topo(self, val):
        if val is None:
            self._octree_levels_topo = val
            return
        p = "octree_levels_topo"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._octree_levels_topo = val

    @property
    def octree_levels_obs(self):
        return self._octree_levels_obs

    @octree_levels_obs.setter
    def octree_levels_obs(self, val):
        if val is None:
            self._octree_levels_obs = val
            return
        p = "octree_levels_obs"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._octree_levels_obs = val

    @property
    def depth_core(self):
        return self._depth_core

    @depth_core.setter
    def depth_core(self, val):
        if val is None:
            self._depth_core = val
            return
        p = "depth_core"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._depth_core = val

    @property
    def max_distance(self):
        return self._max_distance

    @max_distance.setter
    def max_distance(self, val):
        if val is None:
            self._max_distance = val
            return
        p = "max_distance"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._max_distance = val

    @property
    def horizontal_padding(self):
        return self._horizontal_padding

    @horizontal_padding.setter
    def horizontal_padding(self, val):
        if val is None:
            self._horizontal_padding = val
            return
        p = "horizontal_padding"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._horizontal_padding = val

    @property
    def vertical_padding(self):
        return self._vertical_padding

    @vertical_padding.setter
    def vertical_padding(self, val):
        if val is None:
            self._vertical_padding = val
            return
        p = "vertical_padding"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._vertical_padding = val

    @property
    def window_center_x(self):
        return self._window_center_x

    @window_center_x.setter
    def window_center_x(self, val):
        if val is None:
            self._window_center_x = val
            return
        p = "window_center_x"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._window_center_x = val

    @property
    def window_center_y(self):
        return self._window_center_y

    @window_center_y.setter
    def window_center_y(self, val):
        if val is None:
            self._window_center_y = val
            return
        p = "window_center_y"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._window_center_y = val

    @property
    def window_width(self):
        return self._window_width

    @window_width.setter
    def window_width(self, val):
        if val is None:
            self._window_width = val
            return
        p = "window_width"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._window_width = val

    @property
    def window_height(self):
        return self._window_height

    @window_height.setter
    def window_height(self, val):
        if val is None:
            self._window_height = val
            return
        p = "window_height"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._window_height = val

    @property
    def window_azimuth(self):
        return self._window_azimuth

    @window_azimuth.setter
    def window_azimuth(self, val):
        if val is None:
            self._window_azimuth = val
            return
        p = "window_azimuth"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._window_azimuth = val

    @property
    def inversion_style(self):
        return self._inversion_style

    @inversion_style.setter
    def inversion_style(self, val):
        if val is None:
            self._inversion_style = val
            return
        p = "inversion_style"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._inversion_style = val

    @property
    def chi_factor(self):
        return self._chi_factor

    @chi_factor.setter
    def chi_factor(self, val):
        if val is None:
            self._chi_factor = val
            return
        p = "chi_factor"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._chi_factor = val

    @property
    def sens_wts_threshold(self):
        return self._sens_wts_threshold

    @sens_wts_threshold.setter
    def sens_wts_threshold(self, val):
        if val is None:
            self._sens_wts_threshold = val
            return
        p = "sens_wts_threshold"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._sens_wts_threshold = val

    @property
    def every_iteration_bool(self):
        return self._every_iteration_bool

    @every_iteration_bool.setter
    def every_iteration_bool(self, val):
        if val is None:
            self._every_iteration_bool = val
            return
        p = "every_iteration_bool"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._every_iteration_bool = val

    @property
    def f_min_change(self):
        return self._f_min_change

    @f_min_change.setter
    def f_min_change(self, val):
        if val is None:
            self._f_min_change = val
            return
        p = "f_min_change"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._f_min_change = val

    @property
    def minGNiter(self):
        return self._minGNiter

    @minGNiter.setter
    def minGNiter(self, val):
        if val is None:
            self._minGNiter = val
            return
        p = "minGNiter"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._minGNiter = val

    @property
    def beta_tol(self):
        return self._beta_tol

    @beta_tol.setter
    def beta_tol(self, val):
        if val is None:
            self._beta_tol = val
            return
        p = "beta_tol"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._beta_tol = val

    @property
    def prctile(self):
        return self._prctile

    @prctile.setter
    def prctile(self, val):
        if val is None:
            self._prctile = val
            return
        p = "prctile"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._prctile = val

    @property
    def coolingRate(self):
        return self._coolingRate

    @coolingRate.setter
    def coolingRate(self, val):
        if val is None:
            self._coolingRate = val
            return
        p = "coolingRate"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._coolingRate = val

    @property
    def coolEps_q(self):
        return self._coolEps_q

    @coolEps_q.setter
    def coolEps_q(self, val):
        if val is None:
            self._coolEps_q = val
            return
        p = "coolEps_q"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._coolEps_q = val

    @property
    def coolEpsFact(self):
        return self._coolEpsFact

    @coolEpsFact.setter
    def coolEpsFact(self, val):
        if val is None:
            self._coolEpsFact = val
            return
        p = "coolEpsFact"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._coolEpsFact = val

    @property
    def beta_search(self):
        return self._beta_search

    @beta_search.setter
    def beta_search(self, val):
        if val is None:
            self._beta_search = val
            return
        p = "beta_search"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._beta_search = val

    @property
    def starting_chi_factor(self):
        return self._starting_chi_factor

    @starting_chi_factor.setter
    def starting_chi_factor(self, val):
        if val is None:
            self._starting_chi_factor = val
            return
        p = "starting_chi_factor"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._starting_chi_factor = val

    @property
    def max_iterations(self):
        return self._max_iterations

    @max_iterations.setter
    def max_iterations(self, val):
        if val is None:
            self._max_iterations = val
            return
        p = "max_iterations"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._max_iterations = val

    @property
    def max_line_search_iterations(self):
        return self._max_line_search_iterations

    @max_line_search_iterations.setter
    def max_line_search_iterations(self, val):
        if val is None:
            self._max_line_search_iterations = val
            return
        p = "max_line_search_iterations"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._max_line_search_iterations = val

    @property
    def max_cg_iterations(self):
        return self._max_cg_iterations

    @max_cg_iterations.setter
    def max_cg_iterations(self, val):
        if val is None:
            self._max_cg_iterations = val
            return
        p = "max_cg_iterations"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._max_cg_iterations = val

    @property
    def max_global_iterations(self):
        return self._max_global_iterations

    @max_global_iterations.setter
    def max_global_iterations(self, val):
        if val is None:
            self._max_global_iterations = val
            return
        p = "max_global_iterations"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._max_global_iterations = val

    @property
    def initial_beta(self):
        return self._initial_beta

    @initial_beta.setter
    def initial_beta(self, val):
        if val is None:
            self._initial_beta = val
            return
        p = "initial_beta"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._initial_beta = val

    @property
    def initial_beta_ratio(self):
        return self._initial_beta_ratio

    @initial_beta_ratio.setter
    def initial_beta_ratio(self, val):
        if val is None:
            self._initial_beta_ratio = val
            return
        p = "initial_beta_ratio"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._initial_beta_ratio = val

    @property
    def tol_cg(self):
        return self._tol_cg

    @tol_cg.setter
    def tol_cg(self, val):
        if val is None:
            self._tol_cg = val
            return
        p = "tol_cg"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._tol_cg = val

    @property
    def alpha_s(self):
        return self._alpha_s

    @alpha_s.setter
    def alpha_s(self, val):
        if val is None:
            self._alpha_s = val
            return
        p = "alpha_s"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._alpha_s = val

    @property
    def alpha_x(self):
        return self._alpha_x

    @alpha_x.setter
    def alpha_x(self, val):
        if val is None:
            self._alpha_x = val
            return
        p = "alpha_x"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._alpha_x = val

    @property
    def alpha_y(self):
        return self._alpha_y

    @alpha_y.setter
    def alpha_y(self, val):
        if val is None:
            self._alpha_y = val
            return
        p = "alpha_y"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._alpha_y = val

    @property
    def alpha_z(self):
        return self._alpha_z

    @alpha_z.setter
    def alpha_z(self, val):
        if val is None:
            self._alpha_z = val
            return
        p = "alpha_z"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._alpha_z = val

    @property
    def s_norm(self):
        return self._s_norm

    @s_norm.setter
    def s_norm(self, val):
        if val is None:
            self._s_norm = val
            return
        p = "s_norm"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._s_norm = val

    @property
    def x_norm(self):
        return self._x_norm

    @x_norm.setter
    def x_norm(self, val):
        if val is None:
            self._x_norm = val
            return
        p = "x_norm"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._x_norm = val

    @property
    def y_norm(self):
        return self._y_norm

    @y_norm.setter
    def y_norm(self, val):
        if val is None:
            self._y_norm = val
            return
        p = "y_norm"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._y_norm = val

    @property
    def z_norm(self):
        return self._z_norm

    @z_norm.setter
    def z_norm(self, val):
        if val is None:
            self._z_norm = val
            return
        p = "z_norm"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._z_norm = val

    @property
    def reference_model_object(self):
        return self._reference_model_object

    @reference_model_object.setter
    def reference_model_object(self, val):
        if val is None:
            self._reference_model_object = val
            return
        p = "reference_model_object"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._reference_model_object = UUID(val) if isinstance(val, str) else val

    @property
    def reference_model(self):
        return self._reference_model

    @reference_model.setter
    def reference_model(self, val):
        if val is None:
            self._reference_model = val
            return
        p = "reference_model"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._reference_model = UUID(val) if isinstance(val, str) else val

    @property
    def gradient_type(self):
        return self._gradient_type

    @gradient_type.setter
    def gradient_type(self, val):
        if val is None:
            self._gradient_type = val
            return
        p = "gradient_type"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._gradient_type = val

    @property
    def lower_bound_object(self):
        return self._lower_bound_object

    @lower_bound_object.setter
    def lower_bound_object(self, val):
        if val is None:
            self._lower_bound_object = val
            return
        p = "lower_bound_object"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._lower_bound_object = UUID(val) if isinstance(val, str) else val

    @property
    def lower_bound(self):
        return self._lower_bound

    @lower_bound.setter
    def lower_bound(self, val):
        if val is None:
            self._lower_bound = val
            return
        p = "lower_bound"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._lower_bound = UUID(val) if isinstance(val, str) else val

    @property
    def upper_bound_object(self):
        return self._upper_bound_object

    @upper_bound_object.setter
    def upper_bound_object(self, val):
        if val is None:
            self._upper_bound_object = val
            return
        p = "upper_bound_object"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._upper_bound_object = UUID(val) if isinstance(val, str) else val

    @property
    def upper_bound(self):
        return self._upper_bound

    @upper_bound.setter
    def upper_bound(self, val):
        if val is None:
            self._upper_bound = val
            return
        p = "upper_bound"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._upper_bound = UUID(val) if isinstance(val, str) else val

    @property
    def parallelized(self):
        return self._parallelized

    @parallelized.setter
    def parallelized(self, val):
        if val is None:
            self._parallelized = val
            return
        p = "parallelized"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._parallelized = val

    @property
    def n_cpu(self):
        return self._n_cpu

    @n_cpu.setter
    def n_cpu(self, val):
        if val is None:
            self._n_cpu = val
            return
        p = "n_cpu"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._n_cpu = val

    @property
    def max_ram(self):
        return self._max_ram

    @max_ram.setter
    def max_ram(self, val):
        if val is None:
            self._max_ram = val
            return
        p = "max_ram"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._max_ram = val

    @property
    def out_group(self):
        return self._out_group

    @out_group.setter
    def out_group(self, val):
        if val is None:
            self._out_group = val
            return

        self.setter_validator(
            "out_group",
            val,
            fun=lambda x: x.name if isinstance(val, ContainerGroup) else x,
        )

    @property
    def ga_group(self) -> ContainerGroup | None:
        if (
            getattr(self, "_ga_group", None) is None
            and isinstance(self.geoh5, Workspace)
            and isinstance(self.out_group, str)
        ):
            self._ga_group = ContainerGroup.create(self.geoh5, name=self.out_group)

        return self._ga_group

    @property
    def no_data_value(self):
        return self._no_data_value

    @no_data_value.setter
    def no_data_value(self, val):
        if val is None:
            self._no_data_value = val
            return
        p = "no_data_value"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._no_data_value = val

    @property
    def distributed_workers(self):
        return self._distributed_workers

    @distributed_workers.setter
    def distributed_workers(self, val):
        if val is None:
            self._distributed_workers = val
            return
        p = "distributed_workers"
        if self.validate:
            self.validator.validate(
                p, val, self.validations[p], self.geoh5, self.associations
            )
        self._distributed_workers = val

    def write_input_file(
        self,
        ui_json: dict = None,
        default: bool = False,
        name: str = None,
        path: str = None,
    ):
        """Write out a ui.json with the current state of parameters"""

        if ui_json is None:
            defaults = deepcopy(self.defaults)
            ui_json = deepcopy(self.default_ui_json)
            ui_json["geoh5"] = self.geoh5
            self.title = defaults["title"]
            self.run_command = defaults["run_command"]

        if default:
            for k, v in defaults.items():
                if isinstance(ui_json[k], dict):
                    key = "value"
                    if "isValue" in ui_json[k].keys():
                        if ui_json[k]["isValue"] == False:
                            key = "property"
                    ui_json[k][key] = v
                else:
                    ui_json[k] = v

            ifile = InputFile.from_dict(ui_json)
        else:
            idict = self.to_dict(ui_json=ui_json)
            # TODO insert validate_chunk call here
            ifile = InputFile.from_dict(self.to_dict(ui_json=ui_json))

        if name is not None:
            if ".ui.json" not in name:
                name += ".ui.json"
        else:
            name = f"{self.out_group}.ui.json"

        if path is not None:
            if not os.path.exists(path):
                raise ValueError(f"Provided path {path} does not exist.")
            ifile.workpath = path

        none_map = {
            "starting_chi_factor": 1.0,
            "resolution": 0.0,
            "detrend_order": 0,
            "detrend_type": "all",
            "initial_beta": 1.0,
            "window_center_x": 0.0,
            "window_center_y": 0.0,
            "window_width": 0.0,
            "window_height": 0.0,
            "window_azimuth": 0.0,
            "n_cpu": 1,
        }

        ifile.write_ui_json(ui_json, name=name, default=default, none_map=none_map)
        if ifile.workpath is not None:
            ifile.filepath = os.path.join(ifile.workpath, name)
        else:
            ifile.filepath = os.path.abspath(name)
        self._input_file = ifile
