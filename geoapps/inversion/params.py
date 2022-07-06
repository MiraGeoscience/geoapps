#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from copy import deepcopy
from uuid import UUID

import numpy as np
from geoh5py.data import NumericData
from geoh5py.groups import SimPEGGroup
from geoh5py.ui_json import InputFile
from geoh5py.workspace import Workspace

from geoapps.driver_base.params import BaseParams


class InversionBaseParams(BaseParams):
    """
    Base parameter class for geophysical->property inversion.
    """

    _directive_list = None
    _default_ui_json = None
    _forward_defaults = None
    _forward_ui_json = None
    _inversion_defaults = None
    _inversion_ui_json = None
    _inversion_type = None
    _ga_group = None

    def __init__(self, input_file=None, forward_only=False, **kwargs):
        self._forward_only: bool = forward_only
        self._topography_object: UUID = None
        self._topography: UUID | float = None
        self._data_object: UUID = None
        self._starting_model_object: UUID = None
        self._starting_model: UUID | float = None
        self._tile_spatial = None
        self._z_from_topo: bool = None
        self._receivers_radar_drape = None
        self._receivers_offset_x: float = None
        self._receivers_offset_y: float = None
        self._receivers_offset_z: float = None
        self._gps_receivers_offset = None
        self._ignore_values: str = None
        self._resolution: float = None
        self._detrend_order: int = None
        self._detrend_type: str = None
        self._max_chunk_size: int = None
        self._chunk_by_rows: bool = None
        self._output_tile_files: bool = None
        self._mesh = None
        self._u_cell_size: float = None
        self._v_cell_size: float = None
        self._w_cell_size: float = None
        self._octree_levels_topo: list[int] = None
        self._octree_levels_obs: list[int] = None
        self._depth_core: float = None
        self._max_distance: float = None
        self._horizontal_padding: float = None
        self._vertical_padding: float = None
        self._window_azimuth: float = None
        self._window_center_x: float = None
        self._window_center_y: float = None
        self._window_height: float = None
        self._window_width: float = None
        self._inversion_style: str = None
        self._chi_factor: float = None
        self._sens_wts_threshold: float = None
        self._every_iteration_bool: bool = None
        self._f_min_change: float = None
        self._minGNiter: float = None
        self._beta_tol: float = None
        self._prctile: float = None
        self._coolingRate: float = None
        self._coolEps_q: bool = None
        self._coolEpsFact: float = None
        self._beta_search: bool = None
        self._starting_chi_factor: float = None
        self._max_iterations: int = None
        self._max_line_search_iterations: int = None
        self._max_cg_iterations: int = None
        self._max_global_iterations: int = None
        self._initial_beta: float = None
        self._initial_beta_ratio: float = None
        self._tol_cg: float = None
        self._alpha_s: float = None
        self._alpha_x: float = None
        self._alpha_y: float = None
        self._alpha_z: float = None
        self._s_norm: float = None
        self._x_norm: float = None
        self._y_norm: float = None
        self._z_norm: float = None
        self._reference_model_object: UUID = None
        self._reference_model = None
        self._gradient_type: str = None
        self._lower_bound_object: UUID = None
        self._lower_bound = None
        self._upper_bound_object: UUID = None
        self._upper_bound = None
        self._parallelized: bool = None
        self._n_cpu: int = None
        self._max_ram: float = None
        self._out_group = None
        self._no_data_value: float = None
        self._distributed_workers = None
        self._defaults = (
            self.forward_defaults if self.forward_only else self.inversion_defaults
        )

        if input_file is None:
            ui_json = deepcopy(self._default_ui_json)
            ui_json.update(
                self._forward_ui_json if self.forward_only else self._inversion_ui_json
            )
            ui_json = {k: ui_json[k] for k in self.defaults}  # Re-order using defaults
            input_file = InputFile(
                ui_json=ui_json,
                data=self.defaults,
                validations=self.validations,
                validation_options={"disabled": True},
            )

        super().__init__(input_file=input_file, **kwargs)

    def data_channel(self, component: str):
        """Return uuid of data channel."""
        return getattr(self, "_".join([component, "channel"]), None)

    def uncertainty_channel(self, component: str):
        """Return uuid of uncertainty channel."""
        return getattr(self, "_".join([component, "uncertainty"]), None)

    def data(self, component: str):
        """Returns array of data for chosen data component."""
        data_entity = self.data_channel(component)
        if isinstance(data_entity, NumericData):
            return data_entity.values.astype(float)
        return None

    def uncertainty(self, component: str) -> np.ndarray | None:
        """Returns uncertainty for chosen data component."""
        val = self.uncertainty_channel(component)

        if isinstance(val, NumericData):
            return val.values.astype(float)
        elif self.data(component) is not None:
            d = self.data(component)
            if isinstance(val, (int, float)):
                return np.array([float(val)] * len(d))
            else:
                return d * 0.0 + 1.0  # Default
        else:
            return None

    def cell_size(self):
        """Returns core cell size in all 3 dimensions."""
        return [self.u_cell_size, self.v_cell_size, self.w_cell_size]

    def components(self) -> list[str]:
        """Retrieve component names used to index channel and uncertainty data."""
        comps = []
        channels = [
            k.lstrip("_").split("_channel_bool")[0]
            for k in self.__dict__.keys()
            if "channel_bool" in k
        ]

        for c in channels:
            if (
                getattr(self, f"{c}_channel", None) is not None
                or getattr(self, f"{c}_channel_bool", None) is True
            ):
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
    def forward_defaults(self):
        if getattr(self, "_forward_defaults", None) is None:
            raise NotImplementedError(
                "The property '_forward_defaults' must be assigned on "
                "the child inversion class."
            )
        return self._forward_defaults

    @property
    def forward_only(self):
        return self._forward_only

    @forward_only.setter
    def forward_only(self, val):
        self.setter_validator("forward_only", val)

    @property
    def inversion_defaults(self):
        if getattr(self, "_inversion_defaults", None) is None:
            raise NotImplementedError(
                "The property '_inversion_defaults' must be assigned on "
                "the child inversion class."
            )
        return self._inversion_defaults

    @property
    def topography_object(self):
        return self._topography_object

    @topography_object.setter
    def topography_object(self, val):
        self.setter_validator("topography_object", val, fun=self._uuid_promoter)

    @property
    def topography(self):
        return self._topography

    @topography.setter
    def topography(self, val):
        self.setter_validator("topography", val, fun=self._uuid_promoter)

    @property
    def data_object(self):
        return self._data_object

    @data_object.setter
    def data_object(self, val):
        self.setter_validator("data_object", val, fun=self._uuid_promoter)

    @property
    def starting_model_object(self):
        return self._starting_model_object

    @starting_model_object.setter
    def starting_model_object(self, val):
        self.setter_validator("starting_model_object", val, fun=self._uuid_promoter)

    @property
    def starting_model(self):
        return self._starting_model

    @starting_model.setter
    def starting_model(self, val):
        self.setter_validator("starting_model", val, fun=self._uuid_promoter)

    @property
    def tile_spatial(self):
        return self._tile_spatial

    @tile_spatial.setter
    def tile_spatial(self, val):
        self.setter_validator("tile_spatial", val, fun=self._uuid_promoter)

    @property
    def z_from_topo(self):
        return self._z_from_topo

    @z_from_topo.setter
    def z_from_topo(self, val):
        self.setter_validator("z_from_topo", val)

    @property
    def receivers_radar_drape(self):
        return self._receivers_radar_drape

    @receivers_radar_drape.setter
    def receivers_radar_drape(self, val):
        self.setter_validator("receivers_radar_drape", val, fun=self._uuid_promoter)

    @property
    def receivers_offset_x(self):
        return self._receivers_offset_x

    @receivers_offset_x.setter
    def receivers_offset_x(self, val):
        self.setter_validator("receivers_offset_x", val)

    @property
    def receivers_offset_y(self):
        return self._receivers_offset_y

    @receivers_offset_y.setter
    def receivers_offset_y(self, val):
        self.setter_validator("receivers_offset_y", val)

    @property
    def receivers_offset_z(self):
        return self._receivers_offset_z

    @receivers_offset_z.setter
    def receivers_offset_z(self, val):
        self.setter_validator("receivers_offset_z", val)

    @property
    def gps_receivers_offset(self):
        return self._gps_receivers_offset

    @gps_receivers_offset.setter
    def gps_receivers_offset(self, val):
        self.setter_validator("gps_receivers_offset", val, fun=self._uuid_promoter)

    @property
    def ignore_values(self):
        return self._ignore_values

    @ignore_values.setter
    def ignore_values(self, val):
        self.setter_validator("ignore_values", val)

    @property
    def inversion_type(self):
        return self._inversion_type

    @inversion_type.setter
    def inversion_type(self, val):
        self.setter_validator("inversion_type", val)

    @property
    def resolution(self):
        return self._resolution

    @resolution.setter
    def resolution(self, val):
        self.setter_validator("resolution", val)

    @property
    def detrend_order(self):
        return self._detrend_order

    @detrend_order.setter
    def detrend_order(self, val):
        self.setter_validator("detrend_order", val)

    @property
    def detrend_type(self):
        return self._detrend_type

    @detrend_type.setter
    def detrend_type(self, val):
        self.setter_validator("detrend_type", val)

    @property
    def max_chunk_size(self):
        return self._max_chunk_size

    @max_chunk_size.setter
    def max_chunk_size(self, val):
        self.setter_validator("max_chunk_size", val)

    @property
    def chunk_by_rows(self):
        return self._chunk_by_rows

    @chunk_by_rows.setter
    def chunk_by_rows(self, val):
        self.setter_validator("chunk_by_rows", val)

    @property
    def output_tile_files(self):
        return self._output_tile_files

    @output_tile_files.setter
    def output_tile_files(self, val):
        self.setter_validator("output_tile_files", val)

    @property
    def mesh(self):
        return self._mesh

    @mesh.setter
    def mesh(self, val):
        self.setter_validator("mesh", val, fun=self._uuid_promoter)

    @property
    def u_cell_size(self):
        return self._u_cell_size

    @u_cell_size.setter
    def u_cell_size(self, val):
        self.setter_validator("u_cell_size", val)

    @property
    def v_cell_size(self):
        return self._v_cell_size

    @v_cell_size.setter
    def v_cell_size(self, val):
        self.setter_validator("v_cell_size", val)

    @property
    def w_cell_size(self):
        return self._w_cell_size

    @w_cell_size.setter
    def w_cell_size(self, val):
        self.setter_validator("w_cell_size", val)

    @property
    def octree_levels_topo(self):
        return self._octree_levels_topo

    @octree_levels_topo.setter
    def octree_levels_topo(self, val):
        self.setter_validator("octree_levels_topo", val)

    @property
    def octree_levels_obs(self):
        return self._octree_levels_obs

    @octree_levels_obs.setter
    def octree_levels_obs(self, val):
        self.setter_validator("octree_levels_obs", val)

    @property
    def depth_core(self):
        return self._depth_core

    @depth_core.setter
    def depth_core(self, val):
        self.setter_validator("depth_core", val)

    @property
    def max_distance(self):
        return self._max_distance

    @max_distance.setter
    def max_distance(self, val):
        self.setter_validator("max_distance", val)

    @property
    def horizontal_padding(self):
        return self._horizontal_padding

    @horizontal_padding.setter
    def horizontal_padding(self, val):
        self.setter_validator("horizontal_padding", val)

    @property
    def vertical_padding(self):
        return self._vertical_padding

    @vertical_padding.setter
    def vertical_padding(self, val):
        self.setter_validator("vertical_padding", val)

    @property
    def window_center_x(self):
        return self._window_center_x

    @window_center_x.setter
    def window_center_x(self, val):
        self.setter_validator("window_center_x", val)

    @property
    def window_center_y(self):
        return self._window_center_y

    @window_center_y.setter
    def window_center_y(self, val):
        self.setter_validator("window_center_y", val)

    @property
    def window_width(self):
        return self._window_width

    @window_width.setter
    def window_width(self, val):
        self.setter_validator("window_width", val)

    @property
    def window_height(self):
        return self._window_height

    @window_height.setter
    def window_height(self, val):
        self.setter_validator("window_height", val)

    @property
    def window_azimuth(self):
        return self._window_azimuth

    @window_azimuth.setter
    def window_azimuth(self, val):
        self.setter_validator("window_azimuth", val)

    @property
    def inversion_style(self):
        return self._inversion_style

    @inversion_style.setter
    def inversion_style(self, val):
        self.setter_validator("inversion_style", val)

    @property
    def chi_factor(self):
        return self._chi_factor

    @chi_factor.setter
    def chi_factor(self, val):
        self.setter_validator("chi_factor", val)

    @property
    def sens_wts_threshold(self):
        return self._sens_wts_threshold

    @sens_wts_threshold.setter
    def sens_wts_threshold(self, val):
        self.setter_validator("sens_wts_threshold", val)

    @property
    def every_iteration_bool(self):
        return self._every_iteration_bool

    @every_iteration_bool.setter
    def every_iteration_bool(self, val):
        self.setter_validator("every_iteration_bool", val)

    @property
    def f_min_change(self):
        return self._f_min_change

    @f_min_change.setter
    def f_min_change(self, val):
        self.setter_validator("f_min_change", val)

    @property
    def minGNiter(self):
        return self._minGNiter

    @minGNiter.setter
    def minGNiter(self, val):
        self.setter_validator("minGNiter", val)

    @property
    def beta_tol(self):
        return self._beta_tol

    @beta_tol.setter
    def beta_tol(self, val):
        self.setter_validator("beta_tol", val)

    @property
    def prctile(self):
        return self._prctile

    @prctile.setter
    def prctile(self, val):
        self.setter_validator("prctile", val)

    @property
    def coolingRate(self):
        return self._coolingRate

    @coolingRate.setter
    def coolingRate(self, val):
        self.setter_validator("coolingRate", val)

    @property
    def coolEps_q(self):
        return self._coolEps_q

    @coolEps_q.setter
    def coolEps_q(self, val):
        self.setter_validator("coolEps_q", val)

    @property
    def coolEpsFact(self):
        return self._coolEpsFact

    @coolEpsFact.setter
    def coolEpsFact(self, val):
        self.setter_validator("coolEpsFact", val)

    @property
    def beta_search(self):
        return self._beta_search

    @beta_search.setter
    def beta_search(self, val):
        self.setter_validator("beta_search", val)

    @property
    def starting_chi_factor(self):
        return self._starting_chi_factor

    @starting_chi_factor.setter
    def starting_chi_factor(self, val):
        self.setter_validator("starting_chi_factor", val)

    @property
    def max_iterations(self):
        return self._max_iterations

    @max_iterations.setter
    def max_iterations(self, val):
        self.setter_validator("max_iterations", val)

    @property
    def max_line_search_iterations(self):
        return self._max_line_search_iterations

    @max_line_search_iterations.setter
    def max_line_search_iterations(self, val):
        self.setter_validator("max_line_search_iterations", val)

    @property
    def max_cg_iterations(self):
        return self._max_cg_iterations

    @max_cg_iterations.setter
    def max_cg_iterations(self, val):
        self.setter_validator("max_cg_iterations", val)

    @property
    def max_global_iterations(self):
        return self._max_global_iterations

    @max_global_iterations.setter
    def max_global_iterations(self, val):
        self.setter_validator("max_global_iterations", val)

    @property
    def initial_beta(self):
        return self._initial_beta

    @initial_beta.setter
    def initial_beta(self, val):
        self.setter_validator("initial_beta", val)

    @property
    def initial_beta_ratio(self):
        return self._initial_beta_ratio

    @initial_beta_ratio.setter
    def initial_beta_ratio(self, val):
        self.setter_validator("initial_beta_ratio", val)

    @property
    def tol_cg(self):
        return self._tol_cg

    @tol_cg.setter
    def tol_cg(self, val):
        self.setter_validator("tol_cg", val)

    @property
    def alpha_s(self):
        return self._alpha_s

    @alpha_s.setter
    def alpha_s(self, val):
        self.setter_validator("alpha_s", val)

    @property
    def alpha_x(self):
        return self._alpha_x

    @alpha_x.setter
    def alpha_x(self, val):
        self.setter_validator("alpha_x", val)

    @property
    def alpha_y(self):
        return self._alpha_y

    @alpha_y.setter
    def alpha_y(self, val):
        self.setter_validator("alpha_y", val)

    @property
    def alpha_z(self):
        return self._alpha_z

    @alpha_z.setter
    def alpha_z(self, val):
        self.setter_validator("alpha_z", val)

    @property
    def s_norm(self):
        return self._s_norm

    @s_norm.setter
    def s_norm(self, val):
        self.setter_validator("s_norm", val)

    @property
    def x_norm(self):
        return self._x_norm

    @x_norm.setter
    def x_norm(self, val):
        self.setter_validator("x_norm", val)

    @property
    def y_norm(self):
        return self._y_norm

    @y_norm.setter
    def y_norm(self, val):
        self.setter_validator("y_norm", val)

    @property
    def z_norm(self):
        return self._z_norm

    @z_norm.setter
    def z_norm(self, val):
        self.setter_validator("z_norm", val)

    @property
    def reference_model_object(self):
        return self._reference_model_object

    @reference_model_object.setter
    def reference_model_object(self, val):
        self.setter_validator("reference_model_object", val, fun=self._uuid_promoter)

    @property
    def reference_model(self):
        return self._reference_model

    @reference_model.setter
    def reference_model(self, val):
        self.setter_validator("reference_model", val, fun=self._uuid_promoter)

    @property
    def gradient_type(self):
        return self._gradient_type

    @gradient_type.setter
    def gradient_type(self, val):
        self.setter_validator("gradient_type", val)

    @property
    def lower_bound_object(self):
        return self._lower_bound_object

    @lower_bound_object.setter
    def lower_bound_object(self, val):
        self.setter_validator("lower_bound_object", val, fun=self._uuid_promoter)

    @property
    def lower_bound(self):
        return self._lower_bound

    @lower_bound.setter
    def lower_bound(self, val):
        self.setter_validator("lower_bound", val, fun=self._uuid_promoter)

    @property
    def upper_bound_object(self):
        return self._upper_bound_object

    @upper_bound_object.setter
    def upper_bound_object(self, val):
        self.setter_validator("upper_bound_object", val, fun=self._uuid_promoter)

    @property
    def upper_bound(self):
        return self._upper_bound

    @upper_bound.setter
    def upper_bound(self, val):
        self.setter_validator("upper_bound", val, fun=self._uuid_promoter)

    @property
    def parallelized(self):
        return self._parallelized

    @parallelized.setter
    def parallelized(self, val):
        self.setter_validator("parallelized", val)

    @property
    def n_cpu(self):
        return self._n_cpu

    @n_cpu.setter
    def n_cpu(self, val):
        self.setter_validator("n_cpu", val)

    @property
    def max_ram(self):
        return self._max_ram

    @max_ram.setter
    def max_ram(self, val):
        self.setter_validator("max_ram", val)

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
        )

    @property
    def ga_group(self) -> SimPEGGroup | None:
        if (
            getattr(self, "_ga_group", None) is None
            and isinstance(self.geoh5, Workspace)
            and isinstance(self.out_group, str)
        ):
            self._ga_group = SimPEGGroup.create(self.geoh5, name=self.out_group)
        elif isinstance(self.out_group, SimPEGGroup):
            self._ga_group = self.out_group
        return self._ga_group

    @property
    def distributed_workers(self):
        return self._distributed_workers

    @distributed_workers.setter
    def distributed_workers(self, val):
        self.setter_validator("distributed_workers", val)
