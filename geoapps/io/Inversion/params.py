#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import os
from uuid import UUID

from geoh5py.groups import ContainerGroup
from geoh5py.workspace import Workspace

from ..input_file import InputFile
from ..params import Params


class InversionParams(Params):

    _ga_group = None

    def __init__(self, **kwargs):

        self.forward_only = None
        self.topography_object: UUID = None
        self.topography = None
        self.data_object: UUID = None
        self.starting_model_object: UUID = None
        self.starting_model = None
        self.tile_spatial = None
        self.z_from_topo: bool = None
        self.receivers_radar_drape = None
        self.receivers_offset_x: float = None
        self.receivers_offset_y: float = None
        self.receivers_offset_z: float = None
        self.gps_receivers_offset = None
        self.ignore_values: str = None
        self.resolution: float = None
        self.detrend_data: bool = None
        self.detrend_order: int = None
        self.detrend_type: str = None
        self.max_chunk_size: int = None
        self.chunk_by_rows: bool = None
        self.output_tile_files: bool = None
        self.mesh = None
        self.mesh_from_params: bool = None
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

        for k, v in self.default_ui_json.items():
            if isinstance(v, dict):
                field = "value"
                if "isValue" in v.keys():
                    if not v["isValue"] or self.defaults[k] is None:
                        v["isValue"] = False
                        field = "property"
                self.default_ui_json[k][field] = self.defaults[k]
            else:
                self.default_ui_json[k] = self.defaults[k]

        super().__init__(**kwargs)

    def uncertainty(self, component: str) -> float:
        """Returns uncertainty for chosen data component."""
        return self.__getattribute__("_".join([component, "uncertainty"]))

    def channel(self, component: str) -> UUID:
        """Returns channel uuid for chosen data component."""
        return self.__getattribute__("_".join([component, "channel"]))

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
        for k, v in self.__dict__.items():
            if ("channel_bool" in k) & (v is True):
                comps.append(k.split("_")[1])
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
        radar = self.workspace.get_entity(self.receivers_radar_drape)
        radar = radar[0].values if radar else None
        return offsets, radar

    def model_norms(self) -> list[float]:
        """Returns model norm components as a list."""
        return [
            self.s_norm,
            self.x_norm,
            self.y_norm,
            self.z_norm,
        ]

    def directive_params(self, directive_name):

        if directive_name == "VectorInversion":
            return {"chifact_target": self.chi_factor * 2}

        elif directive_name == "Update_IRLS":
            kwargs = {}
            kwargs["f_min_change"] = self.f_min_change
            kwargs["max_irls_iterations"] = self.max_iterations
            kwargs[""]

    @property
    def forward_only(self):
        return self._forward_only

    @forward_only.setter
    def forward_only(self, val):
        if val is None:
            self._forward_only = val
            return
        p = "forward_only"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._resolution = val

    @property
    def detrend_data(self):
        return self._detrend_data

    @detrend_data.setter
    def detrend_data(self, val):
        if val is None:
            self._detrend_data = val
            return
        p = "detrend_data"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._detrend_data = val

    @property
    def detrend_order(self):
        return self._detrend_order

    @detrend_order.setter
    def detrend_order(self, val):
        if val is None:
            self._detrend_order = val
            return
        p = "detrend_order"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._mesh = UUID(val) if isinstance(val, str) else val

    @property
    def mesh_from_params(self):
        return self._mesh_from_params

    @mesh_from_params.setter
    def mesh_from_params(self, val):
        if val is None:
            self._mesh_from_params = val
            return
        p = "mesh_from_params"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._mesh_from_params = val

    @property
    def u_cell_size(self):
        return self._u_cell_size

    @u_cell_size.setter
    def u_cell_size(self, val):
        if val is None:
            self._u_cell_size = val
            return
        p = "u_cell_size"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._beta_search = val

    @property
    def max_iterations(self):
        return self._max_iterations

    @max_iterations.setter
    def max_iterations(self, val):
        if val is None:
            self._max_iterations = val
            return
        p = "max_iterations"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
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
            and isinstance(self.workspace, Workspace)
            and isinstance(self.out_group, str)
        ):
            self._ga_group = ContainerGroup.create(self.workspace, name=self.out_group)

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
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._no_data_value = val

    def write_input_file(
        self,
        ui_json: dict = None,
        default: bool = False,
        name: str = None,
        path: str = None,
    ):
        """Write out a ui.json with the current state of parameters"""

        if ui_json is None:
            if self.forward_only:
                defaults = self.forward_defaults
            else:
                defaults = self.inversion_defaults

            ui_json = {k: self.default_ui_json[k] for k in defaults}
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
            ifile = InputFile.from_dict(self.to_dict(ui_json=ui_json), self.validator)

        if name is not None:
            if ".ui.json" not in name:
                name += ".ui.json"
        else:
            name = f"{self.out_group}.ui.json"

        if path is not None:
            if not os.path.exists(path):
                raise ValueError(f"Provided path {path} does not exist.")
            ifile.workpath = path

        ifile.write_ui_json(ui_json, name=name, default=default)
        ifile.filepath = os.path.join(ifile.workpath, name)
        self._input_file = ifile
