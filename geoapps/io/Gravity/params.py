#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from typing import Any, Dict, List, Tuple, Union
from uuid import UUID

from geoh5py.groups import ContainerGroup

from ..input_file import InputFile
from ..params import Params
from ..validators import InputValidator
from .constants import default_ui_json, required_parameters, validations


class GravityParams(Params):

    _default_ui_json = default_ui_json

    def __init__(self, **kwargs):

        self.validations: Dict[str, Any] = validations
        self.validator: InputValidator = InputValidator(
            required_parameters, validations
        )
        self.associations: Dict[Union[str, UUID], Union[str, UUID]] = None
        self.forward_only: bool = None
        self.topography_object: UUID = None
        self.topography = None
        self.data_object = None
        self.gz_channel = None
        self.gz_uncertainty = None
        self.starting_model_object = None
        self.starting_model = None
        self.tile_spatial = None
        self.z_from_topo = None
        self.receivers_radar_drape = None
        self.receivers_offset_x = None
        self.receivers_offset_y = None
        self.receivers_offset_z = None
        self.gps_receivers_offset = None
        self.ignore_values = None
        self.resolution = None
        self.detrend_data = None
        self.detrend_order = None
        self.detrend_type = None
        self.max_chunk_size = None
        self.chunk_by_rows = None
        self.output_tile_files = None
        self.mesh = None
        self.mesh_from_params = None
        self.core_cell_size_x = None
        self.core_cell_size_y = None
        self.core_cell_size_z = None
        self.octree_levels_topo = None
        self.octree_levels_obs = None
        self.octree_levels_padding = None
        self.depth_core = None
        self.max_distance = None
        self.padding_distance_x = None
        self.padding_distance_y = None
        self.padding_distance_z = None
        self.window_center_x = None
        self.window_center_y = None
        self.window_width = None
        self.window_height = None
        self.inversion_style = None
        self.chi_factor = None
        self.max_iterations = None
        self.max_cg_iterations = None
        self.max_global_iterations = None
        self.initial_beta = None
        self.initial_beta_ratio = None
        self.tol_cg = None
        self.alpha_s = None
        self.alpha_x = None
        self.alpha_y = None
        self.alpha_z = None
        self.smallness_norm = None
        self.x_norm = None
        self.y_norm = None
        self.z_norm = None
        self.reference_model_object = None
        self.reference_model = None
        self.gradient_type = None
        self.lower_bound = None
        self.upper_bound = None
        self.parallelized = None
        self.n_cpu = None
        self.max_ram = None
        self.inversion_type = None
        self.out_group = None
        self.no_data_value = None
        self._input_file = InputFile()

        super().__init__(**kwargs)

    def _set_defaults(self) -> None:
        """ Wraps Params._set_defaults """
        return super()._set_defaults(self.default_ui_json)

    def default(self, param) -> Any:
        """ Wraps Params.default. """
        return super().default(self.default_ui_json, param)

    def uncertainty(self, component: str) -> float:
        """ Returns uncertainty for chosen data component. """
        return self.__getattribute__("_".join([component, "uncertainty"]))

    def channel(self, component: str) -> UUID:
        """ Returns channel uuid for chosen data component. """
        return self.__getattribute__("_".join([component, "channel"]))

    def components(self) -> List[str]:
        """ Retrieve component names used to index channel and uncertainty data. """
        comps = []
        for k, v in self.__dict__.item():
            if ("channel_bool" in k) & (v == True):
                comps.append(k.split("_")[1])
        return comps

    def window(self) -> Dict[str, float]:
        """ Returns window dictionary """
        win = {
            "center_x": self.window_center_x,
            "center_y": self.window_center_y,
            "width": self.window_width,
            "height": self.window_height,
            "center": [self.window_center_x, self.window_center_y],
            "size": [self.window_width, self.window_height],
        }
        return win if any([v is not None for v in win.values()]) else None

    def offset(self) -> Tuple[List[float], UUID]:
        """ Returns offset components as list and drape data. """
        offsets = [
            self.receivers_offset_x,
            self.receivers_offset_y,
            self.receivers_offset_z,
        ]
        is_offset = any([(k != 0) for k in offsets])
        offsets = offsets if is_offset else None
        return offsets, self.receivers_radar_drape

    def model_norms(self) -> List[float]:
        """ Returns model norm components as a list. """
        return [
            self.smallness_norm,
            self.x_norm,
            self.y_norm,
            self.z_norm,
        ]

    @property
    def inversion_type(self):
        return self._inversion_type

    @inversion_type.setter
    def inversion_type(self, val):
        if val is None:
            self._inversion_type = val
            return
        p = "inversion_type"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._inversion_type = val

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
    def gz_channel_bool(self):
        return self._gz_channel_bool

    @gz_channel_bool.setter
    def gz_channel_bool(self, val):
        if val is None:
            self._gz_channel_bool = val
            return
        p = "gz_channel_bool"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._gz_channel_bool = val

    @property
    def gz_channel(self):
        return self._gz_channel

    @gz_channel.setter
    def gz_channel(self, val):
        if val is None:
            self._gz_channel = val
            return
        p = "gz_channel"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._gz_channel = UUID(val) if isinstance(val, str) else val

    @property
    def gz_uncertainty(self):
        return self._gz_uncertainty

    @gz_uncertainty.setter
    def gz_uncertainty(self, val):
        if val is None:
            self._gz_uncertainty = val
            return
        p = "gz_uncertainty"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._gz_uncertainty = UUID(val) if isinstance(val, str) else val

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
    def core_cell_size_x(self):
        return self._core_cell_size_x

    @core_cell_size_x.setter
    def core_cell_size_x(self, val):
        if val is None:
            self._core_cell_size_x = val
            return
        p = "core_cell_size_x"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._core_cell_size_x = val

    @property
    def core_cell_size_y(self):
        return self._core_cell_size_y

    @core_cell_size_y.setter
    def core_cell_size_y(self, val):
        if val is None:
            self._core_cell_size_y = val
            return
        p = "core_cell_size_y"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._core_cell_size_y = val

    @property
    def core_cell_size_z(self):
        return self._core_cell_size_z

    @core_cell_size_z.setter
    def core_cell_size_z(self, val):
        if val is None:
            self._core_cell_size_z = val
            return
        p = "core_cell_size_z"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._core_cell_size_z = val

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
    def octree_levels_padding(self):
        return self._octree_levels_padding

    @octree_levels_padding.setter
    def octree_levels_padding(self, val):
        if val is None:
            self._octree_levels_padding = val
            return
        p = "octree_levels_padding"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._octree_levels_padding = val

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
    def padding_distance_x(self):
        return self._padding_distance_x

    @padding_distance_x.setter
    def padding_distance_x(self, val):
        if val is None:
            self._padding_distance_x = val
            return
        p = "padding_distance_x"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._padding_distance_x = val

    @property
    def padding_distance_y(self):
        return self._padding_distance_y

    @padding_distance_y.setter
    def padding_distance_y(self, val):
        if val is None:
            self._padding_distance_y = val
            return
        p = "padding_distance_y"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._padding_distance_y = val

    @property
    def padding_distance_z(self):
        return self._padding_distance_z

    @padding_distance_z.setter
    def padding_distance_z(self, val):
        if val is None:
            self._padding_distance_z = val
            return
        p = "padding_distance_z"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._padding_distance_z = val

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
    def smallness_norm(self):
        return self._smallness_norm

    @smallness_norm.setter
    def smallness_norm(self, val):
        if val is None:
            self._smallness_norm = val
            return
        p = "smallness_norm"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._smallness_norm = val

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
        p = "out_group"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._out_group = ContainerGroup.create(self.workspace, name=val)

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

    def _init_params(self, inputfile: InputFile) -> None:
        """ Wraps Params._init_params. """
        super()._init_params(inputfile, required_parameters, validations)
