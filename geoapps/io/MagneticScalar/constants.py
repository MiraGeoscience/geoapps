#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from uuid import UUID

import numpy as np
from geoh5py.groups import ContainerGroup
from geoh5py.workspace import Workspace

from geoapps.io.Inversion.constants import default_ui_json as base_default_ui_json
from geoapps.io.Inversion.constants import (
    required_parameters as base_required_parameters,
)
from geoapps.io.Inversion.constants import validations as base_validations

################# defaults ##################

inversion_defaults = {
    "inversion_type": "magnetic scalar",
    "forward_only": False,
    "inducing_field_strength": 50000.0,
    "inducing_field_inclination": 90.0,
    "inducing_field_declination": 0.0,
    "topography_object": None,
    "topography": None,
    "data_object": None,
    "tmi_channel_bool": True,
    "tmi_channel": None,
    "tmi_uncertainty": 10.0,
    "starting_model_object": None,
    "starting_model": 0.01,
    "tile_spatial": 1,
    "z_from_topo": True,
    "receivers_radar_drape": None,
    "receivers_offset_x": 0,
    "receivers_offset_y": 0,
    "receivers_offset_z": 0,
    "gps_receivers_offset": None,
    "ignore_values": None,
    "resolution": 50.0,
    "detrend_data": False,
    "detrend_order": 0,
    "detrend_type": "all",
    "max_chunk_size": 128,
    "chunk_by_rows": False,
    "output_tile_files": False,
    "mesh": None,
    "mesh_from_params": False,
    "u_cell_size": 25.0,
    "v_cell_size": 25.0,
    "w_cell_size": 25.0,
    "octree_levels_topo": [0, 0, 0, 2],
    "octree_levels_obs": [5, 5, 5, 5],
    "depth_core": 500.0,
    "max_distance": np.inf,
    "horizontal_padding": 1000.0,
    "vertical_padding": 1000.0,
    "window_center_x": 0.0,
    "window_center_y": 0.0,
    "window_width": 0.0,
    "window_height": 0.0,
    "window_azimuth": 0.0,
    "inversion_style": "voxel",
    "chi_factor": 1.0,
    "max_iterations": 25,
    "max_cg_iterations": 30,
    "max_global_iterations": 100,
    "initial_beta_ratio": 1e1,
    "initial_beta": 0.0,
    "tol_cg": 1e-16,
    "alpha_s": 1.0,
    "alpha_x": 1.0,
    "alpha_y": 1.0,
    "alpha_z": 1.0,
    "s_norm": 2.0,
    "x_norm": 2.0,
    "y_norm": 2.0,
    "z_norm": 2.0,
    "reference_model_object": None,
    "reference_model": None,
    "gradient_type": "total",
    "lower_bound_object": None,
    "lower_bound": 0.0,
    "upper_bound_object": None,
    "upper_bound": 1.0,
    "parallelized": True,
    "n_cpu": None,
    "max_ram": 2,
    "workspace": None,
    "out_group": "SusceptibilityInversion",
    "no_data_value": None,
    "monitoring_directory": None,
    "geoh5": None,
    "run_command": "geoapps.drivers.magnetic_scalar_inversion",
    "run_command_boolean": False,
    "conda_environment": "geoapps",
}
forward_defaults = {
    "inversion_type": "magnetic scalar",
    "forward_only": True,
    "inducing_field_strength": 50000.0,
    "inducing_field_inclination": 90.0,
    "inducing_field_declination": 0.0,
    "topography_object": None,
    "topography": None,
    "data_object": None,
    "tmi_channel_bool": True,
    "starting_model_object": None,
    "starting_model": 0.0,
    "tile_spatial": 1,
    "z_from_topo": True,
    "receivers_radar_drape": None,
    "receivers_offset_x": 0,
    "receivers_offset_y": 0,
    "receivers_offset_z": 0,
    "gps_receivers_offset": None,
    "resolution": 50.0,
    "max_chunk_size": 128,
    "chunk_by_rows": False,
    "mesh": None,
    "mesh_from_params": False,
    "u_cell_size": 25.0,
    "v_cell_size": 25.0,
    "w_cell_size": 25.0,
    "octree_levels_topo": [0, 0, 0, 2],
    "octree_levels_obs": [5, 5, 5, 5],
    "depth_core": 500.0,
    "max_distance": np.inf,
    "horizontal_padding": 1000.0,
    "vertical_padding": 1000.0,
    "window_center_x": 0.0,
    "window_center_y": 0.0,
    "window_width": 0.0,
    "window_height": 0.0,
    "window_azimuth": 0.0,
    "parallelized": True,
    "n_cpu": None,
    "max_ram": 2,
    "workspace": None,
    "out_group": "MagneticScalarForward",
    "monitoring_directory": None,
    "geoh5": None,
    "run_command": "geoapps.drivers.magnetic_scalar_inversion",
    "run_command_boolean": False,
    "conda_environment": "geoapps",
}

default_ui_json = {
    "inversion_type": "magnetic scalar",
    "inducing_field_strength": {
        "association": "Cell",
        "dataType": "Float",
        "min": 0.0,
        "main": True,
        "group": "Inducing Field",
        "isValue": True,
        "label": "Strength",
        "parent": "data_object",
        "property": None,
        "value": 50000.0,
    },
    "inducing_field_inclination": {
        "association": "Cell",
        "dataType": "Float",
        "min": 0.0,
        "main": True,
        "group": "Inducing Field",
        "isValue": True,
        "label": "Inclination",
        "parent": "data_object",
        "property": None,
        "value": 90.0,
    },
    "inducing_field_declination": {
        "association": "Cell",
        "dataType": "Float",
        "min": 0.0,
        "main": True,
        "group": "Inducing Field",
        "isValue": True,
        "parent": "data_object",
        "label": "Declination",
        "property": None,
        "value": 0.0,
    },
    "tmi_channel_bool": True,
    "tmi_channel": {
        "association": "Cell",
        "dataType": "Float",
        "group": "Data",
        "main": True,
        "label": "TMI channel",
        "parent": "data_object",
        "value": None,
    },
    "tmi_uncertainty": {
        "association": "Cell",
        "dataType": "Float",
        "group": "Data",
        "main": True,
        "isValue": True,
        "label": "TMI uncertainty",
        "parent": "data_object",
        "property": None,
        "value": 0.0,
    },
    "out_group": {"label": "Results group name", "value": "SusceptibilityInversion"},
}
default_ui_json.update(base_default_ui_json)
default_ui_json = {k: default_ui_json[k] for k in inversion_defaults}


################ Validations #################

required_parameters = ["inversion_type"]
required_parameters += base_required_parameters

validations = {
    "inversion_type": {
        "types": [str],
        "values": ["magnetic scalar"],
    },
    "inducing_field_strength": {
        "types": [int, float],
    },
    "inducing_field_inclination": {
        "types": [int, float],
    },
    "inducing_field_declination": {
        "types": [int, float],
    },
    "tmi_channel_bool": {"types": [bool]},
    "tmi_channel": {
        "types": [str, UUID],
        "reqs": [("data_object"), (True, "tmi_channel_bool")],
    },
    "tmi_uncertainty": {
        "types": [str, int, float],
        "reqs": [(True, "tmi_channel_bool")],
    },
    "out_group": {"types": [str, ContainerGroup]},
}
validations.update(base_validations)
