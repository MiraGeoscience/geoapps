#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from uuid import UUID

import numpy as np

from geoapps.io.Inversion.constants import default_ui_json as base_default_ui_json
from geoapps.io.Inversion.constants import (
    required_parameters as base_required_parameters,
)
from geoapps.io.Inversion.constants import validations as base_validations

inversion_defaults = {
    "inversion_type": "direct_current",
    "forward_only": False,
    "topography_object": None,
    "topography": None,
    "data_object": None,
    "potential_channel": None,
    "potential_uncertainty": 0.0,
    "starting_model_object": None,
    "starting_model": 0.0,
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
    "smallness_norm": 2.0,
    "x_norm": 2.0,
    "y_norm": 2.0,
    "z_norm": 2.0,
    "reference_model_object": None,
    "reference_model": None,
    "gradient_type": "total",
    "lower_bound_object": None,
    "lower_bound": -1,
    "upper_bound_object": None,
    "upper_bound": 1,
    "parallelized": True,
    "n_cpu": None,
    "max_ram": 2,
    "workspace": None,
    "out_group": "DirectCurrentInversion",
    "no_data_value": None,
    "monitoring_directory": None,
    "geoh5": None,
    "run_command": "geoapps.drivers.direct_current_inversion",
    "run_command_boolean": False,
    "conda_environment": "geoapps",
}
forward_defaults = {
    "inversion_type": "direct_current",
    "forward_only": True,
    "topography_object": None,
    "topography": None,
    "data_object": None,
    "starting_model_object": None,
    "starting_model": None,
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
    "parallelized": True,
    "n_cpu": None,
    "max_ram": 2,
    "workspace": None,
    "out_group": "DirectCurrentForward",
    "monitoring_directory": None,
    "geoh5": None,
    "run_command": "geoapps.drivers.direct_current_inversion",
    "run_command_boolean": False,
    "conda_environment": "geoapps",
}
default_ui_json = {
    "inversion_type": "direct_current",
    "potential_channel": {
        "association": "Cell",
        "dataType": "Float",
        "default": None,
        "group": "Data",
        "main": True,
        "label": "Potential channel",
        "parent": "data_object",
        "value": None,
    },
    "potential_uncertainty": {
        "association": "Cell",
        "dataType": "Float",
        "default": 0.0,
        "group": "Data",
        "main": True,
        "isValue": True,
        "label": "Potential uncertainty",
        "parent": "data_object",
        "property": None,
        "value": 0.0,
    },
    "out_group": {"label": "Results group name", "value": "DirectCurrent"},
}
default_ui_json.update(base_default_ui_json)
default_ui_json = {k: default_ui_json[k] for k in inversion_defaults}
default_ui_json["data_object"]["meshType"] = "{275ecee9-9c24-4378-bf94-65f3c5fbe163}"

required_parameters = ["inversion_type"]
required_parameters += base_required_parameters
validations = {
    "inversion_type": {
        "types": [str],
        "values": ["direct_current"],
    },
    "potential_channel": {
        "types": [str, UUID],
        "reqs": [("data_object")],
    },
    "potential_uncertainty": {"types": [str, int, float]},
}

validations.update(base_validations)
