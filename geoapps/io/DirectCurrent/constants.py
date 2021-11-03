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
    "title": "SimPEG Direct Current Inversion",
    "inversion_type": "direct current",
    "geoh5": None,  # Must remain at top of list for notebook app initialization
    "forward_only": False,
    "topography_object": None,
    "topography": None,
    "data_object": None,
    "potential_channel_bool": True,
    "potential_channel": None,
    "potential_uncertainty": 1.0,
    "starting_model_object": None,
    "starting_model": None,
    "tile_spatial": 1,
    "output_tile_files": False,
    "z_from_topo": False,
    "receivers_radar_drape": None,
    "receivers_offset_x": 0.0,
    "receivers_offset_y": 0.0,
    "receivers_offset_z": 0.0,
    "gps_receivers_offset": None,
    "ignore_values": None,
    "resolution": 0.0,
    "detrend_data": False,
    "detrend_order": None,
    "detrend_type": None,
    "max_chunk_size": 128,
    "chunk_by_rows": False,
    "mesh": None,
    "mesh_from_params": False,
    "u_cell_size": 25.0,
    "v_cell_size": 25.0,
    "w_cell_size": 25.0,
    "octree_levels_topo": [16, 8, 4, 2],
    "octree_levels_obs": [4, 4, 4, 4],
    "depth_core": 500.0,
    "max_distance": 5000.0,
    "horizontal_padding": 1000.0,
    "vertical_padding": 1000.0,
    "window_center_x": None,
    "window_center_y": None,
    "window_width": None,
    "window_height": None,
    "window_azimuth": None,
    "inversion_style": "voxel",
    "chi_factor": 1.0,
    "sens_wts_threshold": 0.0,
    "every_iteration_bool": False,
    "f_min_change": 1e-4,
    "minGNiter": 1,
    "beta_tol": 0.5,
    "prctile": 95,
    "coolingRate": 1,
    "coolEps_q": True,
    "coolEpsFact": 1.2,
    "beta_search": False,
    "max_iterations": 25,
    "max_line_search_iterations": 20,
    "max_cg_iterations": 30,
    "max_global_iterations": 100,
    "initial_beta_ratio": 10.0,
    "initial_beta": None,
    "tol_cg": 1e-4,
    "alpha_s": 1.0,
    "alpha_x": 1.0,
    "alpha_y": 1.0,
    "alpha_z": 1.0,
    "s_norm": 0.0,
    "x_norm": 2.0,
    "y_norm": 2.0,
    "z_norm": 2.0,
    "reference_model_object": None,
    "reference_model": None,
    "gradient_type": "total",
    "lower_bound_object": None,
    "lower_bound": None,
    "upper_bound_object": None,
    "upper_bound": None,
    "parallelized": True,
    "n_cpu": None,
    "max_ram": 2,
    "workspace": None,
    "out_group": "DirectCurrentInversion",
    "no_data_value": None,
    "monitoring_directory": None,
    "run_command": "geoapps.drivers.direct_current_inversion",
    "run_command_boolean": False,
    "conda_environment": "geoapps",
    "distributed_workers": None,
}
forward_defaults = {
    "title": "SimPEG Direct Current Forward",
    "inversion_type": "direct current",
    "geoh5": None,  # Must remain at top of list for notebook app initialization
    "forward_only": True,
    "topography_object": None,
    "topography": None,
    "data_object": None,
    "potential_channel_bool": True,
    "starting_model_object": None,
    "starting_model": None,
    "tile_spatial": 1,
    "output_tile_files": False,
    "z_from_topo": False,
    "receivers_radar_drape": None,
    "receivers_offset_x": 0.0,
    "receivers_offset_y": 0.0,
    "receivers_offset_z": 0.0,
    "gps_receivers_offset": None,
    "resolution": 0.0,
    "max_chunk_size": 128,
    "chunk_by_rows": False,
    "mesh": None,
    "mesh_from_params": False,
    "u_cell_size": 25.0,
    "v_cell_size": 25.0,
    "w_cell_size": 25.0,
    "octree_levels_topo": [16, 8, 4, 2],
    "octree_levels_obs": [4, 4, 4, 4],
    "depth_core": 500.0,
    "max_distance": 5000.0,
    "horizontal_padding": 1000.0,
    "vertical_padding": 1000.0,
    "window_center_x": None,
    "window_center_y": None,
    "window_width": None,
    "window_height": None,
    "window_azimuth": None,
    "parallelized": True,
    "n_cpu": None,
    "workspace": None,
    "out_group": "DirectCurrentForward",
    "monitoring_directory": None,
    "run_command": "geoapps.drivers.direct_current_inversion",
    "run_command_boolean": False,
    "conda_environment": "geoapps",
    "distributed_workers": None,
}
default_ui_json = {
    "title": "SimPEG Inversion - Direct Current",
    "inversion_type": "direct current",
    "potential_channel_bool": True,
    "potential_channel": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Data",
        "main": True,
        "label": "Potential channel",
        "parent": "data_object",
        "value": None,
    },
    "potential_uncertainty": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Data",
        "main": True,
        "isValue": True,
        "label": "Potential uncertainty",
        "parent": "data_object",
        "property": None,
        "value": 1.0,
    },
    "starting_model": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Starting Model",
        "main": True,
        "isValue": False,
        "parent": "starting_model_object",
        "label": "Conductivity (Siemens/m)",
        "property": None,
        "value": 0.0,
    },
    "resolution": {
        "enabled": False,
        "visible": False,
        "label": "resolution",
        "value": 0.0,
    },
    "detrend_order": {
        "enabled": False,
        "visible": False,
        "label": "detrend order",
        "value": 0.0,
    },
    "detrend_type": {
        "enabled": False,
        "visible": False,
        "label": "detrend type",
        "value": "all",
    },
    "out_group": {"label": "Results group name", "value": "DirectCurrent"},
}

base_default_ui_json.update(default_ui_json)
default_ui_json = base_default_ui_json
for k, v in inversion_defaults.items():
    if isinstance(default_ui_json[k], dict):
        key = "value"
        if "isValue" in default_ui_json[k].keys():
            if default_ui_json[k]["isValue"] == False:
                key = "property"
        default_ui_json[k][key] = v
        if "enabled" in default_ui_json[k].keys() and v is not None:
            default_ui_json[k]["enabled"] = True
    else:
        default_ui_json[k] = v

default_ui_json = {k: default_ui_json[k] for k in inversion_defaults}
default_ui_json["data_object"]["meshType"] = "{275ecee9-9c24-4378-bf94-65f3c5fbe163}"

required_parameters = ["inversion_type"]
required_parameters += base_required_parameters
validations = {
    "inversion_type": {
        "types": [str],
        "values": ["direct current"],
    },
    "potential_channel_bool": {"types": [bool]},
    "potential_channel": {
        "types": [str, UUID],
        "reqs": [("data_object")],
    },
    "potential_uncertainty": {"types": [str, int, float]},
}

validations.update(base_validations)

app_initializer = {
    "geoh5": "../../assets/FlinFlon_dcip.geoh5",
    "data_object": "{6e14de2c-9c2f-4976-84c2-b330d869cb82}",
    "potential_channel_bool": True,
    "potential_channel": "{502e7256-aafa-4016-969f-5cc3a4f27315}",
    "potential_uncertainty": "{62746129-3d82-427e-a84c-78cded00c0bc}",
    "mesh_from_params": True,
    "reference_model": 1e-1,
    "starting_model": 1e-1,
    "u_cell_size": 25.0,
    "v_cell_size": 25.0,
    "w_cell_size": 25.0,
    "resolution": 25,
    "window_center_x": None,
    "window_center_y": None,
    "window_width": None,
    "window_height": None,
    "window_azimuth": None,
    "octree_levels_topo": [0, 0, 0, 2],
    "octree_levels_obs": [5, 5, 5, 5],
    "depth_core": 500.0,
    "max_distance": np.inf,
    "horizontal_padding": 1000.0,
    "vertical_padding": 1000.0,
    "s_norm": 0.0,
    "x_norm": 2.0,
    "y_norm": 2.0,
    "z_norm": 2.0,
    "upper_bound": 100.0,
    "lower_bound": 1e-5,
    "max_iterations": 25,
    "topography_object": "{ab3c2083-6ea8-4d31-9230-7aad3ec09525}",
    "topography": "{a603a762-f6cb-4b21-afda-3160e725bf7d}",
    "z_from_topo": True,
    "receivers_offset_x": 0,
    "receivers_offset_y": 0,
    "receivers_offset_z": 0,
    "out_group": "DCInversion",
}
