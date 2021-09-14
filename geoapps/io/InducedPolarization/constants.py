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
    "inversion_type": "induced_polarization",
    "forward_only": False,
    "topography_object": None,
    "topography": None,
    "data_object": None,
    "chargeability_channel_bool": True,
    "chargeability_channel": None,
    "chargeability_uncertainty": 0.0,
    "conductivity_model_object": None,
    "conductivity_model": None,
    "starting_model_object": None,
    "starting_model": None,
    "tile_spatial": 1,
    "z_from_topo": True,
    "receivers_radar_drape": None,
    "receivers_offset_x": 0,
    "receivers_offset_y": 0,
    "receivers_offset_z": 0,
    "gps_receivers_offset": None,
    "ignore_values": None,
    "resolution": 0.0,
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
    "sens_wts_threshold": 1e-3,
    "every_iteration_bool": False,
    "f_min_change": 1e-4,
    "minGNiter": 1,
    "beta_tol": 0.5,
    "prctile": 50,
    "coolingRate": 1,
    "coolEps_q": True,
    "coolEpsFact": 1.2,
    "beta_search": False,
    "max_iterations": 25,
    "max_least_squares_iterations": 20,
    "max_cg_iterations": 30,
    "max_global_iterations": 100,
    "initial_beta_ratio": 1e1,
    "initial_beta": 0.0,
    "tol_cg": 1e-4,
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
    "lower_bound": None,
    "upper_bound_object": None,
    "upper_bound": None,
    "parallelized": True,
    "n_cpu": None,
    "max_ram": 2,
    "workspace": None,
    "out_group": "InducedPolarizationInversion",
    "no_data_value": None,
    "monitoring_directory": None,
    "geoh5": None,
    "run_command": "geoapps.drivers.induced_polarization_inversion",
    "run_command_boolean": False,
    "conda_environment": "geoapps",
}
forward_defaults = {
    "inversion_type": "induced_polarization",
    "forward_only": True,
    "topography_object": None,
    "topography": None,
    "data_object": None,
    "chargeability_channel_bool": True,
    "conductivity_model_object": None,
    "conductivity_model": None,
    "starting_model_object": None,
    "starting_model": None,
    "tile_spatial": 1,
    "z_from_topo": True,
    "receivers_radar_drape": None,
    "receivers_offset_x": 0,
    "receivers_offset_y": 0,
    "receivers_offset_z": 0,
    "gps_receivers_offset": None,
    "resolution": 0.0,
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
    "out_group": "InducedPolarizationForward",
    "monitoring_directory": None,
    "geoh5": None,
    "run_command": "geoapps.drivers.induced_polarization_inversion",
    "run_command_boolean": False,
    "conda_environment": "geoapps",
}
default_ui_json = {
    "inversion_type": "induced_polarization",
    "chargeability_channel_bool": True,
    "chargeability_channel": {
        "association": "Cell",
        "dataType": "Float",
        "default": None,
        "group": "Data",
        "main": True,
        "label": "Chargeability channel",
        "parent": "data_object",
        "value": None,
    },
    "chargeability_uncertainty": {
        "association": "Cell",
        "dataType": "Float",
        "default": 0.0,
        "group": "Data",
        "main": True,
        "isValue": True,
        "label": "chargeability uncertainty",
        "parent": "data_object",
        "property": None,
        "value": 0.0,
    },
    "conductivity_model_object": {
        "group": "Starting Model",
        "main": True,
        "meshType": [
            "{202C5DB1-A56D-4004-9CAD-BAAFD8899406}",
            "{6A057FDC-B355-11E3-95BE-FD84A7FFCB88}",
            "{F26FEBA3-ADED-494B-B9E9-B2BBCBE298E1}",
            "{48F5054A-1C5C-4CA4-9048-80F36DC60A06}",
            "{b020a277-90e2-4cd7-84d6-612ee3f25051}",
        ],
        "label": "Starting conductivity model object",
        "value": None,
    },
    "conductivity_model": {
        "association": "Cell",
        "dataType": "Float",
        "group": "Starting Model",
        "main": True,
        "isValue": False,
        "parent": "conductivity_model_object",
        "label": "Starting conductivity model value",
        "property": None,
        "value": 0.0,
    },
    "out_group": {"label": "Results group name", "value": "InducedPolarization"},
}


default_ui_json.update(base_default_ui_json)
for k, v in inversion_defaults.items():
    if isinstance(default_ui_json[k], dict):
        key = "value"
        if "isValue" in default_ui_json[k].keys():
            if default_ui_json[k]["isValue"] == False:
                key = "property"
        default_ui_json[k][key] = v
    else:
        default_ui_json[k] = v

default_ui_json = {k: default_ui_json[k] for k in inversion_defaults}
default_ui_json["data_object"]["meshType"] = "{275ecee9-9c24-4378-bf94-65f3c5fbe163}"

required_parameters = ["inversion_type"]
required_parameters += base_required_parameters
validations = {
    "inversion_type": {
        "types": [str],
        "values": ["induced_polarization"],
    },
    "chargeability_channel_bool": {"types": [bool]},
    "chargeability_channel": {
        "types": [str, UUID],
        "reqs": [("data_object")],
    },
    "chargeability_uncertainty": {"types": [str, int, float]},
    "conductivity_model_object": {
        "types": [str, UUID],
    },
    "conductivity_model": {
        "types": [str, UUID, int, float],
    },
}

validations.update(base_validations)
