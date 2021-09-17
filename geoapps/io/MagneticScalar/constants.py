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
    "title": "SimPEG Inversion - Magnetic Susceptibility",
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
    "window_center_x": None,
    "window_center_y": None,
    "window_width": None,
    "window_height": None,
    "window_azimuth": None,
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
    "max_line_search_iterations": 20,
    "max_cg_iterations": 30,
    "max_global_iterations": 100,
    "initial_beta_ratio": 1e2,
    "initial_beta": None,
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
    "reference_model": 0.0,
    "gradient_type": "total",
    "lower_bound_object": None,
    "lower_bound": None,
    "upper_bound_object": None,
    "upper_bound": None,
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
    "title": "SimPEG Forward - Magnetic Susceptibility",
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
    "window_center_x": None,
    "window_center_y": None,
    "window_width": None,
    "window_height": None,
    "window_azimuth": None,
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
    "title": "SimPEG Inversion - Magnetic Susceptibility",
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
        "types": [str, int, float, UUID],
        "reqs": [(True, "tmi_channel_bool")],
    },
    "out_group": {"types": [str, ContainerGroup]},
}
validations.update(base_validations)
