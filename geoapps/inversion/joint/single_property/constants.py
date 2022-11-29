#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from geoapps.inversion import default_ui_json as base_default_ui_json

################# defaults ##################

inversion_defaults = {
    "title": "Joint Single Property Inversion",
    "inversion_type": "joint single property",
    "geoh5": None,  # Must remain at top of list for notebook app initialization
    "forward_only": False,
    "topography_object": None,
    "topography": None,
    "simulation_a": None,
    "simulation_b": None,
    "simulation_c": None,
    "mesh": None,
    "starting_model": 1e-3,
    "reference_model": 1e-3,
    "lower_bound": None,
    "upper_bound": None,
    "output_tile_files": False,
    "ignore_values": None,
    "window_center_x": None,
    "window_center_y": None,
    "window_width": None,
    "window_height": None,
    "window_azimuth": None,
    "inversion_style": "voxel",
    "chi_factor": 1.0,
    "initial_beta_ratio": 1e2,
    "initial_beta": None,
    "coolingRate": 2,
    "coolingFactor": 2.0,
    "max_global_iterations": 50,
    "max_line_search_iterations": 20,
    "max_cg_iterations": 30,
    "tol_cg": 1e-4,
    "alpha_s": 1.0,
    "alpha_x": 1.0,
    "alpha_y": 1.0,
    "alpha_z": 1.0,
    "s_norm": 0.0,
    "x_norm": 2.0,
    "y_norm": 2.0,
    "z_norm": 2.0,
    "max_irls_iterations": 25,
    "starting_chi_factor": None,
    "f_min_change": 1e-4,
    "beta_tol": 0.5,
    "prctile": 95,
    "coolEps_q": True,
    "coolEpsFact": 1.2,
    "beta_search": False,
    "gradient_type": "total",
    "sens_wts_threshold": 60.0,
    "every_iteration_bool": False,
    "parallelized": True,
    "n_cpu": None,
    "tile_spatial": 1,
    "max_ram": None,
    "store_sensitivities": "ram",
    "max_chunk_size": 128,
    "chunk_by_rows": True,
    "out_group": "JointInversion",
    "generate_sweep": False,
    "monitoring_directory": None,
    "workspace_geoh5": None,
    "run_command": "geoapps.inversion.driver",
    "conda_environment": "geoapps",
    "distributed_workers": None,
}

forward_defaults = {
    "title": "Joint Single Property Forward",
    "inversion_type": "joint single property",
    "geoh5": None,  # Must remain at top of list for notebook app initialization
    "forward_only": True,
    "topography_object": None,
    "topography": None,
    "simulation_a": None,
    "simulation_b": None,
    "simulation_c": None,
    "mesh": None,
    "starting_model": 1e-3,
    "output_tile_files": False,
    "parallelized": True,
    "n_cpu": None,
    "tile_spatial": 1,
    "max_chunk_size": 128,
    "chunk_by_rows": True,
    "out_group": "JointForward",
    "generate_sweep": False,
    "monitoring_directory": None,
    "workspace_geoh5": None,
    "run_command": "geoapps.inversion.driver",
    "conda_environment": "geoapps",
    "distributed_workers": None,
    "gradient_type": "total",
    "alpha_s": 1.0,
    "alpha_x": 1.0,
    "alpha_y": 1.0,
    "alpha_z": 1.0,
    "s_norm": 0.0,
    "x_norm": 2.0,
    "y_norm": 2.0,
    "z_norm": 2.0,
}

inversion_ui_json = {
    "detrend_type": None,
    "detrend_order": None,
}

forward_ui_json = {
    "gradient_type": "total",
    "alpha_s": 1.0,
    "alpha_x": 1.0,
    "alpha_y": 1.0,
    "alpha_z": 1.0,
    "s_norm": 0.0,
    "x_norm": 2.0,
    "y_norm": 2.0,
    "z_norm": 2.0,
}

default_ui_json = {
    "title": "Joint Single Property inversion",
    "inversion_type": "joint single property",
    "simulation_a": {
        "main": True,
        "group": "Data",
        "label": "Simulation A",
        "fileDescription": ["Input File"],
        "fileType": ["ui.json"],
        "value": None,
    },
    "simulation_b": {
        "main": True,
        "group": "Data",
        "label": "Simulation B",
        "fileDescription": ["Input File"],
        "fileType": ["ui.json"],
        "value": None,
    },
    "simulation_c": {
        "main": True,
        "group": "Data",
        "label": "Simulation C",
        "fileDescription": ["Input File"],
        "fileType": ["ui.json"],
        "optional": True,
        "enabled": False,
        "value": None,
    },
    "starting_model": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Mesh and Models",
        "main": True,
        "isValue": False,
        "parent": "mesh",
        "label": "Initial conductivity (S/m)",
        "property": None,
        "value": 1e-3,
    },
    "reference_model": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "main": True,
        "group": "Mesh and Models",
        "isValue": True,
        "parent": "mesh",
        "label": "Reference conductivity (S/m)",
        "property": None,
        "value": 1e-3,
    },
    "lower_bound": {
        "association": ["Cell", "Vertex"],
        "main": True,
        "dataType": "Float",
        "group": "Mesh and Models",
        "isValue": True,
        "parent": "mesh",
        "label": "Lower bound (S/m)",
        "property": None,
        "optional": True,
        "value": 1e-8,
        "enabled": False,
    },
    "upper_bound": {
        "association": ["Cell", "Vertex"],
        "main": True,
        "dataType": "Float",
        "group": "Mesh and Models",
        "isValue": True,
        "parent": "mesh",
        "label": "Upper bound (S/m)",
        "property": None,
        "optional": True,
        "value": 100.0,
        "enabled": False,
    },
    "out_group": {"label": "Results group name", "value": "JointInversion"},
}

default_ui_json = dict(base_default_ui_json, **default_ui_json)


################ Validations #################


validations = {
    "inversion_type": {
        "required": True,
        "values": ["joint single property"],
    },
    "simulation_a": {"required": True, "types": [str]},
    "simulation_b": {"required": True, "types": [str]},
    "simulation_c": {"types": [str, type(None)]},
}

app_initializer = {}
