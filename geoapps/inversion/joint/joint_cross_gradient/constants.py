#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from geoapps.inversion import default_ui_json as base_default_ui_json
from geoapps.inversion.constants import validations as base_validations

################# defaults ##################

inversion_defaults = {
    "title": "SimPEG Joint Cross Gradient Inversion",
    "inversion_type": "joint cross gradient",
    "geoh5": None,  # Must remain at top of list for notebook app initialization
    "forward_only": False,
    "topography_object": None,
    "topography": None,
    "group_a": None,
    "group_b": None,
    "mesh": None,
    "output_tile_files": False,
    "inversion_style": "voxel",
    "chi_factor": 1.0,
    "initial_beta_ratio": 10.0,
    "initial_beta": None,
    "coolingRate": 1,
    "coolingFactor": 2.0,
    "max_global_iterations": 100,
    "max_line_search_iterations": 20,
    "max_cg_iterations": 30,
    "tol_cg": 1e-4,
    "alpha_s": 1.0,
    "length_scale_x": 1.0,
    "length_scale_y": 1.0,
    "length_scale_z": 1.0,
    "s_norm": 0.0,
    "x_norm": 2.0,
    "y_norm": 2.0,
    "z_norm": 2.0,
    "gradient_type": "total",
    "max_irls_iterations": 25,
    "starting_chi_factor": None,
    "f_min_change": 1e-4,
    "beta_tol": 0.5,
    "prctile": 95,
    "coolEps_q": True,
    "coolEpsFact": 1.2,
    "beta_search": False,
    "sens_wts_threshold": 0.001,
    "every_iteration_bool": True,
    "parallelized": True,
    "n_cpu": None,
    "store_sensitivities": "ram",
    "max_ram": None,
    "max_chunk_size": 128,
    "chunk_by_rows": True,
    "out_group": None,
    "generate_sweep": False,
    "monitoring_directory": None,
    "workspace_geoh5": None,
    "run_command": "geoapps.inversion.driver",
    "conda_environment": "geoapps",
    "distributed_workers": None,
}
forward_defaults = {
    "title": "Joint Cross Gradient Forward",
    "inversion_type": "joint cross gradient",
    "geoh5": None,  # Must remain at top of list for notebook app initialization
    "forward_only": True,
    "topography_object": None,
    "topography": None,
    "group_a": None,
    "group_b": None,
    "gps_receivers_offset": None,
    "mesh": None,
    "output_tile_files": False,
    "parallelized": True,
    "n_cpu": None,
    "max_chunk_size": 128,
    "chunk_by_rows": True,
    "out_group": None,
    "generate_sweep": False,
    "monitoring_directory": None,
    "workspace_geoh5": None,
    "run_command": "geoapps.inversion.driver",
    "conda_environment": "geoapps",
    "distributed_workers": None,
}

default_ui_json = {
    "title": "SimPEG Joint Cross Gradient Inversion",
    "inversion_type": "joint surveys",
    "mesh": {
        "group": "Mesh and Models",
        "main": True,
        "label": "Mesh",
        "meshType": "4EA87376-3ECE-438B-BF12-3479733DED46",
        "value": None,
        "enabled": False,
        "optional": True,
    },
    "group_a": {
        "main": True,
        "group": "Data",
        "label": "Group A",
        "groupType": "{55ed3daf-c192-4d4b-a439-60fa987fe2b8}",
        "value": "",
    },
    "group_b": {
        "main": True,
        "group": "Data",
        "label": "Group B",
        "groupType": "{55ed3daf-c192-4d4b-a439-60fa987fe2b8}",
        "value": "",
    },
    "group_c": {
        "main": True,
        "group": "Data",
        "label": "Group C",
        "groupType": "{55ed3daf-c192-4d4b-a439-60fa987fe2b8}",
        "optional": True,
        "enabled": False,
        "value": "",
    },
    "tile_spatial": None,
}

default_ui_json = dict(base_default_ui_json, **default_ui_json)

################ Validations #################


validations = {
    "inversion_type": {
        "required": True,
        "values": ["joint cross gradient"],
    },
}

validations = dict(base_validations, **validations)
app_initializer = {}
