#  Copyright (c) 2024 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import geoapps
from geoapps.inversion.constants import validations as base_validations
from geoapps.inversion.joint.constants import default_ui_json as joint_default_ui_json

################# defaults ##################

inversion_defaults = {
    "title": "SimPEG Joint Surveys Inversion",
    "inversion_type": "joint surveys",
    "version": geoapps.__version__,
    "geoh5": None,  # Must remain at top of list for notebook app initialization
    "forward_only": False,
    "topography_object": None,
    "topography": None,
    "group_a": None,
    "group_a_multiplier": 1.0,
    "group_b": None,
    "group_b_multiplier": 1.0,
    "group_c": None,
    "group_c_multiplier": 1.0,
    "mesh": None,
    "starting_model": None,
    "reference_model": None,
    "lower_bound": None,
    "upper_bound": None,
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

default_ui_json = {
    "title": "SimPEG Joint Surveys Inversion",
    "inversion_type": "joint surveys",
    "starting_model": {
        "association": "Cell",
        "dataType": "Float",
        "group": "Mesh and Models",
        "main": True,
        "isValue": False,
        "parent": "mesh",
        "label": "Initial model",
        "property": None,
        "optional": True,
        "enabled": False,
        "value": 1e-4,
    },
    "lower_bound": {
        "association": "Cell",
        "main": True,
        "dataType": "Float",
        "group": "Mesh and Models",
        "isValue": False,
        "parent": "mesh",
        "label": "Lower bound)",
        "property": None,
        "optional": True,
        "value": -10.0,
        "enabled": False,
    },
    "upper_bound": {
        "association": "Cell",
        "main": True,
        "dataType": "Float",
        "group": "Mesh and Models",
        "isValue": False,
        "parent": "mesh",
        "label": "Upper bound",
        "property": None,
        "optional": True,
        "value": 10.0,
        "enabled": False,
    },
    "reference_model": {
        "association": "Cell",
        "main": True,
        "dataType": "Float",
        "group": "Mesh and models",
        "isValue": False,
        "parent": "mesh",
        "label": "Reference",
        "property": None,
        "optional": True,
        "value": 1e-4,
        "enabled": False,
    },
}
default_ui_json = dict(joint_default_ui_json, **default_ui_json)
validations = {
    "inversion_type": {
        "required": True,
        "values": ["joint surveys"],
    },
}

validations = dict(base_validations, **validations)
app_initializer = {}
