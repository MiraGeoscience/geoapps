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

required_parameters = []

default_ui_json = {
    "forward_only": False,
    "topography_object": {
        "main": True,
        "group": "Topography",
        "label": "Object",
        "meshType": [
            "{202C5DB1-A56D-4004-9CAD-BAAFD8899406}",
            "{6A057FDC-B355-11E3-95BE-FD84A7FFCB88}",
            "{F26FEBA3-ADED-494B-B9E9-B2BBCBE298E1}",
            "{48F5054A-1C5C-4CA4-9048-80F36DC60A06}",
            "{b020a277-90e2-4cd7-84d6-612ee3f25051}",
        ],
        "value": None,
    },
    "topography": {
        "association": "Vertex",
        "dataType": "Float",
        "group": "Topography",
        "main": True,
        "dependency": "forward_only",
        "dependencyType": "hide",
        "isValue": False,
        "label": "Elevation",
        "parent": "topography_object",
        "property": None,
        "value": 0.0,
    },
    "data_object": {
        "main": True,
        "group": "Data",
        "label": "Object",
        "meshType": [
            "{202C5DB1-A56D-4004-9CAD-BAAFD8899406}",
            "{6A057FDC-B355-11E3-95BE-FD84A7FFCB88}",
            "{F26FEBA3-ADED-494B-B9E9-B2BBCBE298E1}",
            "{48F5054A-1C5C-4CA4-9048-80F36DC60A06}",
            "{b020a277-90e2-4cd7-84d6-612ee3f25051}",
        ],
        "value": None,
    },
    "starting_model_object": {
        "group": "Starting Model",
        "main": True,
        "meshType": [
            "{202C5DB1-A56D-4004-9CAD-BAAFD8899406}",
            "{6A057FDC-B355-11E3-95BE-FD84A7FFCB88}",
            "{F26FEBA3-ADED-494B-B9E9-B2BBCBE298E1}",
            "{48F5054A-1C5C-4CA4-9048-80F36DC60A06}",
            "{b020a277-90e2-4cd7-84d6-612ee3f25051}",
        ],
        "label": "Starting model object",
        "value": None,
    },
    "starting_model": {
        "association": "Cell",
        "dataType": "Float",
        "group": "Starting Model",
        "main": True,
        "isValue": False,
        "parent": "starting_model_object",
        "label": "Starting model value",
        "property": None,
        "value": 0.0,
    },
    "tile_spatial": {
        "association": "Cell",
        "dataType": "Float",
        "group": "Receivers Options",
        "isValue": True,
        "label": "Number of tiles",
        "parent": "data_object",
        "property": None,
        "value": 1,
    },
    "z_from_topo": {
        "main": False,
        "group": "Receivers Options",
        "label": "Take z from topography",
        "value": True,
    },
    "receivers_radar_drape": {
        "association": "Cell",
        "dataType": "Float",
        "main": False,
        "group": "Receivers Options",
        "label": "Drape receivers with radar channel",
        "parent": "data_object",
        "value": None,
    },
    "receivers_offset_x": {
        "group": "Receivers Options",
        "main": False,
        "label": "Constant receiver offset in x",
        "value": 0,
    },
    "receivers_offset_y": {
        "group": "Receivers Options",
        "main": False,
        "label": "Constant receiver offset in y",
        "value": 0,
    },
    "receivers_offset_z": {
        "group": "Receivers Options",
        "main": False,
        "label": "Constant receiver offset in z",
        "value": 0,
    },
    "gps_receivers_offset": {
        "association": "Cell",
        "dataType": "Float",
        "group": "Receivers Options",
        "enabled": False,
        "visible": False,
        "isValue": False,
        "label": "Set data offsets",
        "parent": "data_object",
        "property": None,
        "value": 0.0,
    },
    "ignore_values": {
        "group": "Data Options",
        "optional": True,
        "enabled": False,
        "label": "Values to ignore",
        "value": None,
    },
    "resolution": {
        "min": 0.0,
        "group": "Data Options",
        "optional": True,
        "enabled": True,
        "label": "Resolution",
        "value": 0.0,
    },
    "detrend_data": {
        "group": "Data Options",
        "label": "Remove trend from data",
        "value": False,
    },
    "detrend_order": {
        "choiceList": [0, 1, 2],
        "group": "Data Options",
        "enabled": False,
        "dependency": "detrend_data",
        "dependencyType": "enabled",
        "label": "Detrend order",
        "value": 0,
    },
    "detrend_type": {
        "choiceList": ["all", "corners"],
        "group": "Data Options",
        "dependency": "detrend_data",
        "dependencyType": "enabled",
        "enabled": False,
        "label": "Detrend type",
        "value": "all",
    },
    "max_chunk_size": {
        "min": 0,
        "group": "Data Options",
        "optional": True,
        "enabled": True,
        "label": "Maximum chunk size",
        "value": 128,
    },
    "chunk_by_rows": {
        "group": "Data Options",
        "label": "Chunk by rows",
        "value": False,
    },
    "output_tile_files": {
        "group": "Data Options",
        "label": "Output tile files",
        "value": False,
    },
    "mesh_from_params": {
        "group": "Mesh",
        "main": True,
        "label": "Build from parameters",
        "value": False,
    },
    "mesh": {
        "group": "Mesh",
        "main": True,
        "label": "Mesh",
        "meshType": "4EA87376-3ECE-438B-BF12-3479733DED46",
        "value": None,
    },
    "u_cell_size": {
        "min": 0.0,
        "group": "Mesh",
        "main": True,
        "enabled": False,
        "dependency": "mesh_from_params",
        "dependencyType": "enable",
        "label": "Core cell size in x",
        "value": 25.0,
    },
    "v_cell_size": {
        "min": 0.0,
        "group": "Mesh",
        "main": True,
        "enabled": False,
        "dependency": "mesh_from_params",
        "dependencyType": "enabled",
        "label": "Core cell size in y",
        "value": 25.0,
    },
    "w_cell_size": {
        "min": 0.0,
        "group": "Mesh",
        "main": True,
        "enabled": False,
        "dependency": "mesh_from_params",
        "dependencyType": "enable",
        "label": "Core cell size in z",
        "value": 25.0,
    },
    "octree_levels_topo": {
        "group": "Mesh",
        "main": True,
        "enabled": False,
        "dependency": "mesh_from_params",
        "dependencyType": "enable",
        "label": "Octree levels topography",
        "value": [0, 0, 0, 2],
    },
    "octree_levels_obs": {
        "group": "Mesh",
        "main": True,
        "enabled": False,
        "dependency": "mesh_from_params",
        "dependencyType": "enable",
        "label": "Octree levels observations",
        "value": [5, 5, 5, 5],
    },
    "depth_core": {
        "min": 0,
        "group": "Mesh",
        "main": True,
        "enabled": False,
        "dependency": "mesh_from_params",
        "dependencyType": "enable",
        "label": "Depth of core refinement volume",
        "value": 500.0,
    },
    "max_distance": {
        "min": 0,
        "group": "Mesh",
        "main": True,
        "enabled": False,
        "dependency": "mesh_from_params",
        "dependencyType": "enable",
        "label": "Maximum padding distance",
        "value": np.inf,
    },
    "horizontal_padding": {
        "min": 0.0,
        "group": "Mesh",
        "main": True,
        "enabled": False,
        "dependency": "mesh_from_params",
        "dependency_type": "enable",
        "label": "Horizontal padding",
        "value": 1000.0,
    },
    "vertical_padding": {
        "min": 0.0,
        "group": "Mesh",
        "main": True,
        "enabled": False,
        "dependency": "mesh_from_params",
        "dependency_type": "enable",
        "label": "Vertical padding",
        "value": 1000.0,
    },
    "window_center_x": {
        "group": "window",
        "optional": True,
        "enabled": False,
        "label": "Window center easting",
        "value": 0.0,
    },
    "window_center_y": {
        "group": "window",
        "optional": True,
        "enabled": False,
        "label": "Window center northing",
        "value": 0.0,
    },
    "window_width": {
        "min": 0.0,
        "group": "window",
        "optional": True,
        "enabled": False,
        "label": "Window width",
        "value": 0.0,
    },
    "window_height": {
        "min": 0.0,
        "group": "window",
        "optional": True,
        "enabled": False,
        "label": "Window height",
        "value": 0.0,
    },
    "window_azimuth": {
        "min": -180,
        "max": 180,
        "group": "window",
        "optional": True,
        "enabled": False,
        "label": "Window azimuth",
        "value": 0.0,
    },
    "inversion_style": {
        "choiceList": ["voxel"],
        "group": "Optimization",
        "label": "Inversion style",
        "value": "voxel",
    },
    "chi_factor": {
        "min": 0.0,
        "max": 1.0,
        "group": "Optimization",
        "label": "Chi factor",
        "value": 1.0,
    },
    "sens_wts_threshold": {
        "group": "Update sensitivity weights directive",
        "groupOptional": True,
        "visible": True,
        "enabled": True,
        "label": "Update sensitivity weight threshold",
        "value": 1e-3,
    },
    "every_iteration_bool": {
        "group": "Update sensitivity weights directive",
        "visible": True,
        "enabled": True,
        "label": "Update every iteration",
        "value": False,
    },
    "f_min_change": {
        "group": "Update IRLS directive",
        "groupOptional": True,
        "visible": True,
        "enabled": True,
        "label": "f min change",
        "value": 1e-4,
    },
    "minGNiter": {
        "group": "Update IRLS directive",
        "visible": True,
        "enabled": True,
        "label": "Minimum Gauss-Newton iterations",
        "value": 1,
    },
    "beta_tol": {
        "group": "Update IRLS directive",
        "visible": True,
        "enabled": False,
        "label": "Beta tolerance",
        "value": 0.5,
    },
    "prctile": {
        "group": "Update IRLS directive",
        "visible": True,
        "enabled": False,
        "label": "percentile",
        "value": 50,
    },
    "coolingRate": {
        "group": "Update IRLS directive",
        "visible": True,
        "enabled": False,
        "label": "Beta cooling rate",
        "value": 1,
    },
    "coolEps_q": {
        "group": "Update IRLS directive",
        "visible": True,
        "enabled": False,
        "label": "Cool epsilon q",
        "value": True,
    },
    "coolEpsFact": {
        "group": "Update IRLS directive",
        "visible": True,
        "enabled": False,
        "label": "Cool epsilon fact",
        "value": 1.2,
    },
    "beta_search": {
        "group": "Update IRLS directive",
        "visible": True,
        "enabled": False,
        "label": "Perform beta search",
        "value": False,
    },
    "max_iterations": {
        "min": 0,
        "group": "Optimization",
        "label": "Max iteration count",
        "value": 25,
    },
    "max_line_search_iterations": {
        "group": "Optimization",
        "label": "Max least squares iteration count",
        "value": 20,
    },
    "max_cg_iterations": {
        "min": 0,
        "group": "Optimization",
        "label": "Max conjugate gradient iteration count",
        "value": 30,
    },
    "max_global_iterations": {
        "min": 0,
        "group": "Optimization",
        "label": "Max global iteration count",
        "value": 100,
    },
    "initial_beta_ratio": {
        "min": 0.0,
        "group": "Optimization",
        "optional": True,
        "enabled": True,
        "dependency": "initial_beta",
        "dependencyType": "disabled",
        "label": "Initial beta ratio",
        "value": 1e2,
    },
    "initial_beta": {
        "min": 0.0,
        "group": "Optimization",
        "optional": True,
        "enabled": False,
        "dependency": "provide_beta",
        "dependencyType": "enabled",
        "label": "Initial beta",
        "value": "",
    },
    "tol_cg": {
        "min": 0,
        "group": "Optimization",
        "label": "Conjugate gradient tolerance",
        "value": 1e-4,
    },
    "alpha_s": {
        "min": 0.0,
        "group": "Regularization",
        "label": "Smallness weight",
        "value": 1.0,
    },
    "alpha_x": {
        "min": 0.0,
        "group": "Regularization",
        "label": "X-smoothness weight",
        "value": 1.0,
    },
    "alpha_y": {
        "min": 0.0,
        "group": "Regularization",
        "label": "Y-smoothness weight",
        "value": 1.0,
    },
    "alpha_z": {
        "min": 0.0,
        "group": "Regularization",
        "label": "Z-smoothness weight",
        "value": 1.0,
    },
    "s_norm": {
        "min": 0.0,
        "max": 2.0,
        "group": "Regularization",
        "label": "Smallness norm",
        "value": 2.0,
    },
    "x_norm": {
        "min": 0.0,
        "max": 2.0,
        "group": "Regularization",
        "label": "X-smoothness norm",
        "value": 2.0,
    },
    "y_norm": {
        "min": 0.0,
        "max": 2.0,
        "group": "Regularization",
        "label": "Y-smoothness norm",
        "value": 2.0,
    },
    "z_norm": {
        "min": 0.0,
        "max": 2.0,
        "group": "Regularization",
        "label": "Z-smoothness norm",
        "value": 2.0,
    },
    "reference_model_object": {
        "group": "Regularization",
        "label": "Reference model object",
        "meshType": [
            "{202C5DB1-A56D-4004-9CAD-BAAFD8899406}",
            "{6A057FDC-B355-11E3-95BE-FD84A7FFCB88}",
            "{F26FEBA3-ADED-494B-B9E9-B2BBCBE298E1}",
            "{48F5054A-1C5C-4CA4-9048-80F36DC60A06}",
            "{b020a277-90e2-4cd7-84d6-612ee3f25051}",
        ],
        "value": None,
    },
    "reference_model": {
        "association": "Cell",
        "dataType": "Float",
        "group": "Regularization",
        "isValue": True,
        "optional": True,
        "enabled": False,
        "label": "Reference model value",
        "parent": "reference_model_object",
        "property": None,
        "value": 0.0,
    },
    "gradient_type": {
        "choiceList": ["total", "components"],
        "group": "Regularization",
        "label": "Gradient type",
        "value": "total",
    },
    "lower_bound_object": {
        "group": "Regularization",
        "meshType": [
            "{202C5DB1-A56D-4004-9CAD-BAAFD8899406}",
            "{6A057FDC-B355-11E3-95BE-FD84A7FFCB88}",
            "{F26FEBA3-ADED-494B-B9E9-B2BBCBE298E1}",
            "{48F5054A-1C5C-4CA4-9048-80F36DC60A06}",
            "{b020a277-90e2-4cd7-84d6-612ee3f25051}",
        ],
        "label": "Lower bound object",
        "value": None,
    },
    "lower_bound": {
        "association": "Cell",
        "dataType": "Float",
        "group": "Regularization",
        "isValue": False,
        "parent": "lower_bound_object",
        "label": "Lower bound on model",
        "property": None,
        "value": 0.0,
    },
    "upper_bound_object": {
        "group": "Regularization",
        "visible": True,
        "label": "Upper bound object",
        "meshType": [
            "{202C5DB1-A56D-4004-9CAD-BAAFD8899406}",
            "{6A057FDC-B355-11E3-95BE-FD84A7FFCB88}",
            "{F26FEBA3-ADED-494B-B9E9-B2BBCBE298E1}",
            "{48F5054A-1C5C-4CA4-9048-80F36DC60A06}",
            "{b020a277-90e2-4cd7-84d6-612ee3f25051}",
        ],
        "value": None,
    },
    "upper_bound": {
        "association": "Cell",
        "dataType": "Float",
        "group": "Regularization",
        "isValue": False,
        "parent": "upper_bound_object",
        "label": "Upper bound on model",
        "property": None,
        "value": 0.0,
    },
    "parallelized": {
        "group": "Compute",
        "label": "Use parallelization",
        "value": True,
    },
    "n_cpu": {
        "min": 1,
        "group": "Compute",
        "dependency": "parallelized",
        "dependencyType": "enabled",
        "optional": True,
        "enabled": False,
        "label": "Number of cpu",
        "value": None,
    },
    "max_ram": {
        "min": 0,
        "group": "Compute",
        "dependency": "parallelized",
        "dependencyType": "enabled",
        "optional": True,
        "enabled": False,
        "label": "Set RAM limit",
        "value": 2,
    },
    "workspace": {
        "visible": False,
        "enabled": False,
        "label": "Path to workspace",
        "value": None,
    },
    "no_data_value": {
        "default": 0,
        "group": "Data Options",
        "optional": True,
        "enabled": False,
        "visible": False,
        "label": "No data value",
        "value": 0,
    },
    "monitoring_directory": {
        "default": None,
        "enabled": False,
        "value": None,
    },
    "geoh5": {
        "default": None,
        "enabled": False,
        "value": None,
    },
    "run_command": "geoapps.drivers.magnetic_vector_inversion",
    "run_command_boolean": {
        "default": False,
        "value": False,
        "label": "Run python module ",
        "tooltip": "Warning: launches process to run python model on save",
        "main": True,
    },
    "conda_environment": "geoapps",
}

validations = {
    "title": {
        "types": [str],
    },
    "forward_only": {
        "types": [bool],
        "reqs": [
            (True, "starting_model"),
        ],
    },
    "topography_object": {
        "types": [str, UUID],
        "uuid": [],
    },
    "topography": {
        "types": [str, UUID, int, float],
        "reqs": [("topography_object")],
        "uuid": ["topography_object"],
    },
    "data_object": {
        "types": [str, UUID],
    },
    "starting_model_object": {
        "types": [str, UUID],
    },
    "starting_model": {
        "types": [str, UUID, int, float],
    },
    "tile_spatial": {
        "types": [str, int, float],
    },
    "z_from_topo": {"types": [bool]},
    "receivers_radar_drape": {"types": [str], "reqs": [("data_object")]},
    "receivers_offset_x": {
        "types": [int, float],
    },
    "receivers_offset_y": {
        "types": [int, float],
    },
    "receivers_offset_z": {
        "types": [int, float],
    },
    "gps_receivers_offset": {
        "types": [int, float, str],
    },
    "ignore_values": {
        "types": [str],
    },
    "resolution": {
        "types": [int, float],
    },
    "detrend_data": {
        "types": [bool],
    },
    "detrend_order": {
        "types": [int],
        "values": [0, 1, 2],
    },
    "detrend_type": {
        "types": [str],
        "values": ["all", "corners"],
    },
    "max_chunk_size": {"types": [int, float]},
    "chunk_by_rows": {
        "types": [bool],
    },
    "output_tile_files": {
        "types": [bool],
    },
    "mesh": {
        "uuid": [],
        "types": [str, UUID],
    },
    "mesh_from_params": {"types": [bool], "reqs": [(True, "u_cell_size")]},
    "u_cell_size": {
        "types": [int, float],
    },
    "v_cell_size": {
        "types": [int, float],
    },
    "w_cell_size": {
        "types": [int, float],
    },
    "octree_levels_topo": {
        "types": [int, float],
    },
    "octree_levels_obs": {
        "types": [int, float],
    },
    "depth_core": {
        "types": [int, float],
    },
    "max_distance": {
        "types": [int, float],
    },
    "horizontal_padding": {
        "types": [int, float],
    },
    "vertical_padding": {
        "types": [int, float],
    },
    "window_center_x": {
        "types": [int, float],
    },
    "window_center_y": {
        "types": [int, float],
    },
    "window_center_z": {
        "types": [int, float],
    },
    "window_width": {
        "types": [int, float],
    },
    "window_height": {
        "types": [int, float],
    },
    "window_azimuth": {
        "types": [int, float],
    },
    "inversion_style": {
        "types": [str],
        "values": ["voxel"],
    },
    "chi_factor": {
        "types": [int, float],
    },
    "sens_wts_threshold": {
        "types": [int, float],
    },
    "every_iteration_bool": {
        "types": [bool],
    },
    "f_min_change": {
        "types": [int, float],
    },
    "minGNiter": {
        "types": [int, float],
    },
    "beta_tol": {
        "types": [int, float],
    },
    "prctile": {
        "types": [int, float],
    },
    "coolingRate": {
        "types": [int, float],
    },
    "coolEps_q": {
        "types": [bool],
    },
    "coolEpsFact": {
        "types": [int, float],
    },
    "beta_search": {
        "types": [bool],
    },
    "max_iterations": {
        "types": [int, float],
    },
    "max_line_search_iterations": {
        "types": [int, float],
    },
    "max_cg_iterations": {
        "types": [int, float],
    },
    "max_global_iterations": {
        "types": [int, float],
    },
    "initial_beta_ratio": {
        "types": [float],
    },
    "provide_beta": {
        "types": [bool],
    },
    "initial_beta": {
        "types": [int, float],
    },
    "tol_cg": {"types": [int, float]},
    "alpha_s": {
        "types": [int, float],
    },
    "alpha_x": {
        "types": [int, float],
    },
    "alpha_y": {
        "types": [int, float],
    },
    "alpha_z": {
        "types": [int, float],
    },
    "s_norm": {
        "types": [int, float],
    },
    "x_norm": {
        "types": [int, float],
    },
    "y_norm": {
        "types": [int, float],
    },
    "z_norm": {
        "types": [int, float],
    },
    "reference_model_object": {
        "types": [str],
    },
    "reference_model": {
        "types": [str, int, float],
        "reqs": [("reference_model_object")],
    },
    "gradient_type": {
        "types": [str],
        "values": ["total", "components"],
    },
    "lower_bound_object": {
        "types": [str, UUID],
    },
    "lower_bound": {
        "types": [str, int, float, UUID],
    },
    "upper_bound_object": {
        "types": [str, UUID],
    },
    "upper_bound": {
        "types": [str, int, float, UUID],
    },
    "parallelized": {
        "types": [bool],
    },
    "n_cpu": {
        "types": [int, float],
    },
    "max_ram": {
        "types": [int, float],
    },
    "workspace": {
        "types": [str, Workspace],
    },
    "no_data_value": {
        "types": [int, float],
    },
    "monitoring_directory": {
        "types": [str],
    },
    "workspace_geoh5": {
        "types": [str],
    },
    "geoh5": {
        "types": [str, Workspace],
    },
    "run_command": {
        "types": [str],
    },
    "run_command_boolean": {
        "types": [bool],
    },
    "conda_environment": {
        "types": [str],
    },
    "conda_environment_boolean": {
        "types": [bool],
    },
    "out_group": {"types": [str, ContainerGroup]},
}
