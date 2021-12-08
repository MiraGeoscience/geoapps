#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from uuid import UUID

from geoh5py.data import FloatData
from geoh5py.groups import ContainerGroup
from geoh5py.objects import Octree, Points, Surface
from geoh5py.workspace import Workspace

required_parameters = ["data_object", "topography_object", "starting_model"]

default_ui_json = {
    "forward_only": False,
    "topography_object": {
        "main": True,
        "group": "Topography",
        "label": "Object",
        "optional": True,
        "enabled": False,
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
            "{4ea87376-3ece-438b-bf12-3479733ded46}",
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
            "{4ea87376-3ece-438b-bf12-3479733ded46}",
        ],
        "label": "Object",
        "value": None,
    },
    "starting_model": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Starting Model",
        "main": True,
        "isValue": False,
        "parent": "starting_model_object",
        "label": "Value",
        "property": None,
        "value": 0.0,
    },
    "tile_spatial": {
        "group": "Receivers location options",
        "label": "Number of tiles",
        "parent": "data_object",
        "property": None,
        "value": 1,
        "min": 1,
        "max": 1000,
    },
    "output_tile_files": False,
    "z_from_topo": {
        "main": False,
        "group": "Receivers location options",
        "label": "Take z from topography",
        "value": False,
    },
    "receivers_radar_drape": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "main": False,
        "group": "Receivers location options",
        "label": "Drape receivers with radar channel",
        "optional": True,
        "parent": "data_object",
        "value": None,
        "enabled": False,
    },
    "receivers_offset_x": {
        "group": "Receivers location options",
        "main": False,
        "label": "Receiver X offset (m)",
        "value": 0.0,
        "enabled": True,
    },
    "receivers_offset_y": {
        "group": "Receivers location options",
        "main": False,
        "label": "Receiver Y offset (m)",
        "value": 0.0,
        "enabled": True,
    },
    "receivers_offset_z": {
        "group": "Receivers location options",
        "main": False,
        "label": "Receiver Z offset (m)",
        "value": 0.0,
        "enabled": True,
    },
    "gps_receivers_offset": None,
    "ignore_values": {
        "group": "Data pre-processing",
        "optional": True,
        "enabled": False,
        "label": "Values to ignore",
        "value": None,
    },
    "resolution": {
        "min": 0.0,
        "group": "Data pre-processing",
        "optional": True,
        "enabled": False,
        "label": "Downsampling resolution",
        "value": 0.0,
    },
    "detrend_order": {
        "min": 0,
        "group": "Data pre-processing",
        "enabled": False,
        "dependencyType": "enabled",
        "label": "Detrend order",
        "optional": True,
        "value": 0,
    },
    "detrend_type": {
        "choiceList": ["all", "perimeter"],
        "group": "Data pre-processing",
        "dependency": "detrend_order",
        "dependencyType": "enabled",
        "enabled": False,
        "label": "Detrend type",
        "value": "all",
    },
    "max_chunk_size": {
        "min": 0,
        "group": "Data pre-processing",
        "optional": True,
        "enabled": True,
        "label": "Maximum chunk size",
        "value": 128,
    },
    "chunk_by_rows": {
        "group": "Data pre-processing",
        "label": "Chunk by rows",
        "value": True,
    },
    "mesh": {
        "group": "Mesh",
        "main": True,
        "optional": True,
        "enabled": False,
        "dependency": "mesh_from_params",
        "dependencyType": "disable",
        "label": "Mesh",
        "meshType": "4EA87376-3ECE-438B-BF12-3479733DED46",
        "value": None,
    },
    "u_cell_size": {
        "min": 0.0,
        "group": "Mesh",
        "main": True,
        "enabled": True,
        "dependency": "mesh",
        "dependencyType": "disabled",
        "label": "Easting core cell size (m)",
        "value": 25.0,
    },
    "v_cell_size": {
        "min": 0.0,
        "group": "Mesh",
        "main": True,
        "enabled": True,
        "dependency": "mesh",
        "dependencyType": "disabled",
        "label": "Northing core cell size (m)",
        "value": 25.0,
    },
    "w_cell_size": {
        "min": 0.0,
        "group": "Mesh",
        "main": True,
        "enabled": True,
        "dependency": "mesh",
        "dependencyType": "disabled",
        "label": "Vertical core cell size (m)",
        "value": 25.0,
    },
    "octree_levels_topo": {
        "group": "Mesh",
        "main": True,
        "enabled": True,
        "dependency": "mesh",
        "dependencyType": "disabled",
        "label": "Octree levels topography",
        "value": [16, 8, 4, 2],
    },
    "octree_levels_obs": {
        "group": "Mesh",
        "main": True,
        "enabled": True,
        "dependency": "mesh",
        "dependencyType": "disabled",
        "label": "Octree levels observations",
        "value": [4, 4, 4, 4],
    },
    "depth_core": {
        "min": 0.0,
        "group": "Mesh",
        "main": True,
        "enabled": True,
        "dependency": "mesh",
        "dependencyType": "disabled",
        "label": "Depth of core refinement volume",
        "value": 500.0,
    },
    "max_distance": {
        "min": 0.0,
        "group": "Mesh",
        "main": True,
        "enabled": True,
        "dependency": "mesh",
        "dependencyType": "disabled",
        "label": "Maximum padding distance",
        "value": 5000.0,
    },
    "horizontal_padding": {
        "min": 0.0,
        "group": "Mesh",
        "main": True,
        "enabled": True,
        "dependency": "mesh",
        "dependencyType": "disabled",
        "label": "Horizontal padding",
        "value": 1000.0,
    },
    "vertical_padding": {
        "min": 0.0,
        "group": "Mesh",
        "main": True,
        "enabled": True,
        "dependency": "mesh",
        "dependencyType": "disabled",
        "label": "Vertical padding",
        "value": 1000.0,
    },
    "window_center_x": {
        "group": "Data window",
        "enabled": False,
        "groupOptional": True,
        "label": "Window center easting",
        "value": 0.0,
    },
    "window_center_y": {
        "group": "Data window",
        "enabled": False,
        "label": "Window center northing",
        "value": 0.0,
    },
    "window_width": {
        "min": 0.0,
        "group": "Data window",
        "enabled": False,
        "label": "Window width",
        "value": 0.0,
    },
    "window_height": {
        "min": 0.0,
        "group": "Data window",
        "enabled": False,
        "label": "Window height",
        "value": 0.0,
    },
    "window_azimuth": {
        "min": -180,
        "max": 180,
        "group": "Data window",
        "enabled": False,
        "label": "Window azimuth",
        "value": 0.0,
    },
    "inversion_style": "voxel",
    "chi_factor": {
        "min": 0.0,
        "max": 1.0,
        "group": "Optimization",
        "label": "Chi factor",
        "value": 1.0,
        "enabled": True,
    },
    "sens_wts_threshold": {
        "group": "Update sensitivity weights directive",
        "label": "Update sensitivity weight threshold",
        "value": 0.0,
    },
    "every_iteration_bool": {
        "group": "Update sensitivity weights directive",
        "label": "Update every iteration",
        "value": False,
    },
    "f_min_change": {
        "group": "Update IRLS directive",
        "label": "f min change",
        "value": 1e-4,
    },
    "minGNiter": {
        "group": "Update IRLS directive",
        "label": "Minimum Gauss-Newton iterations",
        "value": 1,
    },
    "beta_tol": {
        "group": "Update IRLS directive",
        "label": "Beta tolerance",
        "value": 0.5,
    },
    "prctile": {
        "group": "Update IRLS directive",
        "label": "percentile",
        "value": 95,
    },
    "coolingRate": {
        "group": "Update IRLS directive",
        "label": "Beta cooling rate",
        "value": 1,
    },
    "coolEps_q": {
        "group": "Update IRLS directive",
        "label": "Cool epsilon q",
        "value": True,
    },
    "coolEpsFact": {
        "group": "Update IRLS directive",
        "label": "Cool epsilon fact",
        "value": 1.2,
    },
    "beta_search": {
        "group": "Update IRLS directive",
        "label": "Perform beta search",
        "value": False,
    },
    "starting_chi_factor": {
        "group": "Update IRLS directive",
        "label": "IRLS start chi factor",
        "optional": True,
        "enabled": False,
        "value": 1.0,
        "tooltip": "This chi factor will be used to determine the misfit"
        " threshold after which IRLS iterations begin.",
    },
    "max_iterations": {
        "min": 0,
        "group": "Optimization",
        "label": "Maximum number of IRLS iterations",
        "tooltip": "Incomplete Re-weighted Least Squares iterations for non-L2 problems",
        "value": 25,
        "enabled": True,
    },
    "max_global_iterations": {
        "min": 0,
        "group": "Optimization",
        "label": "Max iterations",
        "tooltip": "Number of L2 and IRLS iterations combined",
        "value": 100,
        "enabled": True,
    },
    "max_line_search_iterations": {
        "group": "Optimization",
        "label": "Maximum number of line searches",
        "value": 20,
        "enabled": True,
    },
    "max_cg_iterations": {
        "min": 0,
        "group": "Optimization",
        "label": "Maximum CG iterations",
        "value": 30,
        "enabled": True,
    },
    "initial_beta_ratio": {
        "min": 0.0,
        "group": "Optimization",
        "optional": True,
        "enabled": True,
        "dependency": "initial_beta",
        "dependencyType": "disabled",
        "label": "Initial beta ratio",
        "value": 100.0,
    },
    "initial_beta": {
        "min": 0.0,
        "group": "Optimization",
        "optional": True,
        "enabled": False,
        "dependency": "initial_beta_ratio",
        "dependencyType": "disabled",
        "label": "Initial beta",
        "value": 1.0,
    },
    "tol_cg": {
        "min": 0,
        "group": "Optimization",
        "label": "Conjugate gradient tolerance",
        "value": 1e-4,
        "enabled": True,
    },
    "alpha_s": {
        "min": 0.0,
        "group": "Regularization",
        "label": "Smallness weight",
        "value": 1.0,
        "enabled": True,
    },
    "alpha_x": {
        "min": 0.0,
        "group": "Regularization",
        "label": "X-smoothness weight",
        "value": 1.0,
        "enabled": True,
    },
    "alpha_y": {
        "min": 0.0,
        "group": "Regularization",
        "label": "Y-smoothness weight",
        "value": 1.0,
        "enabled": True,
    },
    "alpha_z": {
        "min": 0.0,
        "group": "Regularization",
        "label": "Z-smoothness weight",
        "value": 1.0,
        "enabled": True,
    },
    "s_norm": {
        "min": 0.0,
        "max": 2.0,
        "group": "Regularization",
        "label": "Smallness norm",
        "value": 0.0,
        "enabled": True,
    },
    "x_norm": {
        "min": 0.0,
        "max": 2.0,
        "group": "Regularization",
        "label": "X-smoothness norm",
        "value": 2.0,
        "enabled": True,
    },
    "y_norm": {
        "min": 0.0,
        "max": 2.0,
        "group": "Regularization",
        "label": "Y-smoothness norm",
        "value": 2.0,
        "enabled": True,
    },
    "z_norm": {
        "min": 0.0,
        "max": 2.0,
        "group": "Regularization",
        "label": "Z-smoothness norm",
        "value": 2.0,
        "enabled": True,
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
            "{4ea87376-3ece-438b-bf12-3479733ded46}",
        ],
        "value": None,
        "enabled": True,
    },
    "reference_model": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Regularization",
        "isValue": False,
        "parent": "reference_model_object",
        "label": "Reference model value",
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
            "{4ea87376-3ece-438b-bf12-3479733ded46}",
        ],
        "label": "Lower bound object",
        "value": None,
    },
    "lower_bound": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Regularization",
        "isValue": False,
        "parent": "lower_bound_object",
        "label": "Lower bound",
        "property": None,
        "value": 0.0,
    },
    "upper_bound_object": {
        "group": "Regularization",
        "label": "Upper bound object",
        "meshType": [
            "{202C5DB1-A56D-4004-9CAD-BAAFD8899406}",
            "{6A057FDC-B355-11E3-95BE-FD84A7FFCB88}",
            "{F26FEBA3-ADED-494B-B9E9-B2BBCBE298E1}",
            "{48F5054A-1C5C-4CA4-9048-80F36DC60A06}",
            "{b020a277-90e2-4cd7-84d6-612ee3f25051}",
            "{4ea87376-3ece-438b-bf12-3479733ded46}",
        ],
        "value": None,
    },
    "upper_bound": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Regularization",
        "isValue": False,
        "parent": "upper_bound_object",
        "label": "Upper bound",
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
    "max_ram": None,
    "no_data_value": None,
    "monitoring_directory": None,
    "workspace_geoh5": None,
    "geoh5": None,
    "run_command": "geoapps.drivers.magnetic_vector_inversion",
    "run_command_boolean": {
        "value": False,
        "label": "Run python module ",
        "tooltip": "Warning: launches process to run python model on save",
        "main": True,
    },
    "conda_environment": "geoapps",
    "distributed_workers": None,
}

######################## Validations ###########################

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
        "types": [str, UUID, Surface],
        "uuid": [],
    },
    "topography": {
        "types": [str, UUID, int, float, FloatData],
        "reqs": [("topography_object",)],
        "uuid": [],
    },
    "data_object": {
        "types": [str, UUID, Points],
    },
    "starting_model_object": {
        "types": [str, UUID, Octree],
    },
    "starting_model": {
        "types": [str, UUID, int, float, FloatData],
    },
    "tile_spatial": {
        "types": [str, int, float, FloatData],
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
    "detrend_order": {
        "types": [int],
    },
    "detrend_type": {
        "types": [str],
        "values": ["all", "perimeter"],
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
        "types": [str, UUID, Octree],
    },
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
    "starting_chi_factor": {
        "types": [int, float],
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
        "types": [str, UUID, Octree],
    },
    "reference_model": {
        "types": [str, int, float, UUID, FloatData],
        "reqs": [("reference_model_object")],
    },
    "gradient_type": {
        "types": [str],
        "values": ["total", "components"],
    },
    "lower_bound_object": {
        "types": [str, UUID, Octree],
    },
    "lower_bound": {
        "types": [str, int, float, UUID, FloatData],
    },
    "upper_bound_object": {
        "types": [str, UUID, Octree],
    },
    "upper_bound": {
        "types": [str, int, float, UUID, FloatData],
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
    "distributed_workers": {"types": [str, bool]},
}
