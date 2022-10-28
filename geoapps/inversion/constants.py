#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from uuid import UUID

from geoh5py.objects import Curve, Grid2D, Points, Surface

default_ui_json = {
    "forward_only": False,
    "topography_object": {
        "main": True,
        "group": "Topography",
        "label": "Topography",
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
        "association": ["Vertex", "Cell"],
        "dataType": "Float",
        "group": "Topography",
        "main": True,
        "optional": True,
        "enabled": False,
        "isValue": False,
        "label": "Elevation adjustment",
        "tooltip": "Adjust elevation given from topography object",
        "parent": "topography_object",
        "property": None,
        "value": 0.0,
    },
    "data_object": {
        "main": True,
        "group": "Survey",
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
    "starting_model": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Mesh and Models",
        "main": True,
        "isValue": True,
        "parent": "mesh",
        "label": "Initial Density (g/cc)",
        "property": None,
        "value": 1e-4,
    },
    "tile_spatial": {
        "group": "Compute",
        "label": "Number of tiles",
        "parent": "data_object",
        "isValue": True,
        "property": None,
        "value": 1,
        "min": 1,
        "max": 1000,
    },
    "output_tile_files": False,
    "z_from_topo": {
        "main": True,
        "group": "Survey",
        "label": "Take z from topography",
        "tooltip": "Sets survey elevation to topography before any offsets are applied.",
        "value": False,
    },
    "receivers_offset_x": {
        "group": "Survey",
        "main": True,
        "label": "Receiver X offset (m)",
        "optional": True,
        "enabled": False,
        "value": 0.0,
        "visible": False,
    },
    "receivers_offset_y": {
        "group": "Survey",
        "main": True,
        "label": "Receiver Y offset (m)",
        "optional": True,
        "enabled": False,
        "value": 0.0,
        "visible": False,
    },
    "receivers_offset_z": {
        "group": "Survey",
        "main": True,
        "label": "Z static offset",
        "optional": True,
        "enabled": False,
        "value": 0.0,
    },
    "receivers_radar_drape": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "main": True,
        "group": "Survey",
        "label": "Z radar offset",
        "tooltip": "Apply a non-homogeneous offset to survey object from radar channel.",
        "optional": True,
        "parent": "data_object",
        "value": None,
        "enabled": False,
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
        "group": "Survey",
        "main": True,
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
        "optional": True,
        "label": "Detrend type",
        "value": "all",
    },
    "mesh": {
        "group": "Mesh and Models",
        "main": True,
        "optional": True,
        "enabled": False,
        "label": "Mesh",
        "meshType": "4EA87376-3ECE-438B-BF12-3479733DED46",
        "value": None,
    },
    "u_cell_size": {
        "min": 0.0,
        "group": "Mesh and Models",
        "main": True,
        "enabled": True,
        "dependency": "mesh",
        "dependencyType": "disabled",
        "label": "Easting core cell size (m)",
        "value": 25.0,
    },
    "v_cell_size": {
        "min": 0.0,
        "group": "Mesh and Models",
        "main": True,
        "enabled": True,
        "dependency": "mesh",
        "dependencyType": "disabled",
        "label": "Northing core cell size (m)",
        "value": 25.0,
    },
    "w_cell_size": {
        "min": 0.0,
        "group": "Mesh and Models",
        "main": True,
        "enabled": True,
        "dependency": "mesh",
        "dependencyType": "disabled",
        "label": "Vertical core cell size (m)",
        "value": 25.0,
    },
    "octree_levels_topo": {
        "group": "Mesh and Models",
        "main": True,
        "enabled": True,
        "dependency": "mesh",
        "dependencyType": "disabled",
        "label": "octree levels topography",
        "value": [0, 0, 4, 4],
    },
    "octree_levels_obs": {
        "group": "Mesh and Models",
        "main": True,
        "enabled": True,
        "dependency": "mesh",
        "dependencyType": "disabled",
        "label": "octree levels observations",
        "value": [4, 4, 4, 4],
    },
    "depth_core": {
        "min": 0.0,
        "group": "Mesh and Models",
        "main": True,
        "enabled": True,
        "dependency": "mesh",
        "dependencyType": "disabled",
        "label": "Depth of core refinement volume",
        "value": 500.0,
    },
    "max_distance": {
        "min": 0.0,
        "group": "Mesh and Models",
        "main": True,
        "enabled": True,
        "dependency": "mesh",
        "dependencyType": "disabled",
        "label": "Maximum padding distance",
        "value": 5000.0,
    },
    "horizontal_padding": {
        "min": 0.0,
        "group": "Mesh and Models",
        "main": True,
        "enabled": True,
        "dependency": "mesh",
        "dependencyType": "disabled",
        "label": "Horizontal padding",
        "value": 1000.0,
    },
    "vertical_padding": {
        "min": 0.0,
        "group": "Mesh and Models",
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
        "tooltip": "Update sensitivity weight threshold",
        "label": "Threshold (%)",
        "value": 30.0,
        "max": 99.9,
        "min": 0.0,
        "precision": 1,
        "lineEdit": False,
    },
    "every_iteration_bool": {
        "group": "Update sensitivity weights directive",
        "tooltip": "Update weights at every iteration",
        "label": "Every iteration",
        "value": False,
    },
    "f_min_change": {
        "group": "Update IRLS directive",
        "label": "f min change",
        "value": 1e-4,
        "min": 1e-6,
    },
    "beta_tol": {
        "group": "Update IRLS directive",
        "label": "Beta tolerance",
        "value": 0.5,
        "min": 0.0001,
    },
    "prctile": {
        "group": "Update IRLS directive",
        "label": "Percentile",
        "value": 95,
        "max": 100,
        "min": 5,
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
    "max_global_iterations": {
        "min": 1,
        "lineEdit": False,
        "group": "Optimization",
        "label": "Maximum iterations",
        "tooltip": "Number of L2 and IRLS iterations combined",
        "value": 100,
        "enabled": True,
    },
    "max_irls_iterations": {
        "min": 0,
        "group": "Update IRLS directive",
        "label": "Maximum number of IRLS iterations",
        "tooltip": "Incomplete Re-weighted Least Squares iterations for non-L2 problems",
        "value": 25,
        "enabled": True,
    },
    "coolingRate": {
        "group": "Optimization",
        "label": "Iterations per beta",
        "value": 1,
        "min": 1,
    },
    "coolingFactor": {
        "group": "Optimization",
        "label": "Beta cooling factor",
        "tooltip": "Each beta cooling step will be calculated by dividing the current beta by this factor.",
        "value": 2.0,
        "min": 1.0,
    },
    "max_line_search_iterations": {
        "group": "Optimization",
        "label": "Maximum number of line searches",
        "value": 20,
        "min": 1,
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
        "precision": 2,
        "group": "Optimization",
        "optional": True,
        "enabled": True,
        "label": "Initial beta ratio",
        "value": 100.0,
    },
    "initial_beta": {
        "min": 0.0,
        "group": "Optimization",
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
        "tooltip": "Constant ratio compared to other weights. Larger values result in models that remain close to the reference model",
        "enabled": True,
    },
    "alpha_x": {
        "min": 0.0,
        "group": "Regularization",
        "label": "X-smoothness weight",
        "tooltip": "Larger values relative to other smoothness weights will result in x biased smoothness",
        "value": 1.0,
        "enabled": True,
    },
    "alpha_y": {
        "min": 0.0,
        "group": "Regularization",
        "label": "Y-smoothness weight",
        "tooltip": "Larger values relative to other smoothness weights will result in y biased smoothness",
        "value": 1.0,
        "enabled": True,
    },
    "alpha_z": {
        "min": 0.0,
        "group": "Regularization",
        "label": "Z-smoothness weight",
        "tooltip": "Larger values relative to other smoothness weights will result in z biased smoothess",
        "value": 1.0,
        "enabled": True,
    },
    "s_norm": {
        "min": 0.0,
        "max": 2.0,
        "group": "Regularization",
        "label": "Smallness norm",
        "value": 0.0,
        "precision": 2,
        "lineEdit": False,
        "enabled": True,
    },
    "x_norm": {
        "min": 0.0,
        "max": 2.0,
        "group": "Regularization",
        "label": "X-smoothness norm",
        "value": 2.0,
        "precision": 2,
        "lineEdit": False,
        "enabled": True,
    },
    "y_norm": {
        "min": 0.0,
        "max": 2.0,
        "group": "Regularization",
        "label": "Y-smoothness norm",
        "value": 2.0,
        "precision": 2,
        "lineEdit": False,
        "enabled": True,
    },
    "z_norm": {
        "min": 0.0,
        "max": 2.0,
        "group": "Regularization",
        "label": "Z-smoothness norm",
        "value": 2.0,
        "precision": 2,
        "lineEdit": False,
        "enabled": True,
    },
    "reference_model": {
        "association": ["Cell", "Vertex"],
        "main": True,
        "dataType": "Float",
        "group": "Mesh and Models",
        "isValue": True,
        "parent": "mesh",
        "label": "Reference",
        "property": None,
        "value": 0.0,
    },
    "gradient_type": {
        "choiceList": ["total", "components"],
        "group": "Regularization",
        "label": "Gradient type",
        "value": "total",
    },
    "lower_bound": {
        "association": ["Cell", "Vertex"],
        "main": True,
        "dataType": "Float",
        "group": "Mesh and Models",
        "isValue": True,
        "parent": "mesh",
        "label": "Lower bound",
        "property": None,
        "optional": True,
        "value": -10.0,
        "enabled": False,
    },
    "upper_bound": {
        "association": ["Cell", "Vertex"],
        "main": True,
        "dataType": "Float",
        "group": "Mesh and Models",
        "isValue": True,
        "parent": "mesh",
        "label": "Upper bound",
        "property": None,
        "optional": True,
        "value": 10.0,
        "enabled": False,
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
        "value": 1,
    },
    "store_sensitivities": {
        "choiceList": ["disk", "ram"],
        "group": "Compute",
        "label": "Storage device",
        "value": "disk",
    },
    "max_chunk_size": {
        "min": 0,
        "group": "Compute",
        "optional": True,
        "enabled": True,
        "label": "Maximum chunk size",
        "value": 128,
    },
    "chunk_by_rows": {
        "group": "Compute",
        "label": "Chunk by rows",
        "value": True,
    },
    "max_ram": None,
    "monitoring_directory": None,
    "workspace_geoh5": None,
    "geoh5": None,
    "run_command": "geoapps.inversion.driver",
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
    "topography_object": {
        "types": [str, UUID, Surface, Points, Grid2D, Curve],
    },
    "alpha_s": {"types": [int, float]},
    "alpha_x": {"types": [int, float]},
    "alpha_y": {"types": [int, float]},
    "alpha_z": {"types": [int, float]},
    "norm_s": {"types": [int, float]},
    "norm_x": {"types": [int, float]},
    "norm_y": {"types": [int, float]},
    "norm_z": {"types": [int, float]},
    "distributed_workers": {"types": [tuple, type(None)]},
}
