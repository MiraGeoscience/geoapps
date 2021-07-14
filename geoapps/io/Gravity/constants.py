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

required_parameters = [
    "inversion_type",
]
defaults = {}

default_ui_json = {
    "inversion_type": {
        "default": "gravity",
        "visible": False,
        "enabled": False,
        "value": "gravity",
    },
    "forward_only": {
        "default": False,
        "main": True,
        "label": "forward model only?",
        "value": False,
    },
    "topography_object": {
        "default": None,
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
        "default": None,
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
        "default": None,
        "main": True,
        "group": "Receivers",
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
    "gz_channel_bool": {
        "default": False,
        "group": "Data",
        "main": True,
        "label": "use Gz?",
        "value": False,
    },
    "gz_channel": {
        "association": "Cell",
        "dataType": "Float",
        "default": None,
        "group": "Data",
        "main": True,
        "dependency": "forward_only",
        "dependencyType": "hide",
        "label": "Channel",
        "parent": "data_object",
        "value": None,
    },
    "gz_uncertainty": {
        "association": "Cell",
        "dataType": "Float",
        "default": None,
        "group": "Data",
        "main": True,
        "dependency": "forward_only",
        "dependencyType": "hide",
        "isValue": True,
        "label": "Uncertainty",
        "parent": "data_object",
        "property": None,
        "value": None,
    },
    "starting_model_object": {
        "default": None,
        "group": "Starting Model",
        "meshType": [
            "{202C5DB1-A56D-4004-9CAD-BAAFD8899406}",
            "{6A057FDC-B355-11E3-95BE-FD84A7FFCB88}",
            "{F26FEBA3-ADED-494B-B9E9-B2BBCBE298E1}",
            "{48F5054A-1C5C-4CA4-9048-80F36DC60A06}",
            "{b020a277-90e2-4cd7-84d6-612ee3f25051}",
        ],
        "label": "starting model object",
        "value": None,
    },
    "starting_model": {
        "association": "Cell",
        "dataType": "Float",
        "default": None,
        "group": "Starting Model",
        "isValue": True,
        "parent": "starting_model_object",
        "label": "starting model value",
        "property": None,
        "value": 0.05,
    },
    "tile_spatial": {
        "association": "Cell",
        "dataType": "Float",
        "default": 1,
        "group": "Receivers Options",
        "dependency": "forward_only",
        "dependencyType": "hide",
        "isValue": True,
        "label": "number of tiles",
        "parent": "data_object",
        "property": None,
        "value": 1,
    },
    "z_from_topo": {
        "default": True,
        "main": True,
        "group": "Receivers Options",
        "label": "Take z from topography?",
        "value": True,
    },
    "receivers_radar_drape": {
        "association": "Cell",
        "dataType": "Float",
        "default": None,
        "main": True,
        "group": "Receivers Options",
        "dependency": "forward_only",
        "dependencyType": "hide",
        "label": "drape receivers with radar channel",
        "parent": "data_object",
        "value": None,
    },
    "receivers_offset_x": {
        "default": 0,
        "group": "Receivers Options",
        "main": True,
        "dependency": "forward_only",
        "dependencyType": "hide",
        "label": "constant receiver offset in x",
        "value": 0,
    },
    "receivers_offset_y": {
        "default": 0,
        "group": "Receivers Options",
        "main": True,
        "dependency": "forward_only",
        "dependencyType": "hide",
        "label": "constant receiver offset in y",
        "value": 0,
    },
    "receivers_offset_z": {
        "default": 0,
        "group": "Receivers Options",
        "main": True,
        "dependency": "forward_only",
        "dependencyType": "hide",
        "label": "constant receiver offset in z",
        "value": 0,
    },
    "gps_receivers_offset": {
        "association": "Cell",
        "dataType": "Float",
        "default": None,
        "group": "Receivers Options",
        "enabled": False,
        "visible": False,
        "dependency": "forward_only",
        "dependencyType": "hide",
        "isValue": False,
        "label": "set data offsets",
        "parent": "data_object",
        "property": None,
        "value": 0.0,
    },
    "ignore_values": {
        "default": None,
        "group": "Data Options",
        "optional": True,
        "enabled": False,
        "dependency": "forward_only",
        "dependencyType": "hide",
        "label": "values to ignore",
        "value": None,
    },
    "resolution": {
        "min": 0.0,
        "default": None,
        "group": "Data Options",
        "optional": True,
        "enabled": False,
        "dependency": "forward_only",
        "dependencyType": "hide",
        "label": "resolution",
        "value": 0.0,
    },
    "detrend_data": {
        "default": False,
        "group": "Detrend Data",
        "label": "remove trend from data",
        "value": False,
    },
    "detrend_order": {
        "choiceList": [0, 1, 2],
        "default": 0,
        "group": "Detrend Data",
        "enabled": False,
        "dependency": "detrend_data",
        "dependencyType": "enabled",
        "label": "detrend order",
        "value": 0,
    },
    "detrend_type": {
        "choiceList": ["all", "corners"],
        "default": "all",
        "group": "Detrend Data",
        "dependency": "detrend_data",
        "dependencyType": "enabled",
        "enabled": False,
        "label": "detrend type",
        "value": "all",
    },
    "max_chunk_size": {
        "default": 128,
        "min": 0,
        "group": "Data Options",
        "optional": True,
        "enabled": False,
        "dependency": "forward_only",
        "dependencyType": "hide",
        "label": "maximum chunk size",
        "value": 128,
    },
    "chunk_by_rows": {
        "default": False,
        "group": "Data Options",
        "dependency": "forward_only",
        "dependencyType": "hide",
        "label": "chunk by rows?",
        "value": False,
    },
    "output_tile_files": {
        "default": False,
        "group": "Data Options",
        "dependency": "forward_only",
        "dependencyType": "hide",
        "label": "output tile files?",
        "value": False,
    },
    "mesh": {
        "default": None,
        "group": "Mesh",
        "label": "mesh",
        "meshType": "4EA87376-3ECE-438B-BF12-3479733DED46",
        "value": None,
    },
    "mesh_from_params": {
        "default": False,
        "group": "Mesh",
        "label": "build from parameters?",
        "value": False,
    },
    "core_cell_size_x": {
        "default": None,
        "min": 0.0,
        "group": "Mesh",
        "visible": False,
        "dependency": "mesh_from_params",
        "dependencyType": "show",
        "label": "core cell size in x",
        "value": 2.0,
    },
    "core_cell_size_y": {
        "default": None,
        "min": 0.0,
        "group": "Mesh",
        "visible": False,
        "dependency": "mesh_from_params",
        "dependencyType": "show",
        "label": "core cell size in y",
        "value": 2.0,
    },
    "core_cell_size_z": {
        "default": None,
        "min": 0.0,
        "group": "Mesh",
        "visible": False,
        "dependency": "mesh_from_params",
        "dependencyType": "show",
        "label": "core cell size in z",
        "value": 2.0,
    },
    "octree_levels_topo": {
        "default": [0, 1],
        "group": "Mesh",
        "enabled": False,
        "visible": False,
        "dependency": "mesh_from_params",
        "dependencyType": "show",
        "optional": True,
        "label": "octree levels topography",
        "value": [0, 1],
    },
    "octree_levels_obs": {
        "default": [5, 5],
        "group": "Mesh",
        "enabled": False,
        "visible": False,
        "dependency": "mesh_from_params",
        "dependencyType": "show",
        "optional": True,
        "label": "octree levels observations",
        "value": [5, 5],
    },
    "octree_levels_padding": {
        "default": [2, 2],
        "group": "Mesh",
        "enabled": False,
        "visible": False,
        "dependency": "mesh_from_params",
        "dependencyType": "show",
        "optional": True,
        "label": "octree levels padding",
        "value": [2, 2],
    },
    "depth_core": {
        "default": 100,
        "min": 0,
        "group": "Mesh",
        "enabled": False,
        "visible": False,
        "dependency": "mesh_from_params",
        "dependencyType": "show",
        "optional": True,
        "label": "depth of core refinement volume",
        "value": 100.0,
    },
    "max_distance": {
        "default": np.inf,
        "min": 0,
        "group": "Mesh",
        "enabled": False,
        "visible": False,
        "dependency": "mesh_from_params",
        "dependencyType": "show",
        "optional": True,
        "label": "maximum padding distance",
        "value": np.inf,
    },
    "padding_distance_x": {
        "default": [0, 0],
        "group": "Mesh",
        "enabled": False,
        "visible": False,
        "dependency": "mesh_from_params",
        "dependencyType": "show",
        "label": "padding distance in x",
        "optional": True,
        "value": [0, 0],
    },
    "padding_distance_y": {
        "default": [0, 0],
        "group": "Mesh",
        "enabled": False,
        "visible": False,
        "dependency": "mesh_from_params",
        "dependencyType": "show",
        "label": "padding distance in y",
        "optional": True,
        "value": [0, 0],
    },
    "padding_distance_z": {
        "default": [0, 0],
        "group": "Mesh",
        "enabled": False,
        "visible": False,
        "dependency": "mesh_from_params",
        "dependencyType": "show",
        "label": "padding distance in z",
        "optional": True,
        "value": [0, 0],
    },
    "window_center_x": {
        "default": None,
        "group": "window",
        "optional": True,
        "enabled": False,
        "label": "window center easting",
        "value": 0.0,
    },
    "window_center_y": {
        "default": None,
        "group": "window",
        "optional": True,
        "enabled": False,
        "label": "window center northing",
        "value": 0.0,
    },
    "window_width": {
        "default": None,
        "min": 0.0,
        "group": "window",
        "optional": True,
        "enabled": False,
        "label": "window width",
        "value": 0.0,
    },
    "window_height": {
        "default": None,
        "min": 0.0,
        "group": "window",
        "optional": True,
        "enabled": False,
        "label": "window height",
        "value": 0.0,
    },
    "window_azimuth": {
        "default": None,
        "group": "window",
        "visible": False,
        "enabled": False,
        "label": "window azimuth",
        "value": 0.0,
    },
    "inversion_style": {
        "choiceList": ["voxel"],
        "group": "Optimization",
        "default": "voxel",
        "dependency": "forward_only",
        "dependencyType": "hide",
        "label": "inversion style",
        "value": "voxel",
    },
    "chi_factor": {
        "default": 1.0,
        "min": 0.0,
        "max": 1.0,
        "group": "Optimization",
        "dependency": "forward_only",
        "dependencyType": "hide",
        "label": "chi factor",
        "value": 1.0,
    },
    "max_iterations": {
        "default": 20,
        "min": 0,
        "group": "Optimization",
        "dependency": "forward_only",
        "dependencyType": "hide",
        "label": "max iteration count",
        "value": 20,
    },
    "max_cg_iterations": {
        "default": 10,
        "min": 0,
        "group": "Optimization",
        "dependency": "forward_only",
        "dependencyType": "hide",
        "label": "max conjugate gradient iteration count",
        "value": 10,
    },
    "max_global_iterations": {
        "default": 100,
        "min": 0,
        "group": "Optimization",
        "dependency": "forward_only",
        "dependencyType": "hide",
        "label": "max global iteration count",
        "value": 100,
    },
    "initial_beta_ratio": {
        "default": 1e1,
        "min": 0.0,
        "group": "Optimization",
        "optional": True,
        "enabled": False,
        "dependency": "initial_beta",
        "dependencyType": "disabled",
        "label": "initial beta ratio",
        "value": 1e1,
    },
    "initial_beta": {
        "default": None,
        "min": 0.0,
        "group": "Optimization",
        "optional": True,
        "enabled": False,
        "dependency": "provide_beta",
        "dependencyType": "enabled",
        "label": "initial beta",
        "value": 0.0,
    },
    "tol_cg": {
        "default": 1e-16,
        "min": 0,
        "group": "Optimization",
        "dependency": "forward_only",
        "dependencyType": "hide",
        "label": "conjugate gradient tolerance",
        "value": 1e-16,
    },
    "alpha_s": {
        "default": 1.0,
        "min": 0.0,
        "group": "Regularization",
        "dependency": "forward_only",
        "dependencyType": "hide",
        "label": "smallness weight",
        "value": 1.0,
    },
    "alpha_x": {
        "default": 1.0,
        "min": 0.0,
        "group": "Regularization",
        "dependency": "forward_only",
        "dependencyType": "hide",
        "label": "x-smoothness weight",
        "value": 1.0,
    },
    "alpha_y": {
        "default": 1.0,
        "min": 0.0,
        "group": "Regularization",
        "dependency": "forward_only",
        "dependencyType": "hide",
        "label": "y-smoothness weight",
        "value": 1.0,
    },
    "alpha_z": {
        "default": 1.0,
        "min": 0.0,
        "group": "Regularization",
        "dependency": "forward_only",
        "dependencyType": "hide",
        "label": "z-smoothness weight",
        "value": 1.0,
    },
    "smallness_norm": {
        "default": 2.0,
        "min": 0.0,
        "max": 2.0,
        "group": "Regularization",
        "dependency": "forward_only",
        "dependencyType": "hide",
        "label": "smallness norm",
        "value": 2.0,
    },
    "x_norm": {
        "default": 2.0,
        "min": 0.0,
        "max": 2.0,
        "group": "Regularization",
        "dependency": "forward_only",
        "dependencyType": "hide",
        "label": "x-smoothness norm",
        "value": 2.0,
    },
    "y_norm": {
        "default": 2.0,
        "min": 0.0,
        "max": 2.0,
        "group": "Regularization",
        "dependency": "forward_only",
        "dependencyType": "hide",
        "label": "y-smoothness norm",
        "value": 2.0,
    },
    "z_norm": {
        "default": 2.0,
        "min": 0.0,
        "max": 2.0,
        "group": "Regularization",
        "dependency": "forward_only",
        "dependencyType": "hide",
        "label": "z-smoothness norm",
        "value": 2.0,
    },
    "reference_model_object": {
        "default": None,
        "group": "Models",
        "visible": True,
        "dependency": "forward_only",
        "dependencyType": "hide",
        "label": "reference model object",
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
        "default": 0.0,
        "group": "Models",
        "isValue": True,
        "visible": True,
        "dependency": "forward_only",
        "dependencyType": "hide",
        "label": "reference model value",
        "parent": "reference_model_object",
        "property": None,
        "value": 0.0,
    },
    "gradient_type": {
        "choiceList": ["total", "components"],
        "default": "total",
        "group": "Regularization",
        "dependency": "forward_only",
        "dependencyType": "hide",
        "label": "gradient type",
        "value": "total",
    },
    "lower_bound_object": {
        "default": None,
        "group": "Regularization",
        "visible": True,
        "dependency": "forward_only",
        "dependencyType": "hide",
        "label": "lower bound object",
        "meshType": [
            "{202C5DB1-A56D-4004-9CAD-BAAFD8899406}",
            "{6A057FDC-B355-11E3-95BE-FD84A7FFCB88}",
            "{F26FEBA3-ADED-494B-B9E9-B2BBCBE298E1}",
            "{48F5054A-1C5C-4CA4-9048-80F36DC60A06}",
            "{b020a277-90e2-4cd7-84d6-612ee3f25051}",
        ],
        "value": None,
    },
    "lower_bound": {
        "association": "Cell",
        "dataType": "Float",
        "default": -1,
        "group": "Regularization",
        "isValue": True,
        "visible": True,
        "dependency": "forward_only",
        "dependencyType": "hide",
        "label": "lower bound on model",
        "parent": "lower_bound_object",
        "property": None,
        "value": -1,
    },
    "upper_bound_object": {
        "default": None,
        "group": "Regularization",
        "visible": True,
        "dependency": "forward_only",
        "dependencyType": "hide",
        "label": "upper bound object",
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
        "default": 1,
        "group": "Regularization",
        "isValue": True,
        "visible": True,
        "dependency": "forward_only",
        "dependencyType": "hide",
        "label": "upper bound on model",
        "parent": "upper_bound_object",
        "property": None,
        "value": 1,
    },
    "parallelized": {
        "default": True,
        "group": "Compute",
        "label": "use parallelization",
        "value": True,
    },
    "n_cpu": {
        "default": None,
        "min": 1,
        "group": "Compute",
        "dependency": "parallelized",
        "dependencyType": "enabled",
        "optional": True,
        "enabled": False,
        "label": "number of cpu",
        "value": None,
    },
    "max_ram": {
        "default": 2,
        "min": 0,
        "group": "Compute",
        "dependency": "parallelized",
        "dependencyType": "enabled",
        "optional": True,
        "enabled": False,
        "label": "set RAM limit",
        "value": 2,
    },
    "workspace": {
        "default": None,
        "visible": False,
        "enabled": False,
        "label": "path to workspace",
        "value": None,
    },
    "output_geoh5": {
        "default": None,
        "visible": False,
        "enabled": False,
        "label": "path to results geoh5py file",
        "value": None,
    },
    "out_group": {
        "default": "GravInversion",
        "visible": True,
        "enabled": True,
        "label": "results group name",
        "value": "GravInversion",
    },
    "no_data_value": {
        "default": 0,
        "group": "Data Options",
        "optional": True,
        "enabled": False,
        "visible": False,
        "label": "no data value",
        "value": 0,
    },
    "monitoring_directory": {
        "default": None,
        "enabled": False,
        "value": None,
    },
    "workspace_geoh5": {
        "default": None,
        "enabled": False,
        "value": None,
    },
    "geoh5": {
        "default": None,
        "enabled": False,
        "value": None,
    },
    "run_command": "geoapps.drivers.grav_inversion",
    "run_command_boolean": {
        "default": False,
        "value": False,
        "label": "Run python module ",
        "tooltip": "Warning: launches process to run python model on save",
        "main": True,
    },
    "conda_environment": "geoapps",
}

required_parameters = []

validations = {
    "inversion_type": {
        "types": [str],
        "values": ["gravity"],
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
    "gz_channel_bool": {"types": [bool]},
    "gz_channel": {
        "types": [str, UUID],
        "reqs": [("data_object"), (True, "gz_channel_bool")],
    },
    "gz_uncertainty": {"types": [str, int, float], "reqs": [(True, "gz_channel_bool")]},
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
    "mesh_from_params": {"types": [bool], "reqs": [(True, "core_cell_size_x")]},
    "core_cell_size_x": {
        "types": [int, float],
    },
    "core_cell_size_y": {
        "types": [int, float],
    },
    "core_cell_size_z": {
        "types": [int, float],
    },
    "octree_levels_topo": {
        "types": [int, float],
    },
    "octree_levels_obs": {
        "types": [int, float],
    },
    "octree_levels_padding": {
        "types": [int, float],
    },
    "depth_core": {
        "types": [int, float],
    },
    "max_distance": {
        "types": [int, float],
    },
    "padding_distance_x": {
        "types": [int, float],
    },
    "padding_distance_y": {
        "types": [int, float],
    },
    "padding_distance_z": {
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
    "max_iterations": {
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
    "smallness_norm": {
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
        "types": [str],
    },
    "lower_bound": {
        "types": [str, int, float, UUID],
    },
    "upper_bound_object": {
        "types": [str],
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
    "output_geoh5": {
        "types": [str, Workspace],
    },
    "out_group": {"types": [str, ContainerGroup]},
    "no_data_value": {
        "types": [int, float],
    },
    "monitoring_directory": {
        "types": [str],
    },
    "workspace_geoh5": {
        "types": [str, Workspace],
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
}
