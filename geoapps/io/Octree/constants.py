#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from uuid import UUID

import numpy as np
from geoh5py.workspace import Workspace

required_parameters = [
    "inversion_type",
]
defaults = {}

default_ui_json = {
    "title": "Octree Mesh Creator",
    "geoh5": "../../assets/FlinFlon.geoh5",
    "extent": {
        "enabled": True,
        "group": "1- Core",
        "label": "Core hull extent",
        "main": True,
        "meshType": [
            "{202C5DB1-A56D-4004-9CAD-BAAFD8899406}",
            "{6A057FDC-B355-11E3-95BE-FD84A7FFCB88}",
            "{F26FEBA3-ADED-494B-B9E9-B2BBCBE298E1}",
        ],
        "value": "{656acd40-25de-4865-814c-cb700f6ee51a}",
    },
    "u_cell_size": {
        "enabled": True,
        "group": "2- Core cell size",
        "label": "Easting (m)",
        "main": True,
        "value": 25,
    },
    "v_cell_size": {
        "enabled": True,
        "group": "2- Core cell size",
        "label": "Northing (m)",
        "main": True,
        "value": 25,
    },
    "w_cell_size": {
        "enabled": True,
        "group": "2- Core cell size",
        "label": "Vertical (m)",
        "main": True,
        "value": 25,
    },
    "horizontal_padding": {
        "enabled": True,
        "group": "3- Padding distance",
        "label": "Horizontal (m)",
        "main": True,
        "value": 1000.0,
    },
    "vertical_padding": {
        "enabled": True,
        "group": "3- Padding distance",
        "label": "Vertical (m)",
        "main": True,
        "value": 1000.0,
    },
    "depth_core": {
        "enabled": True,
        "group": "1- Core",
        "label": "Minimum Depth (m)",
        "main": True,
        "value": 500.0,
    },
    "ga_group_name": {
        "enabled": True,
        "group": "",
        "label": "Name:",
        "value": "Octree_Mesh",
    },
    "Refinement A Object": {
        "enabled": True,
        "group": "Refinement A",
        "label": "Object",
        "meshType": [
            "{202C5DB1-A56D-4004-9CAD-BAAFD8899406}",
            "{6A057FDC-B355-11E3-95BE-FD84A7FFCB88}",
            "{F26FEBA3-ADED-494B-B9E9-B2BBCBE298E1}",
        ],
        "value": "{656acd40-25de-4865-814c-cb700f6ee51a}",
    },
    "Refinement A Levels": {
        "enabled": True,
        "group": "Refinement A",
        "label": "Levels",
        "value": "4,4,4",
    },
    "Refinement A Type": {
        "choiceList": ["surface", "radial"],
        "enabled": True,
        "group": "Refinement A",
        "label": "Type",
        "value": "radial",
    },
    "Refinement A Max Distance": {
        "enabled": True,
        "group": "Refinement A",
        "label": "Max Distance",
        "value": 1000.0,
    },
    "Refinement B Object": {
        "enabled": True,
        "group": "Refinement B",
        "label": "Object",
        "meshType": [
            "{202C5DB1-A56D-4004-9CAD-BAAFD8899406}",
            "{6A057FDC-B355-11E3-95BE-FD84A7FFCB88}",
            "{F26FEBA3-ADED-494B-B9E9-B2BBCBE298E1}",
        ],
        "value": "",
    },
    "Refinement B Levels": {
        "enabled": True,
        "group": "Refinement B",
        "label": "Levels",
        "value": "0,0,2",
    },
    "Refinement B Type": {
        "choiceList": ["surface", "radial"],
        "enabled": True,
        "group": "Refinement B",
        "label": "Type",
        "value": "surface",
    },
    "Refinement B Max Distance": {
        "enabled": True,
        "group": "Refinement B",
        "label": "Max Distance",
        "value": 1000.0,
    },
    "run_command": ("geoapps.create.octree_mesh"),
    "monitoring_directory": "",
}

required_parameters = []

validations = {
    "inversion_type": {
        "types": [str],
        "values": ["mvi", "mvic"],
        "reqs": [
            ("mvi", "inducing_field_strength"),
            ("mvi", "inducing_field_inclination"),
            ("mvi", "inducing_field_declination"),
        ],
    },
    "forward_only": {
        "types": [bool],
        "reqs": [
            (True, "starting_model"),
        ],
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
        "types": [str],
    },
    "tmi_channel": {"types": [str], "reqs": [("data_object")]},
    "tmi_uncertainty": {
        "types": [str, int, float],
    },
    "starting_model_object": {
        "types": [str],
    },
    "starting_inclination_object": {
        "types": [str],
    },
    "starting_declination_object": {
        "types": [str],
    },
    "starting_model": {
        "types": [str, int, float],
    },
    "starting_inclination": {
        "types": [str, int, float],
    },
    "starting_declination": {
        "types": [str, int, float],
    },
    "tile_spatial": {
        "types": [str, int, float],
    },
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
    "reference_inclination_object": {
        "types": [str],
    },
    "reference_declination_object": {
        "types": [str],
    },
    "reference_model": {
        "types": [str, int, float],
        "reqs": [("reference_model_object")],
    },
    "reference_inclination": {
        "types": [str, int, float],
        "reqs": [("reference_inclination_object")],
    },
    "reference_declination": {
        "types": [str, int, float],
        "reqs": [("reference_declination_object")],
    },
    "gradient_type": {
        "types": [str],
        "values": ["total", "components"],
    },
    "lower_bound": {
        "types": [int, float],
    },
    "upper_bound": {
        "types": [int, float],
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
    "out_group": {"types": [str]},
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
