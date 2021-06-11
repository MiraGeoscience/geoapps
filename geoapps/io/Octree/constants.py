#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from uuid import UUID

from geoh5py.workspace import Workspace

required_parameters = [
    "inversion_type",
]
defaults = {}

default_ui_json = {
    "title": "Octree Mesh Creator",
    "geoh5": "../../assets/FlinFlon.geoh5",
    "objects": {
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
        "default": "{656acd40-25de-4865-814c-cb700f6ee51a}",
    },
    "u_cell_size": {
        "enabled": True,
        "group": "2- Core cell size",
        "label": "Easting (m)",
        "main": True,
        "value": 25,
        "default": 25,
    },
    "v_cell_size": {
        "enabled": True,
        "group": "2- Core cell size",
        "label": "Northing (m)",
        "main": True,
        "value": 25,
        "default": 25,
    },
    "w_cell_size": {
        "enabled": True,
        "group": "2- Core cell size",
        "label": "Vertical (m)",
        "main": True,
        "value": 25,
        "default": 25,
    },
    "horizontal_padding": {
        "enabled": True,
        "group": "3- Padding distance",
        "label": "Horizontal (m)",
        "main": True,
        "value": 1000.0,
        "default": 1000.0,
    },
    "vertical_padding": {
        "enabled": True,
        "group": "3- Padding distance",
        "label": "Vertical (m)",
        "main": True,
        "value": 1000.0,
        "default": 1000.0,
    },
    "depth_core": {
        "enabled": True,
        "group": "1- Core",
        "label": "Minimum Depth (m)",
        "main": True,
        "value": 500.0,
        "default": 500.0,
    },
    "ga_group_name": {
        "enabled": True,
        "group": "",
        "label": "Name:",
        "value": "Octree_Mesh",
        "default": "Octree_Mesh",
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
        "default": "{656acd40-25de-4865-814c-cb700f6ee51a}",
    },
    "Refinement A Levels": {
        "enabled": True,
        "group": "Refinement A",
        "label": "Levels",
        "value": "4,4,4",
        "default": "4,4,4",
    },
    "Refinement A Type": {
        "choiceList": ["surface", "radial"],
        "enabled": True,
        "group": "Refinement A",
        "label": "Type",
        "value": "radial",
        "default": "radial",
    },
    "Refinement A Distance": {
        "enabled": True,
        "group": "Refinement A",
        "label": "Distance",
        "value": 1000.0,
        "default": 1000.0,
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
        "default": "",
    },
    "Refinement B Levels": {
        "enabled": True,
        "group": "Refinement B",
        "label": "Levels",
        "value": "0,0,2",
        "default": "0,0,2",
    },
    "Refinement B Type": {
        "choiceList": ["surface", "radial"],
        "enabled": True,
        "group": "Refinement B",
        "label": "Type",
        "value": "surface",
        "default": "surface",
    },
    "Refinement B Distance": {
        "enabled": True,
        "group": "Refinement B",
        "label": "Distance",
        "value": 1000.0,
        "default": 1000.0,
    },
    "run_command": ("geoapps.create.octree_mesh"),
    "monitoring_directory": "",
    "conda_environment": "geoapps",
}

required_parameters = []

validations = {
    "title": {
        "types": [str],
    },
    "geoh5": {
        "types": [str, Workspace],
    },
    "objects": {
        "types": [str, UUID],
        "uuid": [],
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
    "horizontal_padding": {
        "types": [int, float],
    },
    "vertical_padding": {
        "types": [int, float],
    },
    "depth_core": {
        "types": [int, float],
    },
    "refinement_object": {
        "types": [str, UUID],
        "uuid": [],
    },
    "refinement_levels": {
        "types": [int, float],
    },
    "refinement_type": {
        "types": [str],
        "values": ["surface", "radial"],
    },
    "refinement_distance": {
        "types": [int, float],
    },
    "ga_group_name": {
        "types": [str],
    },
    "monitoring_directory": {
        "types": [str],
    },
    "workspace_geoh5": {
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
