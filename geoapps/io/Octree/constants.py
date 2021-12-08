#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from uuid import UUID

from geoh5py.workspace import Workspace

required_parameters = []

defaults = {
    "title": "Octree Mesh Creator",
    "geoh5": None,
    "objects": None,
    "u_cell_size": 25,
    "v_cell_size": 25,
    "w_cell_size": 25,
    "horizontal_padding": 1000.0,
    "vertical_padding": 1000.0,
    "depth_core": 500.0,
    "ga_group_name": "Octree_Mesh",
    "Template A Object": None,
    "Template A Levels": [4, 4, 4],
    "Template A Type": "radial",
    "Template A Distance": 1000.0,
    "run_command": ("geoapps.create.octree_mesh"),
    "run_command_boolean": False,
    "monitoring_directory": None,
    "workspace_geoh5": None,
    "conda_environment": "geoapps",
    "conda_environment_boolean": False,
}

default_ui_json = {
    "title": "Octree Mesh Creator",
    "geoh5": None,
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
        "value": None,
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
        "group": None,
        "label": "Name:",
        "value": "Octree_Mesh",
    },
    "Template Object": {
        "enabled": True,
        "group": "Refinement A",
        "label": "Object",
        "meshType": [
            "{202C5DB1-A56D-4004-9CAD-BAAFD8899406}",
            "{6A057FDC-B355-11E3-95BE-FD84A7FFCB88}",
            "{F26FEBA3-ADED-494B-B9E9-B2BBCBE298E1}",
        ],
        "value": None,
    },
    "Template Levels": {
        "enabled": True,
        "group": "Refinement A",
        "label": "Levels",
        "value": "4,4,4",
    },
    "Template Type": {
        "choiceList": ["surface", "radial"],
        "enabled": True,
        "group": "Refinement A",
        "label": "Type",
        "value": "radial",
    },
    "Template Distance": {
        "enabled": True,
        "group": "Refinement A",
        "label": "Distance",
        "value": 1000.0,
    },
    "run_command": ("geoapps.create.octree_mesh"),
    "run_command_boolean": {
        "value": False,
        "label": "Run python module ",
        "tooltip": "Warning: launches process to run python model on save",
        "main": True,
    },
    "monitoring_directory": None,
    "conda_environment": "geoapps",
    "conda_environment_boolean": False,
}

required_parameters = ["objects"]

validations = {
    "title": {
        "types": [str],
    },
    "workspace_geoh5": {
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
    "template_object": {
        "types": [str, UUID],
        "uuid": [],
    },
    "template_levels": {
        "types": [int, float],
    },
    "template_type": {
        "types": [str],
        "values": ["surface", "radial"],
    },
    "template_distance": {
        "types": [int, float],
    },
    "ga_group_name": {
        "types": [str],
    },
    "run_command": {
        "types": [str],
    },
    "run_command_boolean": {
        "types": [bool],
    },
    "monitoring_directory": {
        "types": [str],
    },
    "conda_environment": {
        "types": [str],
    },
    "conda_environment_boolean": {
        "types": [bool],
    },
}

app_initializer = {
    "geoh5": "../../assets/FlinFlon.geoh5",
    "objects": UUID("{656acd40-25de-4865-814c-cb700f6ee51a}"),
    "Refinement A object": UUID("{656acd40-25de-4865-814c-cb700f6ee51a}"),
    "Refinement A levels": [4.0, 4.0, 4.0],
    "Refinement A type": "radial",
    "Refinement A distance": 1000,
    "Refinement B object": None,
    "Refinement B levels": [0.0, 0.0, 2.0],
    "Refinement B type": "surface",
    "Refinement B distance": 1000,
}
