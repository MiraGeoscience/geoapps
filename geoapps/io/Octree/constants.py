#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from uuid import UUID

from geoh5py.objects import Curve
from geoh5py.workspace import Workspace

defaults = {
    "title": "Octree Mesh Creator",
    "geoh5": None,
    "objects": None,
    "u_cell_size": 25.0,
    "v_cell_size": 25.0,
    "w_cell_size": 25.0,
    "horizontal_padding": 1000.0,
    "vertical_padding": 1000.0,
    "depth_core": 500.0,
    "ga_group_name": "Octree_Mesh",
    "run_command": ("geoapps.create.octree_mesh"),
    "run_command_boolean": False,
    "monitoring_directory": None,
    "workspace_geoh5": None,
    "conda_environment": "geoapps",
    "conda_environment_boolean": False,
}
validations = {}
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
        "value": 25.0,
    },
    "v_cell_size": {
        "enabled": True,
        "group": "2- Core cell size",
        "label": "Northing (m)",
        "main": True,
        "value": 25.0,
    },
    "w_cell_size": {
        "enabled": True,
        "group": "2- Core cell size",
        "label": "Vertical (m)",
        "main": True,
        "value": 25.0,
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

template_dict = {
    "object": {
        "groupOptional": True,
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
    "levels": {
        "enabled": True,
        "group": "Refinement A",
        "label": "Levels",
        "value": "4, 4, 4",
    },
    "type": {
        "choiceList": ["surface", "radial"],
        "enabled": True,
        "group": "Refinement A",
        "label": "Type",
        "value": "radial",
    },
    "distance": {
        "enabled": True,
        "group": "Refinement A",
        "label": "Distance",
        "value": 1000.0,
    },
}

app_initializer = {
    "geoh5": "../../assets/FlinFlon.geoh5",
    "objects": UUID("{656acd40-25de-4865-814c-cb700f6ee51a}"),
    "Refinement A object": UUID("{656acd40-25de-4865-814c-cb700f6ee51a}"),
    "Refinement A levels": [4, 4, 4],
    "Refinement A type": "radial",
    "Refinement A distance": 1000.0,
    "Refinement B object": UUID("{ab3c2083-6ea8-4d31-9230-7aad3ec09525}"),
    "Refinement B levels": [0, 0, 4],
    "Refinement B type": "surface",
    "Refinement B distance": 1200.0,
}
