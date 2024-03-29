#  Copyright (c) 2024 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from copy import deepcopy
from uuid import UUID

from geoh5py.ui_json.constants import default_ui_json as base_ui_json

import geoapps
from geoapps import assets_path

defaults = {
    "version": geoapps.__version__,
    "title": "octree Mesh Creator",
    "geoh5": None,
    "objects": None,
    "u_cell_size": 25.0,
    "v_cell_size": 25.0,
    "w_cell_size": 25.0,
    "horizontal_padding": 1000.0,
    "vertical_padding": 1000.0,
    "depth_core": 500.0,
    "ga_group_name": "Octree_Mesh",
    "generate_sweep": False,
    "run_command": "geoapps.octree_creation.driver",
    "monitoring_directory": None,
    "workspace_geoh5": None,
    "conda_environment": "geoapps",
    "conda_environment_boolean": False,
}

default_ui_json = deepcopy(base_ui_json)
default_ui_json.update(
    {
        "version": geoapps.__version__,
        "title": "octree Mesh Creator",
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
                "{b99bd6e5-4fe1-45a5-bd2f-75fc31f91b38}",
                "{0b639533-f35b-44d8-92a8-f70ecff3fd26}",
                "{9b08bb5a-300c-48fe-9007-d206f971ea92}",
                "{19730589-fd28-4649-9de0-ad47249d9aba}",
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
        "diagonal_balance": {
            "group": "Basic",
            "label": "UBC Compatible",
            "main": True,
            "value": True,
        },
        "minimum_level": {
            "enabled": True,
            "group": "Basic",
            "label": "Minimum refinement level.",
            "main": True,
            "min": 1,
            "tooltip": "Minimum refinement in padding region: 2**(n-1) x base_cell.",
            "value": 4,
        },
        "ga_group_name": {
            "enabled": True,
            "group": None,
            "label": "Name:",
            "value": "Octree_Mesh",
        },
        "generate_sweep": {
            "label": "Generate sweep file",
            "group": "Python run preferences",
            "main": True,
            "value": False,
        },
        "conda_environment": "geoapps",
        "workspace_geoh5": None,
        "run_command": "geoapps.octree_creation.driver",
    }
)

template_dict = {
    "object": {
        "groupOptional": True,
        "enabled": False,
        "group": "Refinement A",
        "label": "Object",
        "meshType": [
            "{202C5DB1-A56D-4004-9CAD-BAAFD8899406}",
            "{6A057FDC-B355-11E3-95BE-FD84A7FFCB88}",
            "{F26FEBA3-ADED-494B-B9E9-B2BBCBE298E1}",
            "{b99bd6e5-4fe1-45a5-bd2f-75fc31f91b38}",
            "{0b639533-f35b-44d8-92a8-f70ecff3fd26}",
            "{9b08bb5a-300c-48fe-9007-d206f971ea92}",
            "{19730589-fd28-4649-9de0-ad47249d9aba}",
        ],
        "value": None,
    },
    "levels": {
        "enabled": False,
        "group": "Refinement A",
        "label": "Levels",
        "value": "4, 4, 4",
    },
    "type": {
        "choiceList": ["surface", "radial"],
        "enabled": False,
        "group": "Refinement A",
        "label": "Type",
        "value": "radial",
    },
    "distance": {
        "enabled": False,
        "group": "Refinement A",
        "label": "Distance",
        "value": 1000.0,
    },
}

validations = {}

app_initializer = {
    "geoh5": str(assets_path() / "FlinFlon.geoh5"),
    "objects": UUID("{656acd40-25de-4865-814c-cb700f6ee51a}"),
    "Refinement A object": UUID("{656acd40-25de-4865-814c-cb700f6ee51a}"),
    "Refinement A levels": "4, 4, 4",
    "Refinement A type": "radial",
    "Refinement A distance": 1000.0,
    "Refinement B object": UUID("{ab3c2083-6ea8-4d31-9230-7aad3ec09525}"),
    "Refinement B levels": "0, 0, 4",
    "Refinement B type": "surface",
    "Refinement B distance": 1200.0,
}
