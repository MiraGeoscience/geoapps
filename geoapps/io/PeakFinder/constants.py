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

required_parameters = []
defaults = {}


default_ui_json = {
    "title": "Peak Finder Parameters",
    "geoh5": "../../assets/FlinFlon.geoh5",
    "objects": {
        "main": True,
        "group": "Data",
        "label": "Object",
        "meshType": [
            "{6A057FDC-B355-11E3-95BE-FD84A7FFCB88}",
        ],
        "value": "{bb208abb-dc1f-4820-9ea9-b8883e5ff2c6}",
    },
    "data": {
        "association": "Vertex",
        "dataType": "Float",
        "group": "Data",
        "main": True,
        "dataGroupType": "Multi-element",
        "label": "Channels",
        "parent": "objects",
        "value": "Observed",
    },
    "line_field": {
        "association": "Vertex",
        "dataType": "Float",
        "group": "Data",
        "main": True,
        "label": "Line Field",
        "parent": "objects",
        "value": "ea658d13-9c6f-4ddc-8b53-68a3d1bf2e5c",
    },
    "line_id": {
        "association": "Vertex",
        "dataType": "Text",
        "group": "Data",
        "main": True,
        "label": "Line ID",
        "value": "",
    },
    "smoothing": {
        "group": "Data",
        "main": True,
        "label": "Smoothing window",
        "value": 6,
    },
    "tem_checkbox": {
        "main": True,
        "label": "TEM type",
        "value": True,
    },
    "center": {
        "group": "Window",
        "label": "Window center",
        "value": 4050,
    },
    "width": {
        "group": "Window",
        "label": "Window width",
        "value": 1000,
    },
    "ga_group_name": {
        "visible": True,
        "enabled": True,
        "label": "Results group name",
        "value": "PeakFinder",
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
    "data": {
        "types": [str],
        "reqs": [("objects")],
        "property_groups": ["objects"],
    },
    "tem_checkbox": {
        "types": [bool],
    },
    "line_field": {
        "types": [str, UUID, int, float],
        "reqs": [("objects")],
        "uuid": ["objects"],
    },
    "line_id": {
        "types": [str, int, float],
    },
    "smoothing": {
        "types": [int, float],
    },
    "center": {
        "types": [int, float],
    },
    "width": {
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
