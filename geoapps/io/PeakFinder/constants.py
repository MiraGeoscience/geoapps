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
        "value": "{b834a590-dea9-48cb-abe3-8c714bb0bb7c}",
    },
    "group_auto": {
        "main": True,
        "label": "Auto-group",
        "value": True,
    },
    "line_field": {
        "association": "Vertex",
        "dataType": "Float",
        "group": "Data",
        "main": True,
        "label": "Line Field",
        "parent": "objects",
        "value": "{ea658d13-9c6f-4ddc-8b53-68a3d1bf2e5c}",
    },
    "line_id": 6073400.0,
    "Property Group A Data": {
        "association": "Vertex",
        "dataType": "Float",
        "group": "Property Group A",
        "dataGroupType": "Multi-element",
        "label": "Property Group A:",
        "parent": "objects",
        "value": "",
    },
    "Property Group A Color": {
        "dataType": "Text",
        "group": "Property Group A",
        "label": "Property Group A Color:",
        "value": "",
    },
    "Property Group B Data": {
        "association": "Vertex",
        "dataType": "Float",
        "group": "Property Group B",
        "dataGroupType": "Multi-element",
        "label": "Property Group B:",
        "parent": "objects",
        "value": "",
    },
    "Property Group B Color": {
        "dataType": "Text",
        "group": "Property Group B",
        "label": "Property Group B Color:",
        "value": "",
    },
    "smoothing": {
        "group": "Detection Parameters",
        "label": "Smoothing window",
        "value": 6,
    },
    "min_amplitude": {
        "group": "Detection Parameters",
        "label": "Minimum Amplitude (%)",
        "value": 1,
    },
    "min_value": {
        "group": "Detection Parameters",
        "label": "Minimum Value",
        "value": "",
    },
    "min_width": {
        "group": "Detection Parameters",
        "label": "Minimum Width (m)",
        "value": 100,
    },
    "max_migration": {
        "group": "Detection Parameters",
        "label": "Maximum Peak Migration (m)",
        "value": 25,
    },
    "min_channels": {
        "group": "Detection Parameters",
        "label": "Minimum # Channels",
        "value": 1,
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
        "types": [str, UUID],
        "reqs": [("objects")],
        "property_groups": ["objects"],
    },
    "group_auto": {
        "types": [bool],
    },
    "property_group_data": {
        "types": [str, UUID],
        "reqs": [("objects")],
        "property_groups": ["objects"],
    },
    "property_group_color": {
        "types": [str],
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
        "types": [int, float],
    },
    "smoothing": {
        "types": [int, float],
    },
    "min_amplitude": {
        "types": [int, float],
    },
    "min_value": {
        "types": [int, float],
    },
    "min_width": {
        "types": [int, float],
    },
    "max_migration": {
        "types": [int, float],
    },
    "min_channels": {
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
