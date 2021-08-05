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

from ...utils.geophysical_systems import parameters

required_parameters = []

defaults = {
    "title": "Peak Finder Parameters",
    "geoh5": None,
    "objects": None,
    "data": None,
    "flip_sign": False,
    "line_field": None,
    "tem_checkbox": True,
    "system": None,
    "smoothing": 6,
    "min_amplitude": 1,
    "min_value": None,
    "min_width": 100,
    "max_migration": 25,
    "min_channels": 1,
    "ga_group_name": "PeakFinder",
    "structural_markers": False,
    "line_id": None,
    "group_auto": True,
    "center": None,
    "width": None,
    "Property Group Data": None,
    "Property Group Color": None,
    "run_command": ("geoapps.processing.peak_finder"),
    "run_command_boolean": None,
    "conda_environment": "geoapps",
    "conda_environment_boolean": None,
    "property_group_data": None,
    "property_group_color": None,
    "workspace_geoh5": None,
    "workspace": None,
    "monitoring_directory": None,
}

default_ui_json = {
    "title": "Peak Finder Parameters",
    "geoh5": None,
    "objects": {
        "default": None,
        "main": True,
        "group": "Data",
        "label": "Object",
        "meshType": [
            "{6A057FDC-B355-11E3-95BE-FD84A7FFCB88}",
        ],
        "value": None,
    },
    "data": {
        "default": None,
        "association": "Vertex",
        "dataType": "Float",
        "group": "Data",
        "main": True,
        "dataGroupType": "Multi-element",
        "label": "Channels",
        "parent": "objects",
        "value": None,
    },
    "flip_sign": {
        "default": False,
        "main": True,
        "group": "Data",
        "label": "Flip sign",
        "value": False,
    },
    "line_field": {
        "default": None,
        "association": "Vertex",
        "dataType": "Float",
        "group": "Data",
        "main": True,
        "label": "Line Field",
        "parent": "objects",
        "value": None,
    },
    "tem_checkbox": {
        "default": True,
        "main": True,
        "label": "TEM type",
        "value": True,
    },
    "system": {
        "default": None,
        "choiceList": list(parameters().keys()),
        "main": True,
        "label": "TEM system",
        "dependency": "tem_checkbox",
        "dependencyType": "enabled",
        "value": None,
    },
    "smoothing": {
        "default": 6,
        "group": "Detection Parameters",
        "label": "Smoothing window",
        "main": True,
        "value": 6,
    },
    "min_amplitude": {
        "default": 1,
        "group": "Detection Parameters",
        "label": "Minimum Amplitude (%)",
        "value": 1,
        "main": True,
    },
    "min_value": {
        "default": None,
        "group": "Detection Parameters",
        "label": "Minimum Value",
        "value": None,
        "main": True,
    },
    "min_width": {
        "default": 100,
        "group": "Detection Parameters",
        "label": "Minimum Width (m)",
        "value": 100,
        "main": True,
    },
    "max_migration": {
        "default": 25,
        "group": "Detection Parameters",
        "label": "Maximum Peak Migration (m)",
        "value": 25,
        "main": True,
    },
    "min_channels": {
        "default": 1,
        "group": "Detection Parameters",
        "label": "Minimum # Channels",
        "value": 1,
        "main": True,
    },
    "ga_group_name": {
        "default": "PeakFinder",
        "visible": True,
        "enabled": True,
        "main": True,
        "group": "Python run preferences",
        "label": "Save As",
        "value": "PeakFinder",
    },
    "structural_markers": {
        "default": False,
        "main": True,
        "group": "Python run preferences",
        "label": "Export all markers",
        "value": False,
    },
    "line_id": None,
    "group_auto": {
        "default": True,
        "label": "Auto-group",
        "value": True,
    },
    "center": {
        "default": None,
        "group": "Window",
        "label": "Window center",
        "value": None,
        "visible": False,
    },
    "width": {
        "default": None,
        "group": "Window",
        "label": "Window width",
        "value": None,
        "visible": False,
    },
    "Property Group Data": {
        "association": "Vertex",
        "dataType": "Float",
        "group": "Property Group",
        "dataGroupType": "Multi-element",
        "label": "Property Group",
        "parent": "objects",
        "dependency": "group_auto",
        "dependencyType": "disabled",
        "value": None,
    },
    "Property Group Color": {
        "dataType": "Text",
        "group": "Property Group",
        "label": "Color",
        "dependency": "group_auto",
        "dependencyType": "disabled",
        "value": None,
    },
    "run_command": ("geoapps.processing.peak_finder"),
    "run_command_boolean": None,
    "conda_environment": "geoapps",
    "conda_environment_boolean": None,
    "property_group_data": None,
    "property_group_color": None,
    "workspace_geoh5": None,
    "workspace": None,
    "monitoring_directory": None,
}

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
    "flip_sign": {
        "types": [bool],
    },
    "line_field": {
        "types": [str, UUID, int, float],
        "reqs": [("objects")],
        "uuid": ["objects"],
    },
    "tem_checkbox": {
        "types": [bool],
    },
    "system": {
        "types": [str],
        "values": list(parameters().keys()) + [None],
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
    "ga_group_name": {
        "types": [str],
    },
    "structural_markers": {
        "types": [bool],
    },
    "line_id": {
        "types": [int, float, str],
    },
    "group_auto": {
        "types": [bool],
    },
    "center": {
        "types": [int, float],
    },
    "width": {
        "types": [int, float],
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
    "property_group_data": {
        "types": [str, UUID],
        "reqs": [("objects")],
        "property_groups": ["objects"],
    },
    "property_group_color": {
        "types": [str],
    },
    "workspace_geoh5": {
        "types": [str, Workspace],
    },
    "workspace": {
        "types": [str, Workspace],
    },
    "monitoring_directory": {
        "types": [str],
    },
}

app_initializer = {
    "geoh5": "../../assets/FlinFlon.geoh5",
    "objects": "{bb208abb-dc1f-4820-9ea9-b8883e5ff2c6}",
    "data": "{b834a590-dea9-48cb-abe3-8c714bb0bb7c}",
    "line_field": "{ea658d13-9c6f-4ddc-8b53-68a3d1bf2e5c}",
    "line_id": 6073400.0,
    "center": 4050,
    "width": 1000,
}
