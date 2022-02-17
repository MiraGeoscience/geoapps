#  Copyright (c) 2022 Mira Geoscience Ltd.
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

defaults = {
    "title": "Peak Finder Parameters",
    "geoh5": None,
    "tem_checkbox": False,
    "objects": None,
    "data": None,
    "flip_sign": False,
    "line_field": None,
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
    "run_command": ("geoapps.processing.peak_finder"),
    "run_command_boolean": None,
    "conda_environment": "geoapps",
    "conda_environment_boolean": None,
    "template_data": None,
    "template_color": None,
    "monitoring_directory": None,
}

default_ui_json = {
    "title": "Peak Finder Parameters",
    "geoh5": None,
    "tem_checkbox": {
        "main": True,
        "label": "TEM type",
        "value": False,
    },
    "objects": {
        "main": True,
        "group": "Data",
        "label": "Object",
        "meshType": [
            "{6A057FDC-B355-11E3-95BE-FD84A7FFCB88}",
        ],
        "value": None,
    },
    "data": {
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
        "main": True,
        "group": "Data",
        "label": "Flip sign",
        "value": False,
    },
    "line_field": {
        "association": "Vertex",
        "dataType": "Float",
        "group": "Data",
        "main": True,
        "label": "Line Field",
        "parent": "objects",
        "value": None,
    },
    "system": {
        "choiceList": list(parameters().keys()),
        "main": True,
        "label": "TEM system",
        "dependency": "tem_checkbox",
        "dependencyType": "enabled",
        "value": None,
    },
    "smoothing": {
        "group": "Detection Parameters",
        "label": "Smoothing window",
        "main": True,
        "value": 6,
    },
    "min_amplitude": {
        "group": "Detection Parameters",
        "label": "Minimum Amplitude (%)",
        "value": 1,
        "main": True,
    },
    "min_value": {
        "group": "Detection Parameters",
        "label": "Minimum Value",
        "value": None,
        "main": True,
    },
    "min_width": {
        "group": "Detection Parameters",
        "label": "Minimum Width (m)",
        "value": 100,
        "main": True,
    },
    "max_migration": {
        "group": "Detection Parameters",
        "label": "Maximum Peak Migration (m)",
        "value": 25,
        "main": True,
    },
    "min_channels": {
        "group": "Detection Parameters",
        "label": "Minimum # Channels",
        "value": 1,
        "main": True,
    },
    "ga_group_name": {
        "enabled": True,
        "main": True,
        "group": "Python run preferences",
        "label": "Save As",
        "value": "PeakFinder",
    },
    "structural_markers": {
        "main": True,
        "group": "Python run preferences",
        "label": "Export all markers",
        "value": False,
    },
    "line_id": None,
    "group_auto": {
        "label": "Auto-group",
        "value": True,
    },
    "center": None,
    "width": None,
    "run_command": ("geoapps.processing.peak_finder"),
    "run_command_boolean": None,
    "conda_environment": "geoapps",
    "conda_environment_boolean": None,
    "template_data": None,
    "template_color": None,
    "monitoring_directory": None,
    "plot_result": True,
}

free_format_dict = {
    "Template Data": {
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
    "Template Color": {
        "dataType": "Text",
        "group": "Property Group",
        "label": "Color",
        "dependency": "group_auto",
        "dependencyType": "disabled",
        "value": None,
    },
}

required_parameters = ["objects", "data"]

validations = {
    "title": {
        "types": [str],
    },
    "geoh5": {
        "types": [str, Workspace],
    },
    "tem_checkbox": {
        "types": [bool],
    },
    "objects": {
        "types": [str, UUID],
        "uuid": None,
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
    "template_data": {
        "types": [str, UUID],
        "reqs": [("objects")],
        "property_groups": ["objects"],
    },
    "template_color": {
        "types": [str],
    },
    "monitoring_directory": {
        "types": [str],
    },
    "plot_result": {
        "types": [bool],
    },
    "workspace_geoh5": {
        "types": [str],
    },
}

app_initializer = {
    "geoh5": "../../assets/FlinFlon.geoh5",
    "objects": UUID("{bb208abb-dc1f-4820-9ea9-b8883e5ff2c6}"),
    "data": UUID("{b834a590-dea9-48cb-abe3-8c714bb0bb7c}"),
    "line_field": UUID("{ea658d13-9c6f-4ddc-8b53-68a3d1bf2e5c}"),
    "system": "VTEM (2007)",
    "line_id": 6073400.0,
    "center": 4041.2,
    "width": 1000,
    "tem_checkbox": True,
    "min_value": -0.0004509940918069333,
    "group_auto": True,
}
