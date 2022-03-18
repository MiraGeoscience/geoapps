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
from copy import deepcopy
from uuid import UUID

import numpy as np
from geoh5py.ui_json.constants import default_ui_json as base_ui_json
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
    "min_width": 100.0,
    "max_migration": 25.0,
    "min_channels": 1,
    "ga_group_name": "PeakFinder",
    "structural_markers": False,
    "line_id": None,
    "group_auto": True,
    "center": None,
    "width": None,
}

default_ui_json = deepcopy(base_ui_json)
default_ui_json.update(
    {
        "title": "Peak Finder Parameters",
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
            "value": 0.0,
            "main": True,
        },
        "min_width": {
            "group": "Detection Parameters",
            "label": "Minimum Width (m)",
            "value": 100.0,
            "main": True,
        },
        "max_migration": {
            "group": "Detection Parameters",
            "label": "Maximum Peak Migration (m)",
            "value": 25.0,
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
    }
)

template_dict = {
    "data": {
        "association": "Vertex",
        "group": "Group A",
        "dataGroupType": "Multi-element",
        "label": "Property Group",
        "parent": "objects",
        "dependency": "group_auto",
        "dependencyType": "disabled",
        "value": None,
    },
    "color": {
        "dataType": "Text",
        "group": "Group A",
        "label": "Color",
        "dependency": "group_auto",
        "dependencyType": "disabled",
        "value": None,
    },
}

# Over-write validations for jupyter app parameters
validations = {
    "line_id": {"types": [float, type(None)]},
    "center": {"types": [float, type(None)]},
    "width": {"types": [float, type(None)]},
}

app_initializer = {
    "geoh5": "../../assets/FlinFlon.geoh5",
    "objects": UUID("{bb208abb-dc1f-4820-9ea9-b8883e5ff2c6}"),
    "data": UUID("{b834a590-dea9-48cb-abe3-8c714bb0bb7c}"),
    "line_field": UUID("{ea658d13-9c6f-4ddc-8b53-68a3d1bf2e5c}"),
    "system": "VTEM (2007)",
    "line_id": 6073400.0,
    "center": 4041.2,
    "width": 1000.0,
    "tem_checkbox": True,
    "min_value": -0.0004509940918069333,
    "group_auto": True,
}
