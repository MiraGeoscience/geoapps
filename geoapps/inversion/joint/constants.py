# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024 Mira Geoscience Ltd.                                     '
#                                                                              '
#  This file is part of geoapps.                                               '
#                                                                              '
#  geoapps is distributed under the terms and conditions of the MIT License    '
#  (see LICENSE file at the root of this source code package).                 '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from __future__ import annotations

from geoapps.inversion import default_ui_json as base_default_ui_json
from geoapps.inversion.constants import validations as base_validations

default_ui_json = {
    "title": "SimPEG Joint Surveys Inversion",
    "inversion_type": "joint surveys",
    "mesh": {
        "group": "Mesh and Models",
        "main": True,
        "label": "Mesh",
        "meshType": "4EA87376-3ECE-438B-BF12-3479733DED46",
        "value": None,
        "enabled": False,
        "optional": True,
    },
    "group_a": {
        "main": True,
        "group": "Joint",
        "label": "Group A",
        "groupType": "{55ed3daf-c192-4d4b-a439-60fa987fe2b8}",
        "value": "",
    },
    "group_a_multiplier": {
        "min": 0.0,
        "main": True,
        "group": "Joint",
        "label": "Misfit A Scale",
        "value": 1.0,
        "tooltip": "Constant multiplier for the data misfit function for Group A.",
    },
    "group_b": {
        "main": True,
        "group": "Joint",
        "label": "Group B",
        "groupType": "{55ed3daf-c192-4d4b-a439-60fa987fe2b8}",
        "value": "",
    },
    "group_b_multiplier": {
        "min": 0.0,
        "main": True,
        "group": "Joint",
        "label": "Misfit B Scale",
        "value": 1.0,
        "tooltip": "Constant multiplier for the data misfit function for Group B.",
    },
    "group_c": {
        "main": True,
        "group": "Joint",
        "label": "Group C",
        "groupType": "{55ed3daf-c192-4d4b-a439-60fa987fe2b8}",
        "optional": True,
        "enabled": False,
        "value": "",
    },
    "group_c_multiplier": {
        "min": 0.0,
        "main": True,
        "group": "Joint",
        "label": "Misfit C Scale",
        "value": 1.0,
        "dependency": "group_c",
        "dependencyType": "enabled",
        "tooltip": "Constant multiplier for the data misfit function for Group C.",
    },
}
default_ui_json = dict(base_default_ui_json, **default_ui_json)
validations = {
    "inversion_type": {
        "required": True,
        "values": ["joint surveys"],
    },
}

validations = dict(base_validations, **validations)
app_initializer = {}
