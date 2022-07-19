#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from copy import deepcopy

from geoh5py.ui_json.constants import default_ui_json as base_ui_json

defaults = {
    "title": "Data Transfer",
    "geoh5": None,
    "objects": None,
    "data": None,
    "method": None,
    "skew_angle": None,
    "skew_factor": None,
    "space": None,
    "max_distance": None,
    "xy_extent": None,
    "topography_objects": None,
    "topography_data": None,
    "max_depth": None,
    "no_data_value": None,
    "out_object": None,
    "ga_group_name": None,
    "run_command": "geoapps.interpolation.driver",
    "run_command_boolean": False,
    "workspace_geoh5": None,
    "conda_environment": "geoapps",
    "conda_environment_boolean": False,
}

default_ui_json = deepcopy(base_ui_json)
default_ui_json.update(
    {
        "title": "Data Transfer",
        "geoh5": "",
        "run_command": "geoapps.interpolation.driver",
        "run_command_boolean": {
            "value": False,
            "label": "Run python module ",
            "tooltip": "Warning: launches process to run python model on save",
            "main": True,
        },
        "monitoring_directory": "",
        "conda_environment": "geoapps",
        "conda_environment_boolean": False,
        "objects": {
            "meshType": [
                "{2e814779-c35f-4da0-ad6a-39a6912361f9}",
                "{202C5DB1-A56D-4004-9CAD-BAAFD8899406}",
                "{6A057FDC-B355-11E3-95BE-FD84A7FFCB88}",
                "{F26FEBA3-ADED-494B-B9E9-B2BBCBE298E1}",
                "{4EA87376-3ECE-438B-BF12-3479733DED46}",
                "{48f5054a-1c5c-4ca4-9048-80f36dc60a06}",
                "{B020A277-90E2-4CD7-84D6-612EE3F25051}",
                "{7CAEBF0E-D16E-11E3-BC69-E4632694AA37}",
            ],
            "main": True,
            "group": "Data Selection",
            "label": "Object",
            "value": None,
        },
        "data": {
            "main": True,
            "group": "Data Selection",
            "association": ["Vertex", "Cell"],
            "dataType": "Float",
            "label": "Data",
            "parent": "objects",
            "value": None,
        },
        "out_object": {
            "meshType": [
                "{2e814779-c35f-4da0-ad6a-39a6912361f9}",
                "{202C5DB1-A56D-4004-9CAD-BAAFD8899406}",
                "{6A057FDC-B355-11E3-95BE-FD84A7FFCB88}",
                "{F26FEBA3-ADED-494B-B9E9-B2BBCBE298E1}",
                "{4EA87376-3ECE-438B-BF12-3479733DED46}",
                "{48f5054a-1c5c-4ca4-9048-80f36dc60a06}",
            ],
            "main": True,
            "group": "Destination",
            "label": "Object",
            "value": None,
        },
        "space": {
            "main": False,
            "value": "Linear",
            "choiceList": ["Linear", "Log"],
            "label": "Scaling",
        },
        "no_data_value": {
            "main": False,
            "value": 0.0,
            "label": "No-data-value",
        },
        "method": {
            "main": False,
            "group": "Method",
            "choiceList": ["Nearest", "Inverse Distance"],
            "value": "Inverse Distance",
            "label": "Method",
        },
        "skew_angle": {
            "main": False,
            "group": "Method",
            "optional": True,
            "enabled": False,
            "value": 0.0,
            "label": "Azimuth (d.dd)",
        },
        "skew_factor": {
            "main": False,
            "group": "Method",
            "optional": True,
            "enabled": False,
            "value": 1.0,
            "label": "Factor (>0)",
        },
        "max_distance": {
            "main": False,
            "group": "Horizontal Extent",
            "optional": True,
            "enabled": False,
            "value": 0.0,
            "label": "Maximum distance (m)",
        },
        "xy_extent": {
            "meshType": [
                "{2e814779-c35f-4da0-ad6a-39a6912361f9}",
                "{202C5DB1-A56D-4004-9CAD-BAAFD8899406}",
                "{6A057FDC-B355-11E3-95BE-FD84A7FFCB88}",
                "{F26FEBA3-ADED-494B-B9E9-B2BBCBE298E1}",
                "{4EA87376-3ECE-438B-BF12-3479733DED46}",
                "{48f5054a-1c5c-4ca4-9048-80f36dc60a06}",
                "{B020A277-90E2-4CD7-84D6-612EE3F25051}",
                "{7CAEBF0E-D16E-11E3-BC69-E4632694AA37}",
            ],
            "main": False,
            "optional": True,
            "enabled": False,
            "group": "Horizontal Extent",
            "label": "Object hull",
            "value": None,
        },
        "topography_objects": {
            "meshType": [
                "{2e814779-c35f-4da0-ad6a-39a6912361f9}",
                "{202C5DB1-A56D-4004-9CAD-BAAFD8899406}",
                "{6A057FDC-B355-11E3-95BE-FD84A7FFCB88}",
                "{F26FEBA3-ADED-494B-B9E9-B2BBCBE298E1}",
                "{4EA87376-3ECE-438B-BF12-3479733DED46}",
                "{48f5054a-1c5c-4ca4-9048-80f36dc60a06}",
            ],
            "main": False,
            "group": "Upper Limit",
            "optional": True,
            "enabled": False,
            "label": "Object",
            "value": None,
        },
        "topography_data": {
            "main": False,
            "group": "Upper Limit",
            "optional": True,
            "enabled": False,
            "association": ["Vertex", "Cell"],
            "dataType": "Float",
            "label": "Data",
            "parent": "topography_objects",
            "dependency": "topography_objects",
            "value": None,
        },
        "max_depth": {
            "main": False,
            "optional": True,
            "enabled": False,
            "value": 0.0,
            "label": "Lower Limit (m)",
        },
        "ga_group_name": {
            "main": True,
            "label": "Output Label",
            "value": "_Interp",
            "group": "Python run preferences",
        },
    }
)

validations = {}

app_initializer = {
    "geoh5": "../../assets/FlinFlon.geoh5",
    "objects": "{2e814779-c35f-4da0-ad6a-39a6912361f9}",
    "data": "{f3e36334-be0a-4210-b13e-06933279de25}",
    "max_distance": 2e3,
    "max_depth": 1e3,
    "method": "Inverse Distance",
    "no_data_value": 1e-8,
    "out_object": "{7450be38-1327-4336-a9e4-5cff587b6715}",
    "skew_angle": 0.0,
    "skew_factor": 1.0,
    "space": "Linear",
    "topography_objects": "{ab3c2083-6ea8-4d31-9230-7aad3ec09525}",
}