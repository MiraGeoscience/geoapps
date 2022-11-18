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
    "generate_sweep": False,
    "run_command": "geoapps.interpolation.driver",
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
        "monitoring_directory": "",
        "conda_environment": "geoapps",
        "conda_environment_boolean": False,
        "workspace": "",
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
                "{B020A277-90E2-4CD7-84D6-612EE3F25051}",
                "{7CAEBF0E-D16E-11E3-BC69-E4632694AA37}",
            ],
            "main": True,
            "group": "Destination",
            "label": "Object",
            "value": None,
        },
        "skew_angle": {
            "main": False,
            "group": "Inverse Distance",
            "groupOptional": True,
            "enabled": False,
            "value": 0.0,
            "label": "Azimuth (d.dd)",
            "tooltip": "Overrides default nearest neighbor method.",
        },
        "skew_factor": {
            "main": False,
            "group": "Inverse Distance",
            "enabled": False,
            "min": 1e-14,
            "value": 1.0,
            "label": "Factor",
            "tooltip": "Overrides default nearest neighbor method.",
        },
        "max_distance": {
            "main": False,
            "group": "Horizontal Extent",
            "optional": True,
            "enabled": True,
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
            "enabled": True,
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
            "group": "Lower Limit (m)",
            "groupOptional": True,
            "enabled": True,
            "value": 0.0,
            "label": "Value",
        },
        "space": {
            "main": False,
            "group": "Scaling",
            "value": "Linear",
            "choiceList": ["Linear", "Log"],
            "label": "Value",
        },
        "no_data_value": {
            "main": False,
            "group": "No-data-value",
            "value": 1e-08,
            "label": "Value",
        },
        "ga_group_name": {
            "main": True,
            "label": "Output Label",
            "value": "_Interp",
            "group": "Python run preferences",
        },
        "generate_sweep": {
            "label": "Generate sweep file",
            "group": "Python run preferences",
            "main": True,
            "value": False,
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
    "no_data_value": 1e-8,
    "out_object": "{7450be38-1327-4336-a9e4-5cff587b6715}",
    "skew_angle": 0.0,
    "skew_factor": 1.0,
    "space": "Linear",
    "topography_objects": "{ab3c2083-6ea8-4d31-9230-7aad3ec09525}",
    "topography_data": "{a603a762-f6cb-4b21-afda-3160e725bf7d}",
}
