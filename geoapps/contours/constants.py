#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from copy import deepcopy

from geoh5py.ui_json.constants import default_ui_json as base_ui_json

defaults = {
    "title": "Create Contours",
    "geoh5": None,
    "objects": None,
    "data": None,
    "interval_min": 0.0,
    "interval_max": 0.0,
    "interval_spacing": 0.0,
    "fixed_contours": "",
    "resolution": None,
    "ga_group_name": None,
    "generate_sweep": False,
    "window_azimuth": None,
    "window_center_x": None,
    "window_center_y": None,
    "window_width": None,
    "window_height": None,
    "run_command": "geoapps.contours.driver",
    "workspace_geoh5": None,
    "conda_environment": "geoapps",
    "conda_environment_boolean": False,
}

default_ui_json = deepcopy(base_ui_json)
default_ui_json.update(
    {
        "title": "Create Contours",
        "geoh5": "",
        "run_command": "geoapps.contours.driver",
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
            "label": "Value fields",
            "parent": "objects",
            "value": None,
        },
        "interval_min": {
            "main": True,
            "group": "Contours",
            "optional": True,
            "enabled": False,
            "label": "Interval min",
            "value": 0.0,
        },
        "interval_max": {
            "main": True,
            "group": "Contours",
            "dependency": "interval_min",
            "dependencyType": "enabled",
            "label": "Interval max",
            "value": 0.0,
        },
        "interval_spacing": {
            "main": True,
            "group": "Contours",
            "dependency": "interval_min",
            "dependencyType": "enabled",
            "label": "Interval spacing",
            "value": 0.0,
        },
        "fixed_contours": {
            "main": True,
            "group": "Contours",
            "label": "Fixed Contours",
            "value": "0",
            "optional": True,
            "enabled": False,
        },
        "window_azimuth": {
            "group": "Window",
            "groupOptional": True,
            "enabled": False,
            "main": True,
            "label": "Azimuth",
            "min": -90.0,
            "max": 90.0,
            "value": 0.0,
        },
        "window_center_x": {
            "group": "Window",
            "enabled": False,
            "main": True,
            "label": "Easting",
            "value": 0.0,
        },
        "window_center_y": {
            "group": "Window",
            "enabled": False,
            "main": True,
            "label": "Northing",
            "value": 0.0,
        },
        "window_width": {
            "group": "Window",
            "enabled": False,
            "main": True,
            "label": "Width",
            "min": 0.0,
            "value": 0.0,
        },
        "window_height": {
            "group": "Window",
            "enabled": False,
            "main": True,
            "label": "Height",
            "min": 0.0,
            "value": 0.0,
        },
        "export_as": {
            "main": True,
            "label": "Save as",
            "value": "contours",
            "group": "Python run preferences",
        },
        "z_value": {
            "main": True,
            "label": "Assign Z from values",
            "value": False,
            "group": "Python run preferences",
        },
        "ga_group_name": {
            "main": True,
            "label": "Group",
            "value": "Contours",
            "group": "Python run preferences",
        },
        "generate_sweep": {
            "label": "Generate sweep file",
            "group": "Python run preferences",
            "main": True,
            "value": False,
        },
        "resolution": 50.0,
        "plot_result": True,
    }
)

validations = {}

app_initializer = {
    "geoh5": "../../assets/FlinFlon.geoh5",
    "objects": "{538a7eb1-2218-4bec-98cc-0a759aa0ef4f}",
    "data": "{44822654-b6ae-45b0-8886-2d845f80f422}",
    "interval_min": -400.0,
    "interval_max": 2000.0,
    "interval_spacing": 100.0,
    "fixed_contours": "-240",
    "resolution": 50.0,
    "ga_group_name": "Contours",
    "window_azimuth": -20.0,
    "window_center_x": 315566.45,
    "window_center_y": 6070767.72,
    "window_width": 4401.30,
    "window_height": 6811.12,
}
