#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


from __future__ import annotations

from copy import deepcopy

from geoh5py.ui_json.constants import default_ui_json as base_ui_json

defaults = {
    "title": "Edge Detection",
    "geoh5": None,
    "objects": None,
    "data": None,
    "sigma": None,
    "window_azimuth": None,
    "export_as": "",
    "ga_group_name": None,
    "generate_sweep": False,
    "run_command": "geoapps.edge_detection.driver",
    "workspace_geoh5": None,
    "conda_environment": "geoapps",
    "conda_environment_boolean": False,
}

default_ui_json = deepcopy(base_ui_json)
default_ui_json.update(
    {
        "title": "Edge Detection",
        "geoh5": "",
        "run_command": "geoapps.edge_detection.driver",
        "monitoring_directory": "",
        "conda_environment": "geoapps",
        "conda_environment_boolean": False,
        "objects": {
            "group": "Data Selection",
            "meshType": ["{48f5054a-1c5c-4ca4-9048-80f36dc60a06}"],
            "main": True,
            "label": "Object",
            "value": None,
        },
        "data": {
            "group": "Data Selection",
            "main": True,
            "association": ["Cell"],
            "dataType": "Float",
            "label": "Data",
            "parent": "objects",
            "value": None,
        },
        "line_length": {
            "group": "Parameters",
            "main": True,
            "label": "Line Length",
            "min": 1,
            "max": 100,
            "value": 1,
        },
        "line_gap": {
            "group": "Parameters",
            "main": True,
            "label": "Line Gap",
            "min": 1,
            "max": 100,
            "value": 1,
        },
        "sigma": {
            "group": "Parameters",
            "main": True,
            "label": "Sigma",
            "value": 1.0,
            "min": 0.0,
            "precision": 1,
            "lineEdit": False,
            "max": 10.0,
        },
        "threshold": {
            "group": "Parameters",
            "main": True,
            "label": "Threshold",
            "min": 1,
            "max": 100,
            "value": 1,
        },
        "window_size": {
            "group": "Parameters",
            "main": True,
            "label": "Window Size",
            "min": 16,
            "max": 512,
            "value": 64,
        },
        "window_azimuth": {
            "group": "Figure Options",
            "main": True,
            "label": "Azimuth",
            "min": -90.0,
            "max": 90.0,
            "value": 0.0,
            "optional": True,
            "enabled": False,
        },
        "window_center_x": {
            "group": "Figure Options",
            "main": True,
            "label": "Easting",
            "value": 0.0,
            "optional": True,
            "enabled": False,
        },
        "window_center_y": {
            "group": "Figure Options",
            "main": True,
            "label": "Northing",
            "value": 0.0,
            "optional": True,
            "enabled": False,
        },
        "window_width": {
            "group": "Figure Options",
            "main": True,
            "label": "Width",
            "min": 0.0,
            "value": 100.0,
            "optional": True,
            "enabled": False,
        },
        "window_height": {
            "group": "Figure Options",
            "main": True,
            "label": "Height",
            "min": 0.0,
            "value": 100.0,
            "optional": True,
            "enabled": False,
        },
        "export_as": {
            "main": True,
            "label": "Save as",
            "value": "",
            "group": "Python run preferences",
        },
        "ga_group_name": {
            "main": True,
            "label": "Group",
            "value": "",
            "group": "Python run preferences",
        },
        "generate_sweep": {
            "label": "Generate sweep file",
            "group": "Python run preferences",
            "main": True,
            "value": False,
        },
        "resolution": 50.0,
        "colorbar": False,
        "zoom_extent": False,
        "plot_result": True,
    }
)

validations = {}

app_initializer = {
    "geoh5": "../../assets/FlinFlon.geoh5",
    "objects": "{538a7eb1-2218-4bec-98cc-0a759aa0ef4f}",
    "data": "{53e59b2b-c2ae-4b77-923b-23e06d874e62}",
    "sigma": 0.5,
    "window_azimuth": -20.0,
    "ga_group_name": "Edges",
}
