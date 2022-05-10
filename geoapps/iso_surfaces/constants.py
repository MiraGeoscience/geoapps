#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from copy import deepcopy

from geoh5py.ui_json.constants import default_ui_json as base_ui_json

defaults = {
    "title": "Create Iso Surfaces",
    "geoh5": None,
    "objects": None,
    "data": None,
    "contours": "0.005: 0.02: 0.005, 0.0025",
    "max_distance": 500.0,
    "resolution": 50.0,
    "export_as": "Iso_",
    "run_command": "geoapps.iso_surfaces.driver",
    "run_command_boolean": False,
    "monitoring_directory": None,
    "workspace_geoh5": None,
    "conda_environment": "geoapps",
    "conda_environment_boolean": False,
}

default_ui_json = deepcopy(base_ui_json)
default_ui_json.update(
    {
        "title": "Create Iso Surfaces",
        "geoh5": "",
        "run_command": "",
        "run_command_boolean": {
            "value": False,
            "label": "Run python module ",
            "tooltip": "Warning: launches process to run python model on save",
            "main": True
        },
        "monitoring_directory": "",
        "conda_environment": "geoapps",
        "conda_environment_boolean": False,
        "objects": {
            "meshType": ["{2e814779-c35f-4da0-ad6a-39a6912361f9}"],
            "main": True,
            "group": "Data Selection",
            "label": "Object",
            "value": None
        },
        "data": {
            "main": True,
            "group": "Data Selection",
            "association": ["Vertex", "Cell"],
            "dataType": "Float",
            "label": "Value fields",
            "parent": "objects",
            "value": None
        },
        "contours": {
            "main": True,
            "label": "Iso-values",
            "value": "0.005: 0.02: 0.005, 0.0025"
        },
        "max_distance": {
            "enabled": True,
            "label": "Max Interpolation Distance (m)",
            "main": True,
            "value": 500.0
        },
        "resolution": {
            "enabled": True,
            "label": "Base grid resolution (m)",
            "main": True,
            "value": 50.0
        },
        "export_as": {
            "main": True,
            "label": "Name",
            "value": "Iso"
        }
    }
)

validations = {}

app_initializer = {
    "geoh5": "../../assets/FlinFlon.geoh5",
    "objects": "{2e814779-c35f-4da0-ad6a-39a6912361f9}",
    "data": "{f3e36334-be0a-4210-b13e-06933279de25}",
    "max_distance": 500.0,
    "resolution": 50.0,
    "contours": "0.005: 0.02: 0.005, 0.0025",
}
