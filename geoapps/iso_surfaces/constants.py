#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from copy import deepcopy
from uuid import UUID

from geoh5py.ui_json.constants import default_ui_json as base_ui_json

defaults = {
    "title": "Create Iso Surfaces",
    "geoh5": None,
    "objects": None,
    "data": None,
    "interval_min": 0.005,
    "interval_max": 0.02,
    "interval_spacing": 0.005,
    "fixed_contours": "0.0025",
    "max_distance": 500.0,
    "resolution": 50.0,
    "generate_sweep": False,
    "run_command": "geoapps.iso_surfaces.driver",
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
        "run_command": "geoapps.iso_surfaces.driver",
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
            "group": "Interval Contours",
            "groupOptional": True,
            "label": "Interval min",
            "value": 0.0,
        },
        "interval_max": {
            "main": True,
            "group": "Interval Contours",
            "label": "Interval max",
            "value": 0.0,
        },
        "interval_spacing": {
            "main": True,
            "group": "Interval Contours",
            "label": "Interval spacing",
            "value": 0.0,
        },
        "fixed_contours": {
            "main": True,
            "label": "Fixed Contours",
            "value": "0",
            "optional": True,
            "enabled": True,
        },
        "max_distance": {
            "enabled": True,
            "label": "Max Interpolation Distance (m)",
            "main": True,
            "value": 500.0,
        },
        "resolution": {
            "enabled": True,
            "label": "Base grid resolution (m)",
            "main": True,
            "value": 50.0,
        },
        "generate_sweep": {
            "label": "Generate sweep file",
            "group": "Python run preferences",
            "main": True,
            "value": False,
        },
        "export_as": {"main": True, "label": "Name", "value": "Iso_"},
    }
)

validations = {}

app_initializer = {
    "geoh5": "../../assets/FlinFlon.geoh5",
    "objects": UUID("{2e814779-c35f-4da0-ad6a-39a6912361f9}"),
    "data": UUID("{f3e36334-be0a-4210-b13e-06933279de25}"),
    "max_distance": 500.0,
    "resolution": 50.0,
    "interval_min": 0.005,
    "interval_max": 0.02,
    "interval_spacing": 0.005,
    "fixed_contours": "0.0025",
    "export_as": "Iso_Iteration_7_model",
}
