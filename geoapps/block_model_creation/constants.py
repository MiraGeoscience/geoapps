#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).
import os
from copy import deepcopy

from geoh5py.ui_json.constants import default_ui_json as base_ui_json

defaults = {
    "title": "Block Model Creation",
    "geoh5": None,
    "cell_size_x": None,
    "cell_size_y": None,
    "cell_size_z": None,
    "horizontal_padding": None,
    "bottom_padding": None,
    "depth_core": None,
    "expansion_fact": None,
    "new_grid": None,
    "generate_sweep": False,
    "workspace_geoh5": None,
    "conda_environment": "geoapps",
    "conda_environment_boolean": False,
    "run_command": "geoapps.block_model_creation.driver",
    "monitoring_directory": None,
}

default_ui_json = deepcopy(base_ui_json)
default_ui_json.update(
    {
        "title": "Block Model Creation",
        "geoh5": "",
        "run_command": "geoapps.block_model_creation.driver",
        "monitoring_directory": "",
        "conda_environment": "geoapps",
        "conda_environment_boolean": False,
        "objects": {
            "main": True,
            "meshType": [
                "{2e814779-c35f-4da0-ad6a-39a6912361f9}",
                "{202C5DB1-A56D-4004-9CAD-BAAFD8899406}",
                "{6A057FDC-B355-11E3-95BE-FD84A7FFCB88}",
                "{F26FEBA3-ADED-494B-B9E9-B2BBCBE298E1}",
                "{4EA87376-3ECE-438B-BF12-3479733DED46}",
                "{48f5054a-1c5c-4ca4-9048-80f36dc60a06}",
            ],
            "label": "Object",
            "value": None,
        },
        "new_grid": {
            "main": True,
            "label": "Name",
            "value": "",
        },
        "cell_size_x": {
            "main": True,
            "label": "Minimum x cell size",
            "value": 1.0,
            "min": 1e-14,
        },
        "cell_size_y": {
            "main": True,
            "label": "Minimum y cell size",
            "value": 1.0,
            "min": 1e-14,
        },
        "cell_size_z": {
            "main": True,
            "label": "Minimum z cell size",
            "value": 1.0,
            "min": 1e-14,
        },
        "depth_core": {
            "main": True,
            "label": "Core depth (m)",
            "value": 0.0,
            "min": 0.0,
        },
        "horizontal_padding": {
            "main": True,
            "label": "Horizontal padding",
            "value": 0.0,
            "min": 0.0,
        },
        "bottom_padding": {
            "main": True,
            "label": "Bottom padding",
            "value": 0.0,
            "min": 0.0,
        },
        "expansion_fact": {
            "main": True,
            "label": "Expansion factor",
            "value": 0.0,
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
    "monitoring_directory": os.path.abspath("../../assets"),
    "objects": "{2e814779-c35f-4da0-ad6a-39a6912361f9}",
    "cell_size_x": 50.0,
    "cell_size_y": 50.0,
    "cell_size_z": 50.0,
    "depth_core": 500.0,
    "expansion_fact": 1.05,
    "new_grid": "BlockModel",
    "horizontal_padding": 500.0,
    "bottom_padding": 500.0,
}
