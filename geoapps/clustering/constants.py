#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from copy import deepcopy

import plotly.express as px
from geoh5py.ui_json.constants import default_ui_json as base_ui_json

defaults = {
    "title": "Clustering",
    "run_command": "geoapps.clustering.application",
    "run_command_boolean": False,
    "monitoring_directory": None,
    "workspace_geoh5": None,
    "conda_environment": "geoapps",
    "conda_environment_boolean": False,
}

default_ui_json = deepcopy(base_ui_json)
default_ui_json.update(
    {
        "title": "Clustering",
        "conda_environment": "geoapps",
        "run_command": "geoapps.scatter_plot.application",
    }
)

validations = {}

app_initializer = {
    "geoh5": "../../assets/FlinFlon.geoh5",
    "objects": "{79b719bc-d996-4f52-9af0-10aa9c7bb941}",
    "x": "{cdd7668a-4b5b-49ac-9365-c9ce4fddf733}",
    "x_log": False,
    "x_min": -17.0,
    "x_max": 25.5,
    "y": "{18c2560c-6161-468a-8571-5d9d59649535}",
    "y_log": True,
    "y_min": -17.0,
    "y_max": 29.8,
    "z": "{cb35da1c-7ea4-44f0-8817-e3d80e8ba98c}",
    "z_log": True,
    "z_min": -20.0,
    "z_max": 3200.0,
    "color": "{94a150e8-16d9-4784-a7aa-e6271df3a3ef}",
    "color_log": True,
    "color_min": -17.0,
    "color_max": 640.0,
    "color_maps": "inferno",
    "size": "{41d51965-3670-43ba-8a10-d399070689e3}",
    "size_log": False,
    "size_min": -17.0,
    "size_max": 24.8,
    "downsampling": 80,
    "size_markers": 20,
}
