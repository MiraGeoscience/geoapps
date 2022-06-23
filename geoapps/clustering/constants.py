#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from copy import deepcopy

# import plotly.express as px
# from geoh5py.ui_json.constants import default_ui_json as base_ui_json
from geoapps.scatter_plot.constants import default_ui_json as base_default_ui_json

defaults = {
    "title": "Clustering",
    "n_clusters": None,
    "scale": None,
    "run_command": "geoapps.clustering.application",
    "run_command_boolean": False,
    "monitoring_directory": None,
    "workspace_geoh5": None,
    "conda_environment": "geoapps",
    "conda_environment_boolean": False,
}

default_ui_json = deepcopy(base_default_ui_json)
default_ui_json.update(
    {
        "title": "Clustering",
        "data": {
            "association": ["Vertex"],
            "dataType": "Float",
            "group": "Data Selection",
            "label": "Data",
            "main": True,
            "parent": "objects",
            "value": None,
        },
        "n_clusters": {
            "group": "data",
            "label": "Number of clusters",
            "main": True,
            "value": 0,
            "lineEdit": False,
        },
        "channel": {
            "association": ["Vertex"],
            "dataType": "Float",
            "group": "data",
            "label": "Data",
            "main": True,
            "parent": "objects",
            "value": None,
        },
        "scale": {
            "group": "data",
            "label": "Scale",
            "main": True,
            "value": 0,
            "lineEdit": False,
        },
        "lower_bounds": {
            "group": "data",
            "label": "Lower bounds",
            "main": True,
            "value": 0.0,
            "lineEdit": False,
        },
        "upper_bounds": {
            "group": "x axis",
            "label": "Upper bounds",
            "main": True,
            "value": 0.0,
            "lineEdit": False,
        },
        "ga_group_name": {
            "main": True,
            "label": "Group",
            "value": "",
            "group": "Python run preferences",
        },
        "full_scales": None,
        "full_lower_bounds": None,
        "full_upper_bounds": None,
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
    "data": "{41d51965-3670-43ba-8a10-d399070689e3}",
    "n_clusters": 8,
    "channel": "{41d51965-3670-43ba-8a10-d399070689e3}",
    "scale": 5,
    "lower_bounds": -5.0,
    "upper_bounds": 200.0,
}
