#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from copy import deepcopy
from uuid import UUID

import plotly.express as px

from geoh5py.ui_json.constants import default_ui_json as base_ui_json

defaults = {
    "title": "Scatter Plot",
    "geoh5": None,
    "objects": None,
    "data": None,
    "x": None,
    "x_active": True,
    "x_log": True,
    "y": None,
    "y_active": True,
    "y_log": True,
    "z": None,
    "z_log": True,
    "z_active": True,
    "color": None,
    "color_active": True,
    "color_log": True,
    "color_maps": "",
    "size": None,
    "size_active": True,
    "size_log": True,
    "run_command": "geoapps.scatter_plot.driver",
    "run_command_boolean": False,
    "monitoring_directory": None,
    "workspace_geoh5": None,
    "conda_environment": "geoapps",
    "conda_environment_boolean": False,
}

default_ui_json = deepcopy(base_ui_json)
default_ui_json.update(
    {
        "title": "Scatter Plot",
        "geoh5": None,
        "objects": {
            "label": "Object",
            "main": True,
            "meshType": [
                "{202C5DB1-A56D-4004-9CAD-BAAFD8899406}",
                "{6A057FDC-B355-11E3-95BE-FD84A7FFCB88}",
                "{F26FEBA3-ADED-494B-B9E9-B2BBCBE298E1}",
                "{B020A277-90E2-4CD7-84D6-612EE3F25051}",
                "{48f5054a-1c5c-4ca4-9048-80f36dc60a06}",
                "{7CAEBF0E-D16E-11E3-BC69-E4632694AA37}",
            ],
            "value": None,
        },
        "data": {
            "association": ["Vertex"],
            "dataType": "Float",
            "label": "Data",
            "main": True,
            "parent": "objects",
            "value": None,
        },
        "x": {
            "association": ["Vertex"],
            "dataType": "Float",
            "label": "Data",
            "main": True,
            "parent": "objects",
            "value": None,
        },
        "x_active": {
            "label": "Active",
            "main": True,
            "value": True,
        },
        "x_log": {
            "label": "Log10",
            "main": True,
            "value": True,
        },
        "y": {
            "association": ["Vertex"],
            "dataType": "Float",
            "label": "Data",
            "main": True,
            "parent": "objects",
            "value": None,
        },
        "y_active": {
            "label": "Active",
            "main": True,
            "value": True,
        },
        "y_log": {
            "label": "Log10",
            "main": True,
            "value": True,
        },
        "z": {
            "association": ["Vertex"],
            "dataType": "Float",
            "label": "Data",
            "main": True,
            "parent": "objects",
            "value": None,
        },
        "z_active": {
            "label": "Active",
            "main": True,
            "value": True,
        },
        "z_log": {
            "label": "Log10",
            "main": True,
            "value": True,
        },
        "color": {
            "association": ["Vertex"],
            "dataType": "Float",
            "label": "Data",
            "main": True,
            "parent": "objects",
            "value": None,
        },
        "color_active": {
            "label": "Active",
            "main": True,
            "value": False,
        },
        "color_log": {
            "label": "Log10",
            "main": True,
            "value": True,
        },
        "color_maps": {
            "choiceList": px.colors.named_colorscales(),
            "label": "Colormaps",
            "main": True,
            "value": "",
        },
        "size": {
            "association": ["Vertex"],
            "dataType": "Float",
            "label": "Data",
            "main": True,
            "parent": "objects",
            "value": None,
        },
        "size_active": {
            "label": "Active",
            "main": True,
            "value": False,
        },
        "size_log": {
            "label": "Log10",
            "main": True,
            "value": True,
        },
        "conda_environment": "geoapps",
        "run_command": "geoapps.scatter_plot.driver",
    }
)

validations = {}

app_initializer = {
    "objects": "{79b719bc-d996-4f52-9af0-10aa9c7bb941}",
    "data": [
        "{18c2560c-6161-468a-8571-5d9d59649535}",
        "{41d51965-3670-43ba-8a10-d399070689e3}",
        "{94a150e8-16d9-4784-a7aa-e6271df3a3ef}",
        "{cb35da1c-7ea4-44f0-8817-e3d80e8ba98c}",
        "{cdd7668a-4b5b-49ac-9365-c9ce4fddf733}",
    ],
    "x": "{cdd7668a-4b5b-49ac-9365-c9ce4fddf733}",
    "x_active": True,
    "x_log": True,
    "y": "{18c2560c-6161-468a-8571-5d9d59649535}",
    "y_active": True,
    "y_log": True,
    "z": "{cb35da1c-7ea4-44f0-8817-e3d80e8ba98c}",
    "z_active": True,
    "z_log": True,
    "color": "{94a150e8-16d9-4784-a7aa-e6271df3a3ef}",
    "color_active": True,
    "color_log": True,
    "size": "{41d51965-3670-43ba-8a10-d399070689e3}",
    "size_active": True,
    "size_log": True,
    "color_maps": "inferno",
}
