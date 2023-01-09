#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from uuid import UUID

from geoh5py.objects.surveys.direct_current import PotentialElectrode

from geoapps.inversion.constants import default_ui_json as base_default_ui_json
from geoapps.inversion.constants import validations as base_validations

inversion_defaults = {
    "title": "Direct Current 3D inversion",
    "documentation": "https://geoapps.readthedocs.io/en/stable/content/applications/dcip_inversion.html",
    "icon": "PotentialElectrode",
    "inversion_type": "direct current 3d",
    "geoh5": None,  # Must remain at top of list for notebook app initialization
    "forward_only": False,
    "topography_object": None,
    "topography": None,
    "data_object": None,
    "resolution": None,
    "z_from_topo": True,
    "receivers_offset_z": None,
    "receivers_radar_drape": None,
    "gps_receivers_offset": None,
    "potential_channel": None,
    "potential_uncertainty": 1.0,
    "mesh": None,
    "starting_model": 1e-3,
    "reference_model": 1e-3,
    "lower_bound": None,
    "upper_bound": None,
    "output_tile_files": False,
    "ignore_values": None,
    "detrend_order": None,
    "detrend_type": None,
    "window_center_x": None,
    "window_center_y": None,
    "window_width": None,
    "window_height": None,
    "window_azimuth": None,
    "inversion_style": "voxel",
    "chi_factor": 1.0,
    "initial_beta_ratio": 10.0,
    "initial_beta": None,
    "coolingRate": 2,
    "coolingFactor": 2.0,
    "max_global_iterations": 50,
    "max_line_search_iterations": 20,
    "max_cg_iterations": 30,
    "tol_cg": 1e-4,
    "alpha_s": 1.0,
    "alpha_x": 1.0,
    "alpha_y": 1.0,
    "alpha_z": 1.0,
    "s_norm": 0.0,
    "x_norm": 2.0,
    "y_norm": 2.0,
    "z_norm": 2.0,
    "gradient_type": "total",
    "max_irls_iterations": 25,
    "starting_chi_factor": None,
    "f_min_change": 1e-4,
    "beta_tol": 0.5,
    "prctile": 95,
    "coolEps_q": True,
    "coolEpsFact": 1.2,
    "beta_search": False,
    "sens_wts_threshold": 80.0,
    "every_iteration_bool": True,
    "parallelized": True,
    "n_cpu": None,
    "tile_spatial": 1,
    "store_sensitivities": "ram",
    "max_ram": None,
    "max_chunk_size": 128,
    "chunk_by_rows": True,
    "out_group": "DirectCurrentInversion",
    "generate_sweep": False,
    "monitoring_directory": None,
    "workspace_geoh5": None,
    "run_command": "geoapps.inversion.driver",
    "conda_environment": "geoapps",
    "distributed_workers": None,
    "potential_channel_bool": True,
}
forward_defaults = {
    "title": "Direct Current 3D forward",
    "documentation": "https://geoapps.readthedocs.io/en/stable/content/applications/dcip_inversion.html",
    "icon": "PotentialElectrode",
    "inversion_type": "direct current 3d",
    "geoh5": None,  # Must remain at top of list for notebook app initialization
    "forward_only": True,
    "topography_object": None,
    "topography": None,
    "data_object": None,
    "resolution": None,
    "z_from_topo": True,
    "receivers_offset_z": None,
    "receivers_radar_drape": None,
    "gps_receivers_offset": None,
    "potential_channel_bool": True,
    "mesh": None,
    "starting_model": 1e-3,
    "output_tile_files": False,
    "window_center_x": None,
    "window_center_y": None,
    "window_width": None,
    "window_height": None,
    "window_azimuth": None,
    "parallelized": True,
    "n_cpu": None,
    "tile_spatial": 1,
    "max_chunk_size": 128,
    "chunk_by_rows": True,
    "out_group": "DirectCurrentForward",
    "generate_sweep": False,
    "monitoring_directory": None,
    "workspace_geoh5": None,
    "run_command": "geoapps.inversion.driver",
    "conda_environment": "geoapps",
    "distributed_workers": None,
    "gradient_type": "total",
    "alpha_s": 1.0,
    "alpha_x": 1.0,
    "alpha_y": 1.0,
    "alpha_z": 1.0,
    "s_norm": 0.0,
    "x_norm": 2.0,
    "y_norm": 2.0,
    "z_norm": 2.0,
}

inversion_ui_json = {
    "potential_channel_bool": True,
}

forward_ui_json = {
    "starting_model": {
        "association": "Cell",
        "dataType": "Float",
        "group": "Mesh and Models`",
        "main": True,
        "isValue": False,
        "parent": "mesh",
        "label": "Conductivity (S/m)",
        "property": None,
        "value": 0.0,
    },
    "gradient_type": "total",
    "alpha_s": 1.0,
    "alpha_x": 1.0,
    "alpha_y": 1.0,
    "alpha_z": 1.0,
    "s_norm": 0.0,
    "x_norm": 2.0,
    "y_norm": 2.0,
    "z_norm": 2.0,
}

default_ui_json = {
    "title": "Direct Current 3D inversion",
    "documentation": "https://geoapps.readthedocs.io/en/stable/content/applications/dcip_inversion.html",
    "icon": "PotentialElectrode",
    "inversion_type": "direct current 3d",
    "data_object": {
        "main": True,
        "group": "Data",
        "label": "Object",
        "meshType": "{275ecee9-9c24-4378-bf94-65f3c5fbe163}",
        "value": None,
    },
    "z_from_topo": {
        "group": "Data",
        "main": True,
        "label": "Surface survey",
        "tooltip": "Uncheck if borehole data is present",
        "value": True,
    },
    "potential_channel_bool": True,
    "potential_channel": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Data",
        "main": True,
        "label": "Potential (V/I)",
        "parent": "data_object",
        "value": None,
    },
    "potential_uncertainty": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Data",
        "main": True,
        "isValue": True,
        "label": "Uncertainty",
        "parent": "data_object",
        "property": None,
        "value": 1.0,
    },
    "starting_model": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Mesh and Models`",
        "main": True,
        "isValue": False,
        "parent": "mesh",
        "label": "Initial conductivity (S/m)",
        "property": None,
        "value": 1e-1,
    },
    "reference_model": {
        "association": ["Cell", "Vertex"],
        "main": True,
        "dataType": "Float",
        "group": "Mesh and Models",
        "isValue": True,
        "parent": "mesh",
        "label": "Reference conductivity (S/m)",
        "property": None,
        "value": 0.0,
    },
    "lower_bound": {
        "association": ["Cell", "Vertex"],
        "main": True,
        "dataType": "Float",
        "group": "Mesh and Models",
        "isValue": True,
        "parent": "mesh",
        "label": "Lower bound (S/m)",
        "property": None,
        "optional": True,
        "value": 1e-8,
        "enabled": False,
    },
    "upper_bound": {
        "association": ["Cell", "Vertex"],
        "main": True,
        "dataType": "Float",
        "group": "Mesh and Models",
        "isValue": True,
        "parent": "mesh",
        "label": "Upper bound (S/m)",
        "property": None,
        "optional": True,
        "value": 100.0,
        "enabled": False,
    },
    "receivers_offset_z": {
        "group": "Data pre-processing",
        "label": "Z static offset",
        "optional": True,
        "enabled": False,
        "value": 0.0,
        "visible": False,
    },
    "receivers_radar_drape": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Data pre-processing",
        "label": "Z radar offset",
        "tooltip": "Apply a non-homogeneous offset to survey object from radar channel.",
        "optional": True,
        "parent": "data_object",
        "value": None,
        "enabled": False,
        "visible": False,
    },
    "resolution": None,
    "detrend_order": None,
    "detrend_type": None,
    "out_group": {"label": "Results group name", "value": "DirectCurrentInversion"},
}

default_ui_json = dict(base_default_ui_json, **default_ui_json)


################ Validations #################

validations = {
    "inversion_type": {
        "required": True,
        "values": ["direct current 3d"],
    },
    "data_object": {"required": True, "types": [UUID, PotentialElectrode]},
}

validations = dict(base_validations, **validations)

app_initializer = {
    "geoh5": "../../../assets/FlinFlon_dcip.geoh5",
    "data_object": UUID("{6e14de2c-9c2f-4976-84c2-b330d869cb82}"),
    "potential_channel": UUID("{502e7256-aafa-4016-969f-5cc3a4f27315}"),
    "potential_uncertainty": UUID("{62746129-3d82-427e-a84c-78cded00c0bc}"),
    "mesh": UUID("{da109284-aa8c-4824-a647-29951109b058}"),
    "reference_model": 1e-1,
    "starting_model": 1e-1,
    "resolution": None,
    "window_center_x": None,
    "window_center_y": None,
    "window_width": None,
    "window_height": None,
    "window_azimuth": None,
    "octree_levels_topo": [0, 0, 0, 2],
    "octree_levels_obs": [5, 5, 5, 5],
    "depth_core": 500.0,
    "horizontal_padding": 1000.0,
    "vertical_padding": 1000.0,
    "s_norm": 0.0,
    "x_norm": 2.0,
    "y_norm": 2.0,
    "z_norm": 2.0,
    "upper_bound": 100.0,
    "lower_bound": 1e-5,
    "max_global_iterations": 25,
    "sens_wts_threshold": 80.0,
    "topography_object": UUID("{ab3c2083-6ea8-4d31-9230-7aad3ec09525}"),
    "topography": UUID("{a603a762-f6cb-4b21-afda-3160e725bf7d}"),
    "z_from_topo": True,
    "receivers_offset_z": 0.0,
}
