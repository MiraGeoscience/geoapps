#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from uuid import UUID

from geoh5py.objects.surveys.direct_current import PotentialElectrode

from geoapps.inversion import default_ui_json as base_default_ui_json
from geoapps.inversion.constants import validations as base_validations

inversion_defaults = {
    "title": "Direct Current 2d inversion",
    "inversion_type": "direct current 2d",
    "geoh5": None,  # Must remain at top of list for notebook app initialization
    "forward_only": False,
    "topography_object": None,
    "topography": None,
    "data_object": None,
    "line_object": None,
    "line_id": 1,
    "resolution": None,
    "z_from_topo": False,
    "receivers_radar_drape": None,
    "receivers_offset_x": None,
    "receivers_offset_y": None,
    "receivers_offset_z": None,
    "gps_receivers_offset": None,
    "potential_channel": None,
    "potential_uncertainty": 1.0,
    "mesh": None,
    "u_cell_size": 25.0,
    "v_cell_size": 25.0,
    "depth_core": 500.0,
    "horizontal_padding": 1000.0,
    "vertical_padding": 1000.0,
    "expansion_factor": 1.1,
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
    "sens_wts_threshold": 30.0,
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
    "run_command": "geoapps.inversion.electricals.direct_current.two_dimensions.driver",
    "run_command_boolean": False,
    "conda_environment": "geoapps",
    "distributed_workers": None,
    "potential_channel_bool": True,
}
forward_defaults = {
    "title": "Direct Current 2d forward",
    "inversion_type": "direct current 2d",
    "geoh5": None,  # Must remain at top of list for notebook app initialization
    "forward_only": True,
    "topography_object": None,
    "topography": None,
    "data_object": None,
    "line_object": None,
    "line_id": 1,
    "resolution": None,
    "z_from_topo": False,
    "receivers_radar_drape": None,
    "receivers_offset_x": None,
    "receivers_offset_y": None,
    "receivers_offset_z": None,
    "gps_receivers_offset": None,
    "potential_channel_bool": True,
    "mesh": None,
    "u_cell_size": 25.0,
    "v_cell_size": 25.0,
    "depth_core": 500.0,
    "horizontal_padding": 1000.0,
    "vertical_padding": 1000.0,
    "expansion_factor": 1.1,
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
    "run_command_boolean": False,
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
    "title": "Direct Current inversion",
    "inversion_type": "direct current 2d",
    "line_object": {
        "association": ["Cell", "Vertex"],
        "dataType": "Referenced",
        "group": "Data",
        "main": True,
        "label": "Line field",
        "parent": "data_object",
        "value": None,
    },
    "line_id": {
        "group": "Data",
        "main": True,
        "min": 1,
        "label": "Line number",
        "value": 1,
    },
    "data_object": {
        "main": True,
        "group": "Data",
        "label": "Object",
        "meshType": "{275ecee9-9c24-4378-bf94-65f3c5fbe163}",
        "value": None,
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
    "mesh": {
        "group": "Mesh and Models",
        "main": True,
        "optional": True,
        "enabled": False,
        "label": "Mesh",
        "meshType": "{C94968EA-CF7D-11EB-B8BC-0242AC130003}",
        "value": None,
    },
    "starting_model": {
        "association": "Cell",
        "dataType": "Float",
        "group": "Mesh and Models",
        "main": True,
        "isValue": False,
        "parent": "mesh",
        "label": "Initial Conductivity (S/m)",
        "property": None,
        "value": 1e-3,
    },
    "reference_model": {
        "association": "Cell",
        "dataType": "Float",
        "main": True,
        "group": "Mesh and Models",
        "isValue": True,
        "parent": "mesh",
        "label": "Reference Conductivity (S/m)",
        "property": None,
        "value": 1e-3,
    },
    "lower_bound": {
        "association": "Cell",
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
        "association": "Cell",
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
    "expansion_factor": {
        "main": True,
        "group": "Mesh and Models",
        "label": "Expansion factor",
        "dependency": "mesh",
        "dependencyType": "disabled",
        "value": 1.1,
    },
    "window_center_x": {
        "group": "Data window",
        "enabled": False,
        "groupOptional": True,
        "label": "Window center easting",
        "value": 0.0,
        "visible": False,
    },
    "window_center_y": {
        "group": "Data window",
        "enabled": False,
        "label": "Window center northing",
        "value": 0.0,
        "visible": False,
    },
    "window_width": {
        "min": 0.0,
        "group": "Data window",
        "enabled": False,
        "label": "Window width",
        "value": 0.0,
        "visible": False,
    },
    "window_height": {
        "min": 0.0,
        "group": "Data window",
        "enabled": False,
        "label": "Window height",
        "value": 0.0,
        "visible": False,
    },
    "window_azimuth": {
        "min": -180,
        "max": 180,
        "group": "Data window",
        "enabled": False,
        "label": "Window azimuth",
        "value": 0.0,
        "visible": False,
    },
    "resolution": None,
    "detrend_order": None,
    "detrend_type": None,
    "tile_spatial": 1,
    "out_group": {"label": "Results group name", "value": "DirectCurrentInversion"},
}

default_ui_json = dict(base_default_ui_json, **default_ui_json)


################ Validations #################

validations = {
    "inversion_type": {
        "required": True,
        "values": ["direct current 2d"],
    },
    "data_object": {"required": True, "types": [UUID, PotentialElectrode]},
}

validations = dict(base_validations, **validations)

app_initializer = {
    "geoh5": "../../../assets/FlinFlon_dcip.geoh5",
    "data_object": UUID("{6e14de2c-9c2f-4976-84c2-b330d869cb82}"),
    "potential_channel": UUID("{502e7256-aafa-4016-969f-5cc3a4f27315}"),
    "potential_uncertainty": UUID("{62746129-3d82-427e-a84c-78cded00c0bc}"),
    "line_object": UUID("{d400e8f1-8460-4609-b852-b3b93f945770}"),
    "line_id": 1,
    "starting_model": 1e-1,
    "reference_model": 1e-1,
    "resolution": None,
    "window_center_x": None,
    "window_center_y": None,
    "window_width": None,
    "window_height": None,
    "window_azimuth": None,
    "s_norm": 0.0,
    "x_norm": 2.0,
    "y_norm": 2.0,
    "z_norm": 2.0,
    "upper_bound": 100.0,
    "lower_bound": 1e-5,
    "max_global_iterations": 25,
    "topography_object": UUID("{ab3c2083-6ea8-4d31-9230-7aad3ec09525}"),
    "topography": UUID("{a603a762-f6cb-4b21-afda-3160e725bf7d}"),
    "z_from_topo": True,
    "receivers_offset_x": 0.0,
    "receivers_offset_y": 0.0,
    "receivers_offset_z": 0.0,
    "out_grop": "DCInversion",
}
