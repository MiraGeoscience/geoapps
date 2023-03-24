#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from uuid import UUID

from geoh5py.objects.surveys.electromagnetics.airborne_tem import AirborneTEMReceivers

from geoapps.inversion import default_ui_json as base_default_ui_json
from geoapps.inversion.constants import validations as base_validations

################# defaults ##################

inversion_defaults = {
    "title": "Time domain electromagnetic inversion",
    "inversion_type": "tdem",
    "geoh5": None,  # Must remain at top of list for notebook app initialization
    "forward_only": False,
    "topography_object": None,
    "topography": None,
    "data_object": None,
    "resolution": None,
    "z_from_topo": False,
    "receivers_radar_drape": None,
    "receivers_offset_z": None,
    "gps_receivers_offset": None,
    "z_channel": None,
    "z_uncertainty": None,
    "x_channel": None,
    "x_uncertainty": None,
    "y_channel": None,
    "y_uncertainty": None,
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
    "initial_beta_ratio": 1e2,
    "initial_beta": None,
    "coolingRate": 4,
    "coolingFactor": 2.0,
    "max_global_iterations": 50,
    "max_line_search_iterations": 20,
    "max_cg_iterations": 50,
    "tol_cg": 1e-4,
    "alpha_s": 0.0,
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
    "sens_wts_threshold": 5.0,
    "every_iteration_bool": True,
    "parallelized": True,
    "n_cpu": None,
    "tile_spatial": 1,
    "store_sensitivities": "ram",
    "max_ram": None,
    "max_chunk_size": 128,
    "chunk_by_rows": True,
    "out_group": "TimeDomainElectromagneticInversion",
    "generate_sweep": False,
    "monitoring_directory": None,
    "workspace_geoh5": None,
    "run_command": "geoapps.inversion.driver",
    "conda_environment": "geoapps",
    "distributed_workers": None,
    "z_channel_bool": False,
    "x_channel_bool": False,
    "y_channel_bool": False,
}

forward_defaults = {
    "title": "Time domain electromagnetic forward",
    "inversion_type": "tdem",
    "geoh5": None,  # Must remain at top of list for notebook app initialization
    "forward_only": True,
    "topography_object": None,
    "topography": None,
    "data_object": None,
    "resolution": None,
    "z_from_topo": False,
    "receivers_radar_drape": None,
    "receivers_offset_z": None,
    "gps_receivers_offset": None,
    "z_channel_bool": True,
    "x_channel_bool": True,
    "y_channel_bool": True,
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
    "out_group": "TimeDomainElectromagneticForward",
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
    "z_channel_bool": False,
    "x_channel_bool": False,
    "y_channel_bool": False,
    "detrend_type": None,
    "detrend_order": None,
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
    "title": "Time domain electromagnetic inversion",
    "inversion_type": "tdem",
    "data_object": {
        "main": True,
        "group": "Data",
        "label": "Object",
        "meshType": [
            "{19730589-fd28-4649-9de0-ad47249d9aba}",
            "{6a057fdc-b355-11e3-95be-fd84a7ffcb88}",
        ],
        "value": None,
    },
    "z_channel_bool": {
        "group": "Data",
        "main": True,
        "label": "Z",
        "value": False,
    },
    "z_channel": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Data",
        "dataGroupType": "Multi-element",
        "main": True,
        "label": "z-component",
        "parent": "data_object",
        "optional": True,
        "enabled": False,
        "value": None,
    },
    "z_uncertainty": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Data",
        "dataGroupType": "Multi-element",
        "main": True,
        "label": "Uncertainty",
        "parent": "data_object",
        "dependency": "z_channel",
        "dependencyType": "enabled",
        "value": None,
    },
    "x_channel_bool": {
        "group": "Data",
        "main": True,
        "label": "X",
        "value": False,
    },
    "x_channel": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Data",
        "dataGroupType": "Multi-element",
        "main": True,
        "label": "x-component",
        "parent": "data_object",
        "optional": True,
        "enabled": False,
        "value": None,
    },
    "x_uncertainty": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Data",
        "dataGroupType": "Multi-element",
        "main": True,
        "label": "Uncertainty",
        "parent": "data_object",
        "dependency": "x_channel",
        "dependencyType": "enabled",
        "value": None,
    },
    "y_channel_bool": {
        "group": "Data",
        "main": True,
        "label": "Y",
        "value": False,
    },
    "y_channel": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Data",
        "dataGroupType": "Multi-element",
        "main": True,
        "label": "y-component",
        "parent": "data_object",
        "optional": True,
        "enabled": False,
        "value": None,
    },
    "y_uncertainty": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Data",
        "dataGroupType": "Multi-element",
        "main": True,
        "label": "Uncertainty",
        "parent": "data_object",
        "dependency": "y_channel",
        "dependencyType": "enabled",
        "value": None,
    },
    "starting_model": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Mesh and Models",
        "main": True,
        "isValue": False,
        "parent": "mesh",
        "label": "Initial conductivity (S/m)",
        "property": None,
        "value": 1e-3,
    },
    "reference_model": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "main": True,
        "group": "Mesh and Models",
        "isValue": True,
        "parent": "mesh",
        "label": "Reference conductivity (S/m)",
        "property": None,
        "value": 1e-3,
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
    "out_group": {
        "label": "Results group name",
        "value": "TimeDomainElectromagneticInversion",
    },
}

default_ui_json = dict(base_default_ui_json, **default_ui_json)


################ Validations #################
validations = {
    "inversion_type": {
        "types": [str],
        "required": True,
        "values": ["tdem"],
    },
    "data_object": {"types": [str, UUID, AirborneTEMReceivers]},
    "z_channel": {"one_of": "data_channel"},
    "z_uncertainty": {"one_of": "uncertainty_channel"},
    "x_channel": {"one_of": "data_channel"},
    "x_uncertainty": {"one_of": "uncertainty_channel"},
    "y_channel": {"one_of": "data_channel"},
    "y_uncertainty": {"one_of": "uncertainty_channel"},
}
validations = dict(base_validations, **validations)

app_initializer = {}
#     "geoh5": str(assets_path() / "FlinFlon_natural_sources.geoh5"),
#     "topography_object": UUID("{cfabb8dd-d1ad-4c4e-a87c-7b3dd224c3f5}"),
#     "data_object": UUID("{9664afc1-cbda-4955-b936-526ca771f517}"),
#     "z_channel": UUID("{a73159fc-8c1b-411a-b435-12a5dac4a209}"),
#     "z_uncertainty": UUID("{e752e8d8-e8e3-4575-b20c-bc2d37cbd269}"),
#     "x_channel": UUID("{a73159fc-8c1b-411a-b435-12a5dac4a209}"),
#     "x_uncertainty": UUID("{e752e8d8-e8e3-4575-b20c-bc2d37cbd269}"),
#     "y_channel": UUID("{a73159fc-8c1b-411a-b435-12a5dac4a209}"),
#     "y_uncertainty": UUID("{e752e8d8-e8e3-4575-b20c-bc2d37cbd269}"),
#     "mesh": UUID("{1200396b-bc4a-4519-85e1-558c2dcac1dd}"),
#     "starting_model": 0.0003,
#     "reference_model": 0.0003,
#     "background_conductivity": 0.0003,
#     "resolution": 200.0,
#     "window_center_x": None,
#     "window_center_y": None,
#     "window_width": None,
#     "window_height": None,
#     "window_azimuth": None,
#     "octree_levels_topo": [0, 0, 4, 4],
#     "octree_levels_obs": [4, 4, 4, 4],
#     "depth_core": 500.0,
#     "horizontal_padding": 1000.0,
#     "vertical_padding": 1000.0,
#     "s_norm": 0.0,
#     "x_norm": 2.0,
#     "y_norm": 2.0,
#     "z_norm": 2.0,
#     "upper_bound": 100.0,
#     "lower_bound": 1e-5,
#     "max_global_iterations": 50,
# }
