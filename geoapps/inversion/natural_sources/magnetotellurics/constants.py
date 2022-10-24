#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from uuid import UUID

from geoh5py.objects.surveys.electromagnetics.magnetotellurics import MTReceivers

from geoapps.inversion import default_ui_json as base_default_ui_json
from geoapps.inversion.constants import validations as base_validations

################# defaults ##################

inversion_defaults = {
    "title": "SimPEG Magnetotellurics inversion",
    "inversion_type": "magnetotellurics",
    "geoh5": None,  # Must remain at top of list for notebook app initialization
    "forward_only": False,
    "topography_object": None,
    "topography": None,
    "data_object": None,
    "resolution": None,
    "z_from_topo": False,
    "receivers_radar_drape": None,
    "receivers_offset_x": None,
    "receivers_offset_y": None,
    "receivers_offset_z": None,
    "gps_receivers_offset": None,
    "zxx_real_channel": None,
    "zxx_real_uncertainty": None,
    "zxx_imag_channel": None,
    "zxx_imag_uncertainty": None,
    "zxy_real_channel": None,
    "zxy_real_uncertainty": None,
    "zxy_imag_channel": None,
    "zxy_imag_uncertainty": None,
    "zyx_real_channel": None,
    "zyx_real_uncertainty": None,
    "zyx_imag_channel": None,
    "zyx_imag_uncertainty": None,
    "zyy_real_channel": None,
    "zyy_real_uncertainty": None,
    "zyy_imag_channel": None,
    "zyy_imag_uncertainty": None,
    "mesh": None,
    "background_conductivity": 1e-3,
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
    "coolingRate": 3,
    "coolingFactor": 2.0,
    "max_global_iterations": 100,
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
    "sens_wts_threshold": 60.0,
    "every_iteration_bool": False,
    "parallelized": True,
    "n_cpu": None,
    "tile_spatial": 1,
    "store_sensitivities": "disk",
    "max_ram": None,
    "max_chunk_size": 128,
    "chunk_by_rows": True,
    "out_group": "MagnetotelluricsInversion",
    "monitoring_directory": None,
    "workspace_geoh5": None,
    "run_command": "geoapps.inversion.driver",
    "run_command_boolean": False,
    "conda_environment": "geoapps",
    "distributed_workers": None,
    "zxx_real_channel_bool": False,
    "zxx_imag_channel_bool": False,
    "zxy_real_channel_bool": False,
    "zxy_imag_channel_bool": False,
    "zyx_real_channel_bool": False,
    "zyx_imag_channel_bool": False,
    "zyy_real_channel_bool": False,
    "zyy_imag_channel_bool": False,
}

forward_defaults = {
    "title": "SimPEG Magnetotellurics Forward",
    "inversion_type": "magnetotellurics",
    "geoh5": None,  # Must remain at top of list for notebook app initialization
    "forward_only": True,
    "topography_object": None,
    "topography": None,
    "data_object": None,
    "resolution": None,
    "z_from_topo": False,
    "receivers_radar_drape": None,
    "receivers_offset_x": None,
    "receivers_offset_y": None,
    "receivers_offset_z": None,
    "gps_receivers_offset": None,
    "zxx_real_channel_bool": False,
    "zxx_imag_channel_bool": False,
    "zxy_real_channel_bool": False,
    "zxy_imag_channel_bool": False,
    "zyx_real_channel_bool": False,
    "zyx_imag_channel_bool": False,
    "zyy_real_channel_bool": False,
    "zyy_imag_channel_bool": False,
    "mesh": None,
    "background_conductivity": 1e-3,
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
    "out_group": "MagnetotelluricsForward",
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
    "zxx_real_channel_bool": False,
    "zxx_imag_channel_bool": False,
    "zxy_real_channel_bool": False,
    "zxy_imag_channel_bool": False,
    "zyx_real_channel_bool": False,
    "zyx_imag_channel_bool": False,
    "zyy_real_channel_bool": False,
    "zyy_imag_channel_bool": False,
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
    "title": "SimPEG Magnetotellurics inversion",
    "inversion_type": "magnetotellurics",
    "data_object": {
        "main": True,
        "group": "Survey",
        "label": "Object",
        "meshType": "{b99bd6e5-4fe1-45a5-bd2f-75fc31f91b38}",
        "value": None,
    },
    "zxx_real_channel_bool": {
        "group": "Data",
        "main": True,
        "label": "Zxx real",
        "value": False,
    },
    "zxx_real_channel": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Data",
        "dataGroupType": "Multi-element",
        "main": True,
        "label": "Zxx real",
        "parent": "data_object",
        "optional": True,
        "enabled": False,
        "value": None,
    },
    "zxx_real_uncertainty": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Data",
        "dataGroupType": "Multi-element",
        "main": True,
        "label": "Uncertainty",
        "parent": "data_object",
        "dependency": "zxx_real_channel",
        "dependencyType": "enabled",
        "value": None,
    },
    "zxx_imag_channel_bool": {
        "group": "Data",
        "main": True,
        "label": "Zxx imaginary",
        "value": False,
    },
    "zxx_imag_channel": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Data",
        "dataGroupType": "Multi-element",
        "main": True,
        "label": "Zxx imaginary",
        "parent": "data_object",
        "optional": True,
        "enabled": False,
        "value": None,
    },
    "zxx_imag_uncertainty": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Data",
        "dataGroupType": "Multi-element",
        "main": True,
        "label": "Uncertainty",
        "parent": "data_object",
        "dependency": "zxx_imag_channel",
        "dependencyType": "enabled",
        "value": None,
    },
    "zxy_real_channel_bool": {
        "group": "Data",
        "main": True,
        "label": "Zxy real",
        "value": False,
    },
    "zxy_real_channel": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Data",
        "dataGroupType": "Multi-element",
        "main": True,
        "label": "Zxy real",
        "parent": "data_object",
        "optional": True,
        "enabled": False,
        "value": None,
    },
    "zxy_real_uncertainty": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Data",
        "dataGroupType": "Multi-element",
        "main": True,
        "label": "Uncertainty",
        "parent": "data_object",
        "dependency": "zxy_real_channel",
        "dependencyType": "enabled",
        "value": None,
    },
    "zxy_imag_channel_bool": {
        "group": "Data",
        "main": True,
        "label": "Zxy imaginary",
        "value": False,
    },
    "zxy_imag_channel": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Data",
        "dataGroupType": "Multi-element",
        "main": True,
        "label": "Zxy imaginary",
        "parent": "data_object",
        "optional": True,
        "enabled": False,
        "value": None,
    },
    "zxy_imag_uncertainty": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Data",
        "dataGroupType": "Multi-element",
        "main": True,
        "label": "Uncertainty",
        "parent": "data_object",
        "dependency": "zxy_imag_channel",
        "dependencyType": "enabled",
        "value": None,
    },
    "zyx_real_channel_bool": {
        "group": "Data",
        "main": True,
        "label": "Zyx real",
        "value": False,
    },
    "zyx_real_channel": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Data",
        "dataGroupType": "Multi-element",
        "main": True,
        "label": "Zyx real",
        "parent": "data_object",
        "optional": True,
        "enabled": False,
        "value": None,
    },
    "zyx_real_uncertainty": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Data",
        "dataGroupType": "Multi-element",
        "main": True,
        "label": "Uncertainty",
        "parent": "data_object",
        "dependency": "zyx_real_channel",
        "dependencyType": "enabled",
        "value": None,
    },
    "zyx_imag_channel_bool": {
        "group": "Data",
        "main": True,
        "label": "Zyx imaginary",
        "value": False,
    },
    "zyx_imag_channel": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Data",
        "dataGroupType": "Multi-element",
        "main": True,
        "label": "Zyx imaginary",
        "parent": "data_object",
        "optional": True,
        "enabled": False,
        "value": None,
    },
    "zyx_imag_uncertainty": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Data",
        "dataGroupType": "Multi-element",
        "main": True,
        "label": "Uncertainty",
        "parent": "data_object",
        "dependency": "zyx_imag_channel",
        "dependencyType": "enabled",
        "value": None,
    },
    "zyy_real_channel_bool": {
        "group": "Data",
        "main": True,
        "label": "Zyy real",
        "value": False,
    },
    "zyy_real_channel": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Data",
        "dataGroupType": "Multi-element",
        "main": True,
        "label": "Zyy real",
        "parent": "data_object",
        "optional": True,
        "enabled": False,
        "value": None,
    },
    "zyy_real_uncertainty": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Data",
        "dataGroupType": "Multi-element",
        "main": True,
        "label": "Uncertainty",
        "parent": "data_object",
        "dependency": "zyy_real_channel",
        "dependencyType": "enabled",
        "value": None,
    },
    "zyy_imag_channel_bool": {
        "group": "Data",
        "main": True,
        "label": "Zyy imaginary",
        "value": False,
    },
    "zyy_imag_channel": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Data",
        "dataGroupType": "Multi-element",
        "main": True,
        "label": "Zyy imaginary",
        "parent": "data_object",
        "optional": True,
        "enabled": False,
        "value": None,
    },
    "zyy_imag_uncertainty": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Data",
        "dataGroupType": "Multi-element",
        "main": True,
        "label": "Uncertainty",
        "parent": "data_object",
        "dependency": "zyy_imag_channel",
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
        "label": "Initial Conductivity (Siemens/m)",
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
        "label": "Reference Conductivity (Siemens/m)",
        "property": None,
        "value": 1e-3,
    },
    "background_conductivity": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Mesh and Models",
        "main": True,
        "isValue": True,
        "parent": "mesh",
        "label": "Background conductivity (Siemens/m)",
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
        "label": "Lower bound (Siemens/m)",
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
        "label": "Upper bound (Siemens/m)",
        "property": None,
        "optional": True,
        "value": 100.0,
        "enabled": False,
    },
    "out_group": {"label": "Results group name", "value": "MagnetotelluricsInversion"},
}

default_ui_json = dict(base_default_ui_json, **default_ui_json)


################ Validations #################
validations = {
    "inversion_type": {
        "types": [str],
        "required": True,
        "values": ["magnetotellurics"],
    },
    "data_object": {"types": [str, UUID, MTReceivers]},
    "zxx_real_channel": {"one_of": "data_channel"},
    "zxx_real_uncertainty": {"one_of": "uncertainty_channel"},
    "zxx_imag_channel": {"one_of": "data_channel"},
    "zxx_imag_uncertainty": {"one_of": "uncertainty_channel"},
    "zxy_real_channel": {"one_of": "data_channel"},
    "zxy_real_uncertainty": {"one_of": "uncertainty_channel"},
    "zxy_imag_channel": {"one_of": "data_channel"},
    "zxy_imag_uncertainty": {"one_of": "uncertainty_channel"},
    "zyx_real_channel": {"one_of": "data_channel"},
    "zyx_real_uncertainty": {"one_of": "uncertainty_channel"},
    "zyx_imag_channel": {"one_of": "data_channel"},
    "zyx_imag_uncertainty": {"one_of": "uncertainty_channel"},
    "zyy_real_channel": {"one_of": "data_channel"},
    "zyy_real_uncertainty": {"one_of": "uncertainty_channel"},
    "zyy_imag_channel": {"one_of": "data_channel"},
    "zyy_imag_uncertainty": {"one_of": "uncertainty_channel"},
}
validations = dict(base_validations, **validations)

app_initializer = {
    "geoh5": "../../../assets/FlinFlon_natural_sources.geoh5",
    "topography_object": UUID("{cfabb8dd-d1ad-4c4e-a87c-7b3dd224c3f5}"),
    "data_object": UUID("{c9643051-e623-42d5-b33b-dbc6b2b5b876}"),
    "zxx_real_channel": UUID("{bd103ff8-5cc1-4acc-a23b-3dd9b875c20f}"),
    "zxx_real_uncertainty": UUID("{684297aa-f09d-41ab-9c4c-bd9085818e3e}"),
    "zxx_imag_channel": UUID("{6d463e8b-0e27-4969-bddb-6af21983c469}"),
    "zxx_imag_uncertainty": UUID("{4f514be9-3420-49ff-8e71-cb44160dfd83}"),
    "zxy_real_channel": UUID("{89cb9fc3-1220-4649-80a5-fdcaef8e398e}"),
    "zxy_real_uncertainty": UUID("{79de7a3b-6505-4ee3-aa3f-53966551367c}"),
    "zxy_imag_channel": UUID("{7fea3bf2-4309-4612-a7f2-6e04ffd4f4ea}"),
    "zxy_imag_uncertainty": UUID("{fa4a4311-4059-4885-b2f0-a787f9837a47}"),
    "zyx_real_channel": UUID("{33419705-6555-475c-872c-37fcf69aa351}"),
    "zyx_real_uncertainty": UUID("{45e0fe91-e18d-42b3-80f6-59235b93ec32}"),
    "zyx_imag_channel": UUID("{e294ddcd-4a87-4285-8cc6-0a92c0345a4a}"),
    "zyx_imag_uncertainty": UUID("{07b9bb1f-3b2a-4fba-be09-6f69ddd9b66f}"),
    "zyy_real_channel": UUID("{fe88daca-060c-416f-8011-7fae78fe7582}"),
    "zyy_real_uncertainty": UUID("{cbf999bc-f9b9-4c0c-8198-9dac8135e3a3}"),
    "zyy_imag_channel": UUID("{a8e3f100-43ad-4b37-a56a-34c6cae29f14}"),
    "zyy_imag_uncertainty": UUID("{52212149-d873-4adc-aae1-08414d7f9a9b}"),
    "mesh": UUID("{1200396b-bc4a-4519-85e1-558c2dcac1dd}"),
    "starting_model": 0.0003,
    "reference_model": 0.0003,
    "background_conductivity": 0.0003,
    "resolution": 200.0,
    "window_center_x": None,
    "window_center_y": None,
    "window_width": None,
    "window_height": None,
    "window_azimuth": None,
    "octree_levels_topo": [0, 0, 4, 4],
    "octree_levels_obs": [4, 4, 4, 4],
    "depth_core": 500.0,
    "horizontal_padding": 1000.0,
    "vertical_padding": 1000.0,
    "s_norm": 0.0,
    "x_norm": 2.0,
    "y_norm": 2.0,
    "z_norm": 2.0,
    "upper_bound": 100.0,
    "lower_bound": 1e-5,
    "max_global_iterations": 15,
}
