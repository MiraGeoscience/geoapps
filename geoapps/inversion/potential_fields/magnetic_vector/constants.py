#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import multiprocessing
from uuid import UUID

from geoh5py.objects import Grid2D, Points, Surface

import geoapps
from geoapps import assets_path
from geoapps.inversion import default_ui_json as base_default_ui_json
from geoapps.inversion.constants import validations as base_validations

inversion_defaults = {
    "version": geoapps.__version__,
    "title": "Magnetic Vector (MVI) Inversion",
    "documentation": "https://geoapps.readthedocs.io/en/stable/content/applications/grav_mag_inversion.html",
    "icon": "surveyairbornegravity",
    "inversion_type": "magnetic vector",
    "geoh5": None,  # Must remain at top of list for notebook app initialization
    "forward_only": False,
    "inducing_field_strength": 50000.0,
    "inducing_field_inclination": 90.0,
    "inducing_field_declination": 0.0,
    "topography_object": None,
    "topography": None,
    "data_object": None,
    "z_from_topo": False,
    "receivers_offset_z": None,
    "receivers_radar_drape": None,
    "gps_receivers_offset": None,
    "tmi_channel": None,
    "tmi_uncertainty": 1.0,
    "bx_channel": None,
    "bx_uncertainty": 1.0,
    "by_channel": None,
    "by_uncertainty": 1.0,
    "bz_channel": None,
    "bz_uncertainty": 1.0,
    "bxx_channel": None,
    "bxx_uncertainty": 1.0,
    "bxy_channel": None,
    "bxy_uncertainty": 1.0,
    "bxz_channel": None,
    "bxz_uncertainty": 1.0,
    "byy_channel": None,
    "byy_uncertainty": 1.0,
    "byz_channel": None,
    "byz_uncertainty": 1.0,
    "bzz_channel": None,
    "bzz_uncertainty": 1.0,
    "mesh": None,
    "starting_model": 1e-4,
    "reference_model": 0.0,
    "lower_bound": None,
    "upper_bound": None,
    "starting_inclination": None,
    "starting_declination": None,
    "reference_inclination": None,
    "reference_declination": None,
    "output_tile_files": False,
    "inversion_style": "voxel",
    "chi_factor": 1.0,
    "initial_beta_ratio": 100.0,
    "initial_beta": None,
    "coolingRate": 1,
    "coolingFactor": 2.0,
    "max_global_iterations": 100,
    "max_line_search_iterations": 20,
    "max_cg_iterations": 30,
    "tol_cg": 1e-4,
    "alpha_s": 1.0,
    "length_scale_x": 1.0,
    "length_scale_y": 1.0,
    "length_scale_z": 1.0,
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
    "sens_wts_threshold": 0.001,
    "every_iteration_bool": False,
    "parallelized": True,
    "n_cpu": None,
    "tile_spatial": 1,
    "store_sensitivities": "ram",
    "max_ram": None,
    "max_chunk_size": 128,
    "chunk_by_rows": True,
    "out_group": None,
    "ga_group": None,
    "generate_sweep": False,
    "monitoring_directory": None,
    "workspace_geoh5": None,
    "run_command": "geoapps.inversion.driver",
    "conda_environment": "geoapps",
    "distributed_workers": None,
}
forward_defaults = {
    "version": geoapps.__version__,
    "title": "Magnetic Vector (MVI) Forward",
    "documentation": "https://geoapps.readthedocs.io/en/stable/content/applications/grav_mag_inversion.html",
    "icon": "surveyairbornemagnetics",
    "inversion_type": "magnetic vector",
    "geoh5": None,  # Must remain at top of list for notebook app initialization
    "forward_only": True,
    "inducing_field_strength": 50000.0,
    "inducing_field_inclination": 90.0,
    "inducing_field_declination": 0.0,
    "topography_object": None,
    "topography": None,
    "data_object": None,
    "z_from_topo": False,
    "receivers_offset_z": None,
    "receivers_radar_drape": None,
    "gps_receivers_offset": None,
    "tmi_channel_bool": True,
    "bx_channel_bool": False,
    "by_channel_bool": False,
    "bz_channel_bool": False,
    "bxx_channel_bool": False,
    "bxy_channel_bool": False,
    "bxz_channel_bool": False,
    "byy_channel_bool": False,
    "byz_channel_bool": False,
    "bzz_channel_bool": False,
    "mesh": None,
    "starting_model": None,
    "starting_inclination": None,
    "starting_declination": None,
    "output_tile_files": False,
    "parallelized": True,
    "n_cpu": None,
    "tile_spatial": 1,
    "max_chunk_size": 128,
    "chunk_by_rows": True,
    "out_group": None,
    "ga_group": None,
    "generate_sweep": False,
    "monitoring_directory": None,
    "workspace_geoh5": None,
    "run_command": "geoapps.inversion.driver",
    "conda_environment": "geoapps",
    "distributed_workers": None,
}

default_ui_json = {
    "title": "Magnetic Vector (MVI) Inversion",
    "documentation": "https://geoapps.readthedocs.io/en/stable/content/applications/grav_mag_inversion.html",
    "icon": "surveyairbornegravity",
    "inversion_type": "magnetic vector",
    "inducing_field_strength": {
        "min": 0.1,
        "max": 100000.0,
        "precision": 2,
        "lineEdit": False,
        "main": True,
        "group": "Inducing Field",
        "label": "Strength (nT)",
        "value": 50000.0,
    },
    "inducing_field_inclination": {
        "min": -90.0,
        "max": 90.0,
        "precision": 2,
        "lineEdit": False,
        "main": True,
        "group": "Inducing Field",
        "label": "Inclination (deg)",
        "value": 90.0,
    },
    "inducing_field_declination": {
        "min": -180.0,
        "max": 180.0,
        "precision": 2,
        "lineEdit": False,
        "main": True,
        "group": "Inducing Field",
        "label": "Declination (deg)",
        "value": 0.0,
    },
    "data_object": {
        "main": True,
        "group": "Data",
        "label": "Object",
        "meshType": [
            "{202C5DB1-A56D-4004-9CAD-BAAFD8899406}",
            "{6A057FDC-B355-11E3-95BE-FD84A7FFCB88}",
            "{F26FEBA3-ADED-494B-B9E9-B2BBCBE298E1}",
            "{48F5054A-1C5C-4CA4-9048-80F36DC60A06}",
            "{b020a277-90e2-4cd7-84d6-612ee3f25051}",
            "{4b99204c-d133-4579-a916-a9c8b98cfccb}",
            "{028e4905-cc97-4dab-b1bf-d76f58b501b5}",
        ],
        "value": None,
    },
    "tmi_channel_bool": {
        "group": "Data",
        "main": True,
        "label": "TMI (nT)",
        "value": False,
    },
    "tmi_channel": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Data",
        "main": True,
        "label": "TMI (nT)",
        "parent": "data_object",
        "optional": True,
        "enabled": False,
        "value": None,
    },
    "tmi_uncertainty": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Data",
        "main": True,
        "isValue": True,
        "label": "Uncertainty",
        "parent": "data_object",
        "dependency": "tmi_channel",
        "dependencyType": "enabled",
        "property": None,
        "value": 1.0,
    },
    "bxx_channel_bool": {
        "group": "Data",
        "main": True,
        "label": "Bxx (nT/m)",
        "value": False,
    },
    "bxx_channel": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Data",
        "main": True,
        "label": "Bxx (nT/m)",
        "parent": "data_object",
        "optional": True,
        "enabled": False,
        "value": None,
    },
    "bxx_uncertainty": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Data",
        "main": True,
        "isValue": True,
        "label": "Uncertainty",
        "parent": "data_object",
        "dependency": "bxx_channel",
        "dependencyType": "enabled",
        "property": None,
        "value": 1.0,
    },
    "bxy_channel_bool": {
        "group": "Data",
        "main": True,
        "label": "Bxy (nT/m)",
        "value": False,
    },
    "bxy_channel": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Data",
        "main": True,
        "label": "Bxy (nT/m)",
        "parent": "data_object",
        "optional": True,
        "enabled": False,
        "value": None,
    },
    "bxy_uncertainty": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Data",
        "main": True,
        "isValue": True,
        "label": "Uncertainty",
        "parent": "data_object",
        "dependency": "bxy_channel",
        "dependencyType": "enabled",
        "property": None,
        "value": 1.0,
    },
    "bxz_channel_bool": {
        "group": "Data",
        "main": True,
        "label": "Bxz (nT/m)",
        "value": False,
    },
    "bxz_channel": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Data",
        "main": True,
        "label": "Bxz (nT/m)",
        "parent": "data_object",
        "optional": True,
        "enabled": False,
        "value": None,
    },
    "bxz_uncertainty": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Data",
        "main": True,
        "isValue": True,
        "label": "Uncertainty",
        "parent": "data_object",
        "dependency": "bxz_channel",
        "dependencyType": "enabled",
        "property": None,
        "value": 1.0,
    },
    "byy_channel_bool": {
        "group": "Data",
        "main": True,
        "label": "Byy (nT/m)",
        "value": False,
    },
    "byy_channel": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Data",
        "main": True,
        "label": "Byy (nT/m)",
        "parent": "data_object",
        "optional": True,
        "enabled": False,
        "value": None,
    },
    "byy_uncertainty": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Data",
        "main": True,
        "isValue": True,
        "label": "Uncertainty",
        "parent": "data_object",
        "dependency": "byy_channel",
        "dependencyType": "enabled",
        "property": None,
        "value": 1.0,
    },
    "byz_channel_bool": {
        "group": "Data",
        "main": True,
        "label": "Byz (nT/m)",
        "value": False,
    },
    "byz_channel": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Data",
        "main": True,
        "label": "Byz (nT/m)",
        "parent": "data_object",
        "optional": True,
        "enabled": False,
        "value": None,
    },
    "byz_uncertainty": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Data",
        "main": True,
        "isValue": True,
        "label": "Uncertainty",
        "parent": "data_object",
        "dependency": "byz_channel",
        "dependencyType": "enabled",
        "property": None,
        "value": 1.0,
    },
    "bzz_channel_bool": {
        "group": "Data",
        "main": True,
        "label": "Bzz (nT/m)",
        "value": False,
    },
    "bzz_channel": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Data",
        "main": True,
        "label": "Bzz (nT/m)",
        "parent": "data_object",
        "optional": True,
        "enabled": False,
        "value": None,
    },
    "bzz_uncertainty": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Data",
        "main": True,
        "isValue": True,
        "label": "Uncertainty",
        "parent": "data_object",
        "dependency": "bzz_channel",
        "dependencyType": "enabled",
        "property": None,
        "value": 1.0,
    },
    "bx_channel_bool": {
        "group": "Data",
        "main": True,
        "label": "Bx (nT)",
        "value": False,
    },
    "bx_channel": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Data",
        "main": True,
        "label": "Bx (nT)",
        "parent": "data_object",
        "optional": True,
        "enabled": False,
        "value": None,
    },
    "bx_uncertainty": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Data",
        "main": True,
        "isValue": True,
        "label": "Uncertainty",
        "parent": "data_object",
        "dependency": "bx_channel",
        "dependencyType": "enabled",
        "property": None,
        "value": 1.0,
    },
    "by_channel_bool": {
        "group": "Data",
        "main": True,
        "label": "By (nT)",
        "value": False,
    },
    "by_channel": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Data",
        "main": True,
        "label": "By (nT)",
        "parent": "data_object",
        "optional": True,
        "enabled": False,
        "value": None,
    },
    "by_uncertainty": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Data",
        "main": True,
        "isValue": True,
        "label": "Uncertainty",
        "parent": "data_object",
        "dependency": "by_channel",
        "dependencyType": "enabled",
        "property": None,
        "value": 1.0,
    },
    "bz_channel_bool": {
        "group": "Data",
        "main": True,
        "label": "Bz (nT)",
        "value": False,
    },
    "bz_channel": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Data",
        "main": True,
        "label": "Bz (nT)",
        "parent": "data_object",
        "optional": True,
        "enabled": False,
        "value": None,
    },
    "bz_uncertainty": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Data",
        "main": True,
        "isValue": True,
        "label": "Uncertainty",
        "parent": "data_object",
        "dependency": "bz_channel",
        "dependencyType": "enabled",
        "property": None,
        "value": 1.0,
    },
    "starting_model": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Mesh and models",
        "main": True,
        "isValue": False,
        "parent": "mesh",
        "label": "Initial susceptibility (SI)",
        "property": None,
        "value": 1e-4,
    },
    "starting_inclination": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Mesh and models",
        "main": True,
        "isValue": False,
        "optional": True,
        "enabled": False,
        "parent": "mesh",
        "label": "Initial inclination (deg)",
        "property": None,
        "value": 0.0,
    },
    "starting_inclination_object": {
        "group": "Mesh and models",
        "main": True,
        "meshType": [
            "{202C5DB1-A56D-4004-9CAD-BAAFD8899406}",
            "{6A057FDC-B355-11E3-95BE-FD84A7FFCB88}",
            "{F26FEBA3-ADED-494B-B9E9-B2BBCBE298E1}",
            "{48F5054A-1C5C-4CA4-9048-80F36DC60A06}",
            "{b020a277-90e2-4cd7-84d6-612ee3f25051}",
            "{4ea87376-3ece-438b-bf12-3479733ded46}",
        ],
        "optional": True,
        "enabled": False,
        "label": "Object",
        "value": None,
    },
    "starting_declination": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Mesh and models",
        "main": True,
        "isValue": False,
        "optional": True,
        "enabled": False,
        "parent": "mesh",
        "label": "Initial declination (deg)",
        "property": None,
        "value": 0.0,
    },
    "reference_model": {
        "association": ["Cell", "Vertex"],
        "main": True,
        "dataType": "Float",
        "group": "Mesh and models",
        "isValue": True,
        "optional": True,
        "enabled": False,
        "parent": "mesh",
        "label": "Reference susceptibility (SI)",
        "property": None,
        "value": 0.0,
    },
    "reference_inclination": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Mesh and models",
        "main": True,
        "isValue": False,
        "optional": True,
        "enabled": False,
        "label": "Reference inclination (deg)",
        "parent": "mesh",
        "property": None,
        "value": 0.0,
    },
    "reference_declination": {
        "association": ["Cell", "Vertex"],
        "dataType": "Float",
        "group": "Mesh and models",
        "main": True,
        "isValue": True,
        "optional": True,
        "enabled": False,
        "label": "Reference declination (deg)",
        "parent": "mesh",
        "property": None,
        "value": 0.0,
    },
    "lower_bound": {
        "association": ["Cell", "Vertex"],
        "main": True,
        "dataType": "Float",
        "group": "Mesh and models",
        "isValue": False,
        "parent": "mesh",
        "label": "Lower bound (SI)",
        "property": None,
        "optional": True,
        "value": 0.0,
        "enabled": False,
    },
    "upper_bound": {
        "association": ["Cell", "Vertex"],
        "main": True,
        "dataType": "Float",
        "group": "Mesh and models",
        "isValue": False,
        "parent": "mesh",
        "label": "Upper bound (SI)",
        "property": None,
        "optional": True,
        "value": 1.0,
        "enabled": False,
    },
}
default_ui_json = dict(base_default_ui_json, **default_ui_json)
validations = {
    "inversion_type": {
        "required": True,
        "values": ["magnetic vector"],
    },
    "data_object": {"required": True, "types": [str, UUID, Points, Surface, Grid2D]},
    "tmi_channel": {"one_of": "data channel"},
    "tmi_uncertainty": {"one_of": "uncertainty channel"},
    "bxx_channel": {"one_of": "data channel"},
    "bxx_uncertainty": {"one_of": "uncertainty channel"},
    "bxy_channel": {"one_of": "data channel"},
    "bxy_uncertainty": {"one_of": "uncertainty channel"},
    "bxz_channel": {"one_of": "data channel"},
    "bxz_uncertainty": {"one_of": "uncertainty channel"},
    "byy_channel": {"one_of": "data channel"},
    "byy_uncertainty": {"one_of": "uncertainty channel"},
    "byz_channel": {"one_of": "data channel"},
    "byz_uncertainty": {"one_of": "uncertainty channel"},
    "bzz_channel": {"one_of": "data channel"},
    "bzz_uncertainty": {"one_of": "uncertainty channel"},
    "bx_channel": {"one_of": "data channel"},
    "bx_uncertainty": {"one_of": "uncertainty channel"},
    "by_channel": {"one_of": "data channel"},
    "by_uncertainty": {"one_of": "uncertainty channel"},
    "bz_channel": {"one_of": "data channel"},
    "bz_uncertainty": {"one_of": "uncertainty channel"},
}
validations = dict(base_validations, **validations)
app_initializer = {
    "geoh5": str(assets_path() / "FlinFlon.geoh5"),
    "monitoring_directory": str((assets_path() / "Temp").resolve()),
    "forward_only": False,
    "data_object": UUID("{538a7eb1-2218-4bec-98cc-0a759aa0ef4f}"),
    "tmi_channel": UUID("{44822654-b6ae-45b0-8886-2d845f80f422}"),
    "tmi_uncertainty": 10.0,
    "tmi_channel_bool": True,
    "mesh": UUID("{a8f3b369-10bd-4ca8-8bd6-2d2595bddbdf}"),
    "inducing_field_strength": 60000.0,
    "inducing_field_inclination": 79.0,
    "inducing_field_declination": 11.0,
    "reference_model": 0.0,
    "resolution": 50.0,
    "window_center_x": 314600.0,
    "window_center_y": 6072300.0,
    "window_width": 1000.0,
    "window_height": 1500.0,
    "window_azimuth": 0.0,
    "s_norm": 0.0,
    "x_norm": 2.0,
    "y_norm": 2.0,
    "z_norm": 2.0,
    "lower_bound": None,
    "upper_bound": None,
    "starting_model": 1e-4,
    "starting_inclination": 79.0,
    "starting_declination": 11.0,
    "reference_inclination": None,
    "reference_declination": None,
    "max_global_iterations": 25,
    "topography_object": UUID("{ab3c2083-6ea8-4d31-9230-7aad3ec09525}"),
    "topography": UUID("{a603a762-f6cb-4b21-afda-3160e725bf7d}"),
    "z_from_topo": True,
    "receivers_offset_z": 60.0,
    "fix_aspect_ratio": True,
    "colorbar": False,
    "n_cpu": int(multiprocessing.cpu_count() / 2),
}
