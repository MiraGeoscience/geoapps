#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

valid_parameters = [
    "data_format",
    "data_name",
    "data_channels",
    "out_group",
    "workspace",
    "save_to_geoh5",
    "inversion_type",
    "inversion_style",
    "forward_only",
    "inducing_field_aid",
    "core_cell_size",
    "octree_levels_topo",
    "octree_levels_obs",
    "depth_core",
    "max_distance",
    "padding_distance",
    "chi_factor",
    "max_iterations",
    "max_cg_iterations",
    "max_global_iterations",
    "n_cpu",
    "max_ram",
    "initial_beta_ratio",
    "tol_cg",
    "ignore_values",
    "no_data_value",
    "resolution",
    "window",
    "alphas",
    "reference_model",
    "starting_model",
    "model_norms",
    "data",
    "uncertainty_mode",
    "receivers_offset",
    "topography",
    "result_folder",
    "detrend",
    "data_file",
    "new_uncert",
    "input_mesh",
    "save_to_geoh5",
    "input_mesh_file",
    "inversion_mesh_type",
    "shift_mesh_z0",
    "receivers_offset",
]

required_parameters = ["inversion_type", "core_cell_size"]

valid_parameter_values = {
    "inversion_type": ["gravity", "magnetics", "mvi", "mvic"],
    "inversion_style": ["voxel"],
    "data_format": ["ubc_grav", "ubc_mag", "GA_object"],
    "inversion_mesh_type": ["TREE"],
}

valid_parameter_types = {
    "inversion_type": [str],
    "core_cell_size": [int, float],
    "workpath": [str],
    "inversion_style": [str],
    "forward_only": [bool],
    "result_folder": [str],
    "inducing_field_aid": [int, float],
    "resolution": [int, float],
    "window": [dict],
    "workspace": [str],
    "data_format": [str],
    "data_name": [str],
    "data_channels": [dict],
    "ignore_values": [str],
    "detrend": [dict],
    "data_file": [str],
    "new_uncert": [int, float],
    "input_mesh": [str],
    "save_to_geoh5": [str],
    "inversion_mesh_type": [str],
    "shift_mesh_z0": [int, float],
    "topography": [dict],
    "receivers_offset": [dict],
    "chi_factor": [int, float],
    "model_norms": [int, float],
    "max_iterations": [int],
    "max_cg_iterations": [int],
    "tol_cg": [int, float],
    "max_global_iterations": [int],
}

valid_parameter_shapes = {
    "inducing_field_aid": (3,),
    "new_uncert": (2,),
}

valid_parameter_keys = {
    "window": ["center_x", "center_y", "width", "height", "azimuth", "center", "size"],
    "data_channels": ["tmi"],
    "topography": ["GA_object", "draped", "constant", "file"],
    "receivers_offset": ["constant", "constant_drape", "radar_drape"],
}
