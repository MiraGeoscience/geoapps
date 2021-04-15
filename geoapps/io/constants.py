#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


required_parameters = ["inversion_type"]

validations = {
    "inversion_type": {
        "values": ["gravity", "magnetics", "mvi", "mvic"],
        "types": [str],
    },
    "core_cell_size": {
        "types": [int, float],
    },
    "data": {
        "types": [dict],
        "type": {
            "values": ["GA_object", "ubc_grav", "ubc_mag"],
            "types": [str],
            "reqs": [
                ("GA_object", "workspace"),
                ("ubc_grav", "data_file"),
                ("ubc_mag", "data_file"),
            ],
        },
        "name": {
            "types": [str],
        },
        "channels": {
            "types": [dict],
            "tmi": {
                "types": [dict],
                "name": {
                    "types": [str],
                },
                "uncertainties": {
                    "types": [int, float],
                    "shapes": (2,),
                },
                "offsets": {
                    "types": [int, float],
                    "shapes": (3,),
                },
            },
        },
    },
    "out_group": {
        "types": [str],
    },
    "workspace": {
        "types": [str],
    },
    "save_to_geoh5": {
        "types": [str],
    },
    "inversion_style": {
        "values": ["voxel"],
        "types": [str],
    },
    "forward_only": {"types": [bool], "reqs": [(True, "reference_model")]},
    "inducing_field_aid": {
        "types": [int, float],
        "shapes": (3,),
    },
    "octree_levels_topo": {
        "types": [int, float],
    },
    "octree_levels_obs": {
        "types": [int, float],
    },
    "octree_levels_padding": {
        "types": [int, float],
    },
    "depth_core": {
        "types": [dict],
        "value": {
            "types": [int, float],
        },
    },
    "max_distance": {
        "types": [int, float],
    },
    "padding_distance": {
        "types": [int, float],
        "shapes": (3, 2),
    },
    "chi_factor": {
        "types": [int, float],
    },
    "max_iterations": {
        "types": [int, float],
    },
    "max_cg_iterations": {
        "types": [int, float],
    },
    "max_global_iterations": {
        "types": [int, float],
    },
    "n_cpu": {
        "types": [int, float],
    },
    "max_ram": {
        "types": [int, float],
    },
    "initial_beta": {
        "types": [int, float],
    },
    "initial_beta_ratio": {
        "types": [float],
    },
    "tol_cg": {
        "types": [int, float],
    },
    "ignore_values": {
        "types": [str],
    },
    "no_data_value": {
        "types": [int, float],
    },
    "resolution": {
        "types": [int, float],
    },
    "window": {
        "types": [dict],
        "center_x": {
            "types": [int, float],
        },
        "center_y": {
            "types": [int, float],
        },
        "width": {
            "types": [int, float],
        },
        "height": {
            "types": [int, float],
        },
        "azimuth": {
            "types": [int, float],
        },
        "center": {
            "types": [int, float],
        },
        "size": {
            "types": [int, float],
        },
    },
    "alphas": {
        "types": [int, float],
    },
    "reference_model": {
        "types": [dict],
        "value": {
            "types": [int, float],
        },
        "model": {"types": [str]},
        "none": {},
    },
    "starting_model": {
        "types": [dict],
        "value": {
            "types": [int, float],
        },
        "model": {
            "types": [str],
        },
    },
    "model_norms": {
        "types": [int, float],
    },
    "topography": {
        "types": [dict],
        "GA_object": {
            "types": [dict],
            "name": {
                "types": [str],
            },
            "data": {
                "types": [str],
            },
        },
        "constant": {
            "types": [int, float],
        },
        "drapped": {
            "types": [int, float],
        },
        "file": {
            "types": [str],
        },
    },
    "result_folder": {
        "types": [str],
    },
    "detrend": {
        "types": [dict],
        "all": {
            "types": [int, float],
        },
        "corners": {
            "types": [int, float],
        },
    },
    "data_file": {
        "types": [str],
    },
    "new_uncert": {"types": [int, float], "shapes": (2,)},
    "input_mesh": {"types": [str], "reqs": [("save_to_geoh5",), ("input_mesh_file",)]},
    "save_to_geoh5": {
        "types": [str],
        "reqs": [
            ("out_group",),
        ],
    },
    "input_mesh_file": {
        "types": [str],
    },
    "inversion_mesh_type": {"values": ["TREE"], "types": [str]},
    "shift_mesh_z0": {"types": [int, float]},
    "receivers_offset": {
        "types": [dict],
        "constant": {
            "types": [int, float],
        },
        "constant_drape": {
            "types": [int, float],
        },
        "radar_drape": {
            "types": [int, float, str],
            "shapes": (4,),
        },
    },
    "gradient_type": {"values": ["total", "components"], "types": [str]},
    "lower_bound": {
        "types": [int, float],
    },
    "upper_bound": {
        "types": [int, float],
    },
    "max_chunk_size": {
        "types": [int, float],
    },
    "chunk_by_rows": {
        "types": [bool],
    },
    "output_tile_files": {
        "types": [bool],
    },
    "parallelized": {
        "types": [bool],
    },
    "uncertainty_mode": {"types": [str]},
}
