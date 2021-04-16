#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import numpy as np

required_parameters = [
    "inversion_type",
    "workspace",
    "out_group",
    "data",
    "mesh",
    "topography",
]


defaults = {
    "inversion_type": None,
    "workspace": None,
    "out_group": None,
    "data": None,
    "mesh": None,
    "topography": None,
    "inversion_style": "voxel",
    "forward_only": False,
    "inducing_field_aid": None,
    "core_cell_size": None,
    "octree_levels_topo": [0, 1],
    "octree_levels_obs": [5, 5],
    "octree_levels_padding": [2, 2],
    "depth_core": None,
    "max_distance": np.inf,
    "padding_distance": [[0, 0]] * 3,
    "chi_factor": 1,
    "max_iterations": 10,
    "max_cg_iterations": 30,
    "max_global_iterations": 100,
    "n_cpu": None,
    "max_ram": 2,
    "initial_beta": None,
    "initial_beta_ratio": 1e2,
    "tol_cg": 1e-4,
    "ignore_values": None,
    "no_data_value": 0,
    "resolution": 0,
    "window": None,
    "alphas": [1] * 12,
    "reference_model": None,
    "reference_inclination": None,
    "reference_declination": None,
    "starting_model": None,
    "starting_inclination": None,
    "starting_declination": None,
    "model_norms": [2] * 4,
    "detrend": None,
    "new_uncert": None,
    "output_geoh5": None,
    "receivers_offset": None,
    "gradient_type": "total",
    "lower_bound": -np.inf,
    "upper_bound": np.inf,
    "max_chunk_size": 128,
    "chunk_by_rows": False,
    "output_tile_files": False,
    "parallelized": True,
}


validations = {
    "inversion_type": {
        "values": ["gravity", "magnetics", "mvi", "mvic"],
        "types": [str],
    },
    "data": {
        "types": [dict],
        "reqs": [("workspace",)],
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
    "inversion_style": {
        "values": ["voxel"],
        "types": [str],
    },
    "forward_only": {"types": [bool], "reqs": [(True, "reference_model")]},
    "inducing_field_aid": {
        "types": [int, float],
        "shapes": (3,),
    },
    "core_cell_size": {
        "types": [int, float],
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
        "types": [str, int, float],
    },
    "reference_inclination": {
        "types": [str, int, float],
    },
    "reference_declination": {
        "types": [str, int, float],
    },
    "starting_model": {
        "types": [str, int, float],
    },
    "starting_inclination": {
        "types": [str, int, float],
    },
    "starting_declination": {
        "types": [str, int, float],
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
        "draped": {
            "types": [int, float],
        },
        "file": {
            "types": [str],
        },
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
    "new_uncert": {"types": [int, float], "shapes": (2,)},
    "mesh": {
        "types": [str],
    },
    "output_geoh5": {
        "types": [str],
    },
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
}
