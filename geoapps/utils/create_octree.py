#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import sys

from discretize.utils import meshutils
from geoh5py.workspace import Workspace

from geoapps.utils.utils import load_json_params, treemesh_2_octree


def create_octree(**kwargs):
    """
    Create an octree mesh from input values
    """
    workspace = Workspace(kwargs["workspace_geoh5"])

    obj = workspace.get_entity(kwargs["objects"])

    if not any(obj):
        return

    p_d = [
        [kwargs["horizontal_padding"], kwargs["horizontal_padding"]],
        [kwargs["horizontal_padding"], kwargs["horizontal_padding"]],
        [kwargs["vertical_padding"], kwargs["vertical_padding"]],
    ]

    print("Setting the mesh extent")
    treemesh = meshutils.mesh_builder_xyz(
        obj[0].vertices,
        [kwargs["u_cell_size"], kwargs["v_cell_size"], kwargs["w_cell_size"]],
        padding_distance=p_d,
        mesh_type="tree",
        depth_core=kwargs["depth_core"],
    )

    labels = ["A", "B"]
    for label in labels:
        print(f"Applying refinement {label}")
        entity = workspace.get_entity(kwargs[f"refinement_{label}"])
        if any(entity):
            treemesh = meshutils.refine_tree_xyz(
                treemesh,
                entity[0].vertices,
                method=kwargs[f"method_{label}"],
                octree_levels=[
                    kwargs[f"octree_{label}1"],
                    kwargs[f"octree_{label}2"],
                    kwargs[f"octree_{label}3"],
                ],
                max_distance=kwargs[f"max_distance_{label}"],
                finalize=False,
            )

    print("Finalizing...")
    treemesh.finalize()
    octree = treemesh_2_octree(workspace, treemesh, name=kwargs[f"ga_group_name"])
    print(f"Octree mesh '{octree.name}' completed and exported to {workspace.h5file}")
    return octree


if __name__ == "__main__":

    input_params = load_json_params(sys.argv[1])
    create_octree(**input_params)
