#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


from __future__ import annotations

import sys
from os import path

from discretize.utils import mesh_builder_xyz, refine_tree_xyz
from geoh5py.objects import ObjectBase, Octree
from geoh5py.ui_json import monitored_directory_copy

from geoapps.driver_base.driver import BaseDriver
from geoapps.driver_base.utils import treemesh_2_octree
from geoapps.octree_creation.constants import validations
from geoapps.octree_creation.params import OctreeParams


class OctreeDriver(BaseDriver):

    _params_class = OctreeParams
    _validations = validations

    def __init__(self, params: OctreeParams):
        super().__init__(params)
        self.params: OctreeParams = params

    def run(self) -> Octree:
        """
        Create an octree mesh from input values
        """
        octree = self.octree_from_params(self.params)

        if self.params.monitoring_directory is not None and path.exists(
            self.params.monitoring_directory
        ):
            monitored_directory_copy(self.params.monitoring_directory, octree)

        return octree

    @staticmethod
    def octree_from_params(params: OctreeParams):
        print("Setting the mesh extent")
        entity = params.objects

        p_d = [
            [
                params.horizontal_padding,
                params.horizontal_padding,
            ],
            [
                params.horizontal_padding,
                params.horizontal_padding,
            ],
            [params.vertical_padding, params.vertical_padding],
        ]

        treemesh = mesh_builder_xyz(
            entity.vertices,
            [
                params.u_cell_size,
                params.v_cell_size,
                params.w_cell_size,
            ],
            padding_distance=p_d,
            mesh_type="tree",
            depth_core=params.depth_core,
        )

        for label, value in params.free_parameter_dict.items():
            if not isinstance(getattr(params, value["object"]), ObjectBase):
                continue

            print(f"Applying {label} on: {getattr(params, value['object']).name}")

            treemesh = refine_tree_xyz(
                treemesh,
                getattr(params, value["object"]).vertices,
                method=getattr(params, value["type"]),
                octree_levels=getattr(params, value["levels"]),
                max_distance=getattr(params, value["distance"]),
                finalize=False,
            )

        print("Finalizing . . .")
        treemesh.finalize()

        octree = treemesh_2_octree(params.geoh5, treemesh, name=params.ga_group_name)

        return octree


if __name__ == "__main__":
    file = sys.argv[1]
    OctreeDriver.start(file)
