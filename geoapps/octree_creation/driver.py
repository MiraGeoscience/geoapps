#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geoh5py.objects import Octree

import os

from discretize.utils import mesh_builder_xyz, refine_tree_xyz
from geoh5py.objects import ObjectBase

from geoapps.base.application import BaseApplication
from geoapps.utils.utils import treemesh_2_octree

from . import OctreeParams


class OctreeDriver:
    def __init__(self, params: OctreeParams):
        self.params: OctreeParams = params

    def run(self) -> Octree:
        """
        Create an octree mesh from input values
        """
        entity = self.params.objects

        p_d = [
            [
                self.params.horizontal_padding,
                self.params.horizontal_padding,
            ],
            [
                self.params.horizontal_padding,
                self.params.horizontal_padding,
            ],
            [self.params.vertical_padding, self.params.vertical_padding],
        ]

        print("Setting the mesh extent")
        treemesh = mesh_builder_xyz(
            entity.vertices,
            [
                self.params.u_cell_size,
                self.params.v_cell_size,
                self.params.w_cell_size,
            ],
            padding_distance=p_d,
            mesh_type="tree",
            depth_core=self.params.depth_core,
        )

        for label, value in self.params.free_parameter_dict.items():
            if not isinstance(getattr(self.params, value["object"]), ObjectBase):
                continue

            print(f"Applying {label} on: {getattr(self.params, value['object']).name}")

            treemesh = refine_tree_xyz(
                treemesh,
                getattr(self.params, value["object"]).vertices,
                method=getattr(self.params, value["type"]),
                octree_levels=getattr(self.params, value["levels"]),
                max_distance=getattr(self.params, value["distance"]),
                finalize=False,
            )

        print("Finalizing...")
        treemesh.finalize()

        print("Writing to file ")
        octree = treemesh_2_octree(
            self.params.geoh5, treemesh, name=self.params.ga_group_name
        )

        if self.params.monitoring_directory is not None and os.path.exists(
            self.params.monitoring_directory
        ):
            BaseApplication.live_link_output(self.params.monitoring_directory, octree)

        print(
            f"Octree mesh '{octree.name}' completed and exported to {os.path.abspath(self.params.geoh5.h5file)}"
        )

        return octree
