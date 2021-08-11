#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geoh5py.workspace import Workspace
    from geoapps.io.params import Params
    from geoapps.drivers.components import InversionData
    from discretize import TreeMesh

import numpy as np
from discretize.utils import mesh_builder_xyz, refine_tree_xyz
from geoh5py.workspace import Workspace

from geoapps.io import Params
from geoapps.utils import octree_2_treemesh, rotate_xy


class InversionMesh:
    """
    Retrieve Octree mesh data from workspace and convert to Treemesh.

    Attributes
    ----------

    nC:
        Number of cells in the mesh.
    rotation :
        Rotation of original Octree mesh.
    octree_permutation:
        Permutation vector to restore cell centers or model values to
        origin Octree mesh order.


    Methods
    -------
    original_cc() :
        Returns the cell centers of the original Octree mesh type.

    """

    def __init__(self, workspace: Workspace, params: Params) -> None:
        """
        :param workspace: Workspace object containing mesh data.
        :param params: Params object containing mesh parameters.
        :param window: Center and size defining window for data, topography, etc.

        """
        self.workspace = workspace
        self.params = params
        self.mesh: TreeMesh = None
        self.nC: int = None
        self.rotation: dict[str, float] = None
        self.octree_permutation: np.ndarray = None
        self._initialize()

    def _initialize(self) -> None:
        """
        Collects mesh data stored in geoh5 workspace into TreeMesh object.

        Handles conversion from geoh5's native Octree mesh type to TreeMesh
        type required for SimPEG inversion and stores data needed to restore
        original the Octree mesh type.
        """

        if self.params.mesh_from_params:

            return

        else:

            self.mesh = self.workspace.get_entity(self.params.mesh)[0]
            self.nC = self.mesh.n_cells

            if self.mesh.rotation:
                origin = self.mesh.origin.tolist()
                angle = self.mesh.rotation[0]
                self.rotation = {"origin": origin, "angle": angle}

            self.mesh = octree_2_treemesh(self.mesh)
            self.octree_permutation = self.mesh._ubc_order

    def original_cc(self) -> np.ndarray:
        """Returns the cell centers of the original Octree mesh type."""
        cc = self.mesh.cell_centers
        cc = rotate_xy(cc, self.rotation["origin"], self.rotation["angle"])
        return cc[self.octree_permutation]

    def build_from_params(
        self,
        inversion_data: InversionData,
        inversion_topography: InversionTopography,
    ):

        # from geoapps.create.octree_mesh import OctreeMesh
        # octree_params = TODO
        # OctreeMesh.run(octree_params)

        # topography_locs = inversion_data.set_z_from_topo(inversion_data.locations)
        print("Creating Global TreeMesh")
        mesh = mesh_builder_xyz(
            inversion_data.locations,
            self.params.core_cell_size(),
            padding_distance=self.params.padding_distance(),
            mesh_type="TREE",
            depth_core=self.params.depth_core,
        )

        mesh = refine_tree_xyz(
            mesh,
            inversion_topography.locations,
            method="surface",
            octree_levels=self.params.octree_levels_topo,
            finalize=False,
        )

        mesh = refine_tree_xyz(
            mesh,
            inversion_data.locations,
            method="surface",
            octree_levels=self.params.octree_levels_obs,
            max_distance=self.params.max_distance,
            finalize=True,
        )

        self.mesh = mesh
        self.nC = self.mesh.nC
        self.octree_permutation = self.mesh._ubc_order
