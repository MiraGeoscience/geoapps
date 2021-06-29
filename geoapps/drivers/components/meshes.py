#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from typing import List

import numpy as np
from geoh5py.workspace import Workspace

from geoapps.io import Params
from geoapps.utils import octree_2_treemesh, rotate_xy


class InversionMesh:
    """
    Retrieve Octree mesh data from workspace and convert to Treemesh.

    Methods
    -------
    original_cc: Returns the cell centers of the original Octree mesh type.

    """

    def __init__(
        self, workspace: Workspace, params: Params, window: List[float] = None
    ) -> None:
        """
        :param workspace: Workspace object containing mesh data.
        :param params: Params object containing mesh parameters.
        :param window: Center and size defining window for data, topography, etc.
        :param mesh: TreeMesh object.
        :param nC: Number of cells of mesh.
        :param rotation: Rotation of original Octree mesh.
        :param: octree_permutation: Permutation vector to restore cell centers or
            model values to origin Octree mesh order.
        """
        self.workspace = workspace
        self.params = params
        self.window = window
        self.mesh = None
        self.nC = None
        self.rotation = None
        self.octree_permutation = None
        self._initialize()

    def _initialize(self) -> None:
        """
        Collects mesh data stored in geoh5 workspace into TreeMesh object.

        Handles conversion from geoh5's native Octree mesh type to TreeMesh
        type required for SimPEG inversion and stores data needed to restore
        original the Octree mesh type.
        """

        if self.params.mesh_from_params:

            # TODO implement the mesh_from_params option
            msg = "Cannot currently mesh from parameters. Must provide mesh object."
            raise NotImplementedError(msg)

        else:

            self.mesh = self.workspace.get_entity(self.params.mesh)[0]
            self.nC = self.mesh.n_cells

            if self.mesh.rotation:
                origin = self.mesh.origin.tolist()
                angle = self.mesh.rotation[0]
                self.rotation = {"origin": origin, "angle": angle}
            else:
                if self.window is not None:
                    if "azimuth" in self.window.keys():
                        origin = self.window["center"]
                        angle = self.window["azimuth"]
                        self.rotation = {"origin": origin, "angle": angle}

            self.mesh = octree_2_treemesh(self.mesh)
            self.octree_permutation = self.mesh._ubc_order

    def original_cc(self) -> np.ndarray:
        """ Returns the cell centers of the original Octree mesh type. """
        cc = self.mesh.cell_centers
        cc = rotate_xy(cc, self.rotation["origin"], self.rotation["angle"])
        return cc[self.octree_permutation]
