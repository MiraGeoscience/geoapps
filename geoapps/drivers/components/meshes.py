#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from typing import List

from geoh5py.workspace import Workspace

from geoapps.io import Params
from geoapps.utils import octree_2_treemesh, rotate_xy


class InversionMesh:
    """A class for handling conversion of Octree mesh type to TreeMesh type.

    :param params: Params object containing "mesh" attribute that stores UUID
        addressing an Octree mesh within the workspace.
    :param workspace: Workspace object containing mesh data.
    :param window: Data defining the limits for a restricted size inversion,
        and possibly rotation information.
    :param mesh: TreeMesh object.
    :param nC: Number of cells of mesh.
    :param rotation: Rotation of original Octree mesh.
    :octree_permutation: Permutation vector to restore cell centers or
        model values to origin Octree mesh order.
    """

    def __init__(
        self, params: Params, workspace: Workspace, window: List[float] = None
    ) -> None:

        self.params = params
        self.workspace = workspace
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
            else:
                if window is not None:
                    origin = self.window["center"]
                    angle = self.window["azimuth"]

            self.mesh = octree_2_treemesh(self.mesh)
            self.rotation = {"origin": origin, "angle": angle}
            self.octree_permutation = self.mesh._ubc_order

    def original_cc(self):
        """ Returns the cell centers of the original Octree mesh type. """
        cc = self.mesh.cell_centers
        cc = rotate_xy(cc, self.rotation["origin"], self.rotation["angle"])
        return cc[self.octree_permutation]
