# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024 Mira Geoscience Ltd.                                     '
#                                                                              '
#  This file is part of geoapps.                                               '
#                                                                              '
#  geoapps is distributed under the terms and conditions of the MIT License    '
#  (see LICENSE file at the root of this source code package).                 '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from geoh5py.objects import DrapeModel, Octree
from octree_creation_app.params import OctreeParams

from geoapps.shared_utils.utils import drape_2_tensor, octree_2_treemesh

if TYPE_CHECKING:
    from geoh5py.workspace import Workspace

    from . import InversionData, InversionTopography

from discretize import TensorMesh, TreeMesh


class InversionMesh:
    """
    Retrieve octree mesh data from workspace and convert to Treemesh.

    Attributes
    ----------

    n_cells:
        Number of cells in the mesh.
    rotation :
        Rotation of original octree mesh.
    permutation:
        Permutation vector to restore cell centers or model values to
        origin octree mesh order.

    """

    def __init__(
        self,
        workspace: Workspace,
        params: OctreeParams,
        inversion_data: InversionData | None,
        inversion_topography: InversionTopography,
    ) -> None:
        """
        :param workspace: Workspace object containing mesh data.
        :param params: Params object containing mesh parameters.
        :param window: Center and size defining window for data, topography, etc.

        """
        self.workspace = workspace
        self.params = params
        self.inversion_data = inversion_data
        self.inversion_topography = inversion_topography
        self._mesh: TreeMesh | TensorMesh | None = None
        self.n_cells: int | None = None
        self.rotation: dict[str, float] | None = None
        self._permutation: np.ndarray | None = None
        self.entity: Octree | DrapeModel | None = None
        self._initialize()

    def _initialize(self) -> None:
        """
        Collects mesh data stored in geoh5 workspace into TreeMesh object.

        Handles conversion from geoh5's native octree mesh type to TreeMesh
        type required for SimPEG inversion and stores data needed to restore
        original the octree mesh type.
        """

        if self.params.mesh is None:
            raise ValueError("Must pass pre-constructed mesh.")
        else:
            self.entity = self.params.mesh.copy(
                parent=self.params.out_group, copy_children=False
            )
            self.params.mesh = self.entity

        if (
            getattr(self.entity, "rotation", None)
            and self.inversion_data is not None
            and self.inversion_data.has_tensor
        ):
            msg = "Cannot use tensor components with rotated mesh."
            raise NotImplementedError(msg)

        self.uid = self.entity.uid
        self.n_cells = self.entity.n_cells

    @property
    def mesh(self) -> TreeMesh | TensorMesh:
        """"""
        if self._mesh is None:
            if isinstance(self.entity, Octree):
                if self.entity.rotation:
                    origin = self.entity.origin.tolist()
                    angle = self.entity.rotation[0]
                    self.rotation = {"origin": origin, "angle": angle}

                self._mesh = octree_2_treemesh(self.entity)
                self._permutation = getattr(self.mesh, "_ubc_order")

            if isinstance(self.entity, DrapeModel) and self._mesh is None:
                self._mesh, self._permutation = drape_2_tensor(
                    self.entity, return_sorting=True
                )

        return self._mesh

    @property
    def permutation(self) -> np.ndarray:
        """Permutation vector between discretize and geoh5py ordering."""
        if self.mesh is None:
            raise ValueError("A 'mesh' must be assigned before accessing permutation.")

        return self._permutation
