#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from geoh5py.objects import DrapeModel, Octree

from geoapps.octree_creation.params import OctreeParams
from geoapps.shared_utils.utils import drape_2_tensor, octree_2_treemesh
from geoapps.utils.models import get_drape_model

if TYPE_CHECKING:
    from discretize import TreeMesh
    from geoh5py.workspace import Workspace

    from . import InversionData, InversionTopography


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
        inversion_data: InversionData,
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
        self.mesh: TreeMesh = None
        self.n_cells: int = None
        self.rotation: dict[str, float] = None
        self.permutation: np.ndarray = None
        self.entity: Octree = None
        self._initialize()

    def _initialize(self) -> None:
        """
        Collects mesh data stored in geoh5 workspace into TreeMesh object.

        Handles conversion from geoh5's native octree mesh type to TreeMesh
        type required for SimPEG inversion and stores data needed to restore
        original the octree mesh type.
        """

        if self.params.mesh is None:
            self.build_from_params()
        else:
            self.entity = self.params.mesh.copy(
                parent=self.params.ga_group, copy_children=False
            )

        if getattr(self.entity, "rotation", None) and self.inversion_data.has_tensor:
            msg = "Cannot use tensor components with rotated mesh."
            raise NotImplementedError(msg)

        self.uid = self.entity.uid
        self.n_cells = self.entity.n_cells

        if isinstance(self.entity, Octree):

            if self.entity.rotation:
                origin = self.entity.origin.tolist()
                angle = self.entity.rotation[0]
                self.rotation = {"origin": origin, "angle": angle}

            self.mesh = octree_2_treemesh(self.entity)
            self.permutation = getattr(self.mesh, "_ubc_order")

        if isinstance(self.entity, DrapeModel):
            self.mesh, self.permutation = drape_2_tensor(
                self.entity, return_sorting=True
            )

    def build_from_params(self) -> Octree:
        """Runs geoapps.create.OctreeMesh to create mesh from params."""
        if "2d" in self.params.inversion_type:
            (  # pylint: disable=W0632
                self.entity,
                self.mesh,
                self.permutation,
            ) = get_drape_model(
                self.workspace,
                "Models",
                self.inversion_data._survey.unique_locations,  # pylint: disable=W0212
                [self.params.u_cell_size, self.params.v_cell_size],
                self.params.depth_core,
                [self.params.horizontal_padding] * 2
                + [self.params.vertical_padding, 1],
                self.params.expansion_factor,
                parent=self.params.ga_group,
                return_colocated_mesh=True,
                return_sorting=True,
            )
        else:
            raise NotImplementedError("Must pass a pre-constructed mesh.")
