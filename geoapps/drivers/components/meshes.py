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
    from geoh5py.objects import Octree
    from geoapps.io.params import Params
    from discretize import TreeMesh
    from . import InversionData, InversionTopography

import numpy as np
from geoh5py.workspace import Workspace

from geoapps.io import Params
from geoapps.io.Octree import OctreeParams
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

    def __init__(
        self,
        workspace: Workspace,
        params: Params,
        inversion_data: InversionData,
        inversion_topography: InversionTopography,
    ) -> None:
        """
        :param workspace: Workspace object containing mesh data.
        :param params: Params object containing mesh parameters.
        :param window: Center and size defining window for data, topography, etc.
        :param

        """
        self.workspace = workspace
        self.params = params
        self.inversion_data = inversion_data
        self.inversion_topography = inversion_topography
        self.mesh: TreeMesh = None
        self.nC: int = None
        self.rotation: dict[str, float] = None
        self.octree_permutation: np.ndarray = None
        self.entity: Octree = None
        self._initialize()

    def _initialize(self) -> None:
        """
        Collects mesh data stored in geoh5 workspace into TreeMesh object.

        Handles conversion from geoh5's native Octree mesh type to TreeMesh
        type required for SimPEG inversion and stores data needed to restore
        original the Octree mesh type.
        """

        if self.params.mesh_from_params:
            self.build_from_params()
            self.entity = self.workspace.get_entity("Octree_Mesh")[0]
            self.entity.parent = self.params.out_group
        else:
            orig_octree = self.workspace.get_entity(self.params.mesh)[0]

            self.entity = orig_octree.copy(
                parent=self.params.out_group, copy_children=False
            )

        self.uid = self.entity.uid
        self.nC = self.entity.n_cells

        if self.entity.rotation:
            origin = self.entity.origin.tolist()
            angle = self.entity.rotation[0]
            self.rotation = {"origin": origin, "angle": angle}

        self.mesh = octree_2_treemesh(self.entity)
        self.octree_permutation = self.mesh._ubc_order

    def original_cc(self) -> np.ndarray:
        """Returns the cell centers of the original Octree mesh type."""
        cc = self.mesh.cell_centers
        if self.rotation is not None:
            cc = rotate_xy(cc, self.rotation["origin"], self.rotation["angle"])
        return cc[self.octree_permutation]

    def collect_mesh_params(self, params: Params) -> OctreeParams:
        """Collect mesh params from inversion params set and return Octree Params object."""

        mesh_param_names = [
            "u_cell_size",
            "v_cell_size",
            "w_cell_size",
            "depth_core",
            "horizontal_padding",
            "vertical_padding",
            "workspace",
        ]

        mesh_params_dict = params.to_dict(ui_json_format=False)
        for k in mesh_param_names:
            if (k not in mesh_params_dict.keys()) or (mesh_params_dict[k] is None):
                msg = f"Cannot create OctreeParams from {type(params)} instance. "
                msg += f"Missing param: {k}."
                raise ValueError(msg)

        mesh_params_dict = {
            k: v for k, v in mesh_params_dict.items() if k in mesh_param_names
        }
        mesh_params_dict["Refinement A"] = {
            "object": self.inversion_data.entity.uid,
            "levels": params.octree_levels_obs,
            "type": "radial",
            "distance": params.max_distance,
        }
        mesh_params_dict["Refinement B"] = {
            "object": self.inversion_topography.entity.uid,
            "levels": params.octree_levels_topo,
            "type": "surface",
            "distance": params.max_distance,
        }
        mesh_params_dict["objects"] = self.workspace.get_entity("Data")[0].uid

        return OctreeParams(**mesh_params_dict)

    def build_from_params(self) -> Octree:
        """Runs geoapps.create.OctreeMesh to create mesh from params."""

        from geoapps.create.octree_mesh import OctreeMesh

        octree_params = self.collect_mesh_params(self.params)
        octree_mesh = OctreeMesh.run(octree_params)

        self.uid = octree_mesh.uid
        self.mesh = octree_2_treemesh(octree_mesh)

        self.nC = self.mesh.nC
        self.octree_permutation = self.mesh._ubc_order
