#  Copyright (c) 2022 Mira Geoscience Ltd.
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
    from geoapps.base.params import BaseParams
    from discretize import TreeMesh
    from . import InversionData, InversionTopography

import numpy as np
from geoh5py.objects import PotentialElectrode
from geoh5py.workspace import Workspace

from geoapps.octree_creation.params import OctreeParams
from geoapps.utils import octree_2_treemesh


class InversionMesh:
    """
    Retrieve octree mesh data from workspace and convert to Treemesh.

    Attributes
    ----------

    nC:
        Number of cells in the mesh.
    rotation :
        Rotation of original octree mesh.
    octree_permutation:
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
        self.nC: int = None
        self.rotation: dict[str, float] = None
        self.octree_permutation: np.ndarray = None
        self.entity: Octree = None
        self._initialize()

    def _initialize(self) -> None:
        """
        Collects mesh data stored in geoh5 workspace into TreeMesh object.

        Handles conversion from geoh5's native octree mesh type to TreeMesh
        type required for SimPEG inversion and stores data needed to restore
        original the octree mesh type.
        """

        if self.params.mesh is not None:
            self.entity = self.params.mesh.copy(
                parent=self.params.ga_group, copy_children=False
            )
        else:
            self.build_from_params()

        self.uid = self.entity.uid
        self.nC = self.entity.n_cells

        if self.entity.rotation:
            origin = self.entity.origin.tolist()
            angle = self.entity.rotation[0]
            self.rotation = {"origin": origin, "angle": angle}

        self.mesh = octree_2_treemesh(self.entity)
        self.octree_permutation = self.mesh._ubc_order

    def collect_mesh_params(self, params: BaseParams) -> OctreeParams:
        """Collect mesh params from inversion params set and return octree Params object."""

        mesh_param_names = [
            "u_cell_size",
            "v_cell_size",
            "w_cell_size",
            "depth_core",
            "horizontal_padding",
            "vertical_padding",
            "geoh5",
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
        mesh_params_dict["Refinement A object"] = self.inversion_data.entity.uid
        mesh_params_dict["Refinement A levels"] = params.octree_levels_obs
        mesh_params_dict["Refinement A type"] = "radial"
        mesh_params_dict["Refinement A distance"] = params.max_distance

        mesh_params_dict["Refinement B object"] = self.inversion_topography.entity.uid
        mesh_params_dict["Refinement B levels"] = params.octree_levels_topo
        mesh_params_dict["Refinement B type"] = "surface"
        mesh_params_dict["Refinement B distance"] = params.max_distance

        if isinstance(self.inversion_data.entity, PotentialElectrode):
            mesh_params_dict["Refinement C object"] = (
                self.inversion_data.entity.current_electrodes.uid,
            )
            mesh_params_dict["Refinement C levels"] = params.octree_levels_obs
            mesh_params_dict["Refinement C type"] = "radial"
            mesh_params_dict["Refinement C distance"] = params.max_distance

        mesh_params_dict["objects"] = self.inversion_data.entity.uid
        mesh_params_dict["geoh5"] = self.workspace

        return OctreeParams(**mesh_params_dict, validate=False)

    def build_from_params(self) -> Octree:
        """Runs geoapps.create.OctreeMesh to create mesh from params."""

        from geoapps.octree_creation.application import OctreeMesh

        octree_params = self.collect_mesh_params(self.params)
        self.entity = OctreeMesh.run(octree_params)
        self.entity.parent = self.params.ga_group
