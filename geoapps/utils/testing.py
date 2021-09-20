#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import os
from uuid import UUID

import numpy as np
from discretize.utils import active_from_xyz, mesh_builder_xyz, refine_tree_xyz
from geoh5py.objects import Points, Surface
from geoh5py.workspace import Workspace
from scipy.spatial import Delaunay
from SimPEG import utils

from geoapps.io import InputFile
from geoapps.utils import treemesh_2_octree


class Geoh5Tester:
    """Create temp workspace, copy entities, and setup params class."""

    def __init__(self, workspace, path, name, ui=None, params_class=None):

        self.workspace = workspace
        self.tmp_path = os.path.join(path, name)

        if params_class is not None:
            ws = Workspace(self.tmp_path)
            self.input_file = InputFile()
            self.input_file.data["geoh5"] = None
            self.input_file.data["workspace"] = None
            self.params = params_class.from_input_file(self.input_file, workspace=ws)
            self.ws = self.params.workspace
            self.has_params = True

        else:

            self.ws = Workspace(self.tmp_path)
            self.has_params = False

    def copy_entity(self, uid):
        self.workspace.get_entity(uid)[0].copy(parent=self.ws)

    def set_param(self, param, value):
        if self.has_params:
            try:
                uid = UUID(value)
                self.copy_entity(uid)
                setattr(self.params, param, value)
            except:
                setattr(self.params, param, value)
        else:
            msg = "No params class has been initialized."
            raise (ValueError(msg))

    def make(self):
        if self.has_params:
            return self.ws, self.params
        else:
            return self.ws


def setup_inversion_workspace(work_dir, phys_prop, n_grid_points=20, refinement=(4, 6)):
    project = os.path.join(work_dir, "mag_test.geoh5")
    workspace = Workspace(project)
    # Topography
    xx, yy = np.meshgrid(np.linspace(-200.0, 200.0, 50), np.linspace(-200.0, 200.0, 50))
    b = 100
    A = 50
    zz = A * np.exp(-0.5 * ((xx / b) ** 2.0 + (yy / b) ** 2.0))
    topo = np.c_[utils.mkvc(xx), utils.mkvc(yy), utils.mkvc(zz)]
    triang = Delaunay(topo[:, :2])
    surf = Surface.create(
        workspace, vertices=topo, cells=triang.simplices, name="topography"
    )
    # Observation points
    xr = np.linspace(-100.0, 100.0, n_grid_points)
    yr = np.linspace(-100.0, 100.0, n_grid_points)
    X, Y = np.meshgrid(xr, yr)
    Z = A * np.exp(-0.5 * ((X / b) ** 2.0 + (Y / b) ** 2.0)) + 5.0
    points = Points.create(
        workspace,
        vertices=np.c_[utils.mkvc(X.T), utils.mkvc(Y.T), utils.mkvc(Z.T)],
        name="survey",
    )
    # Create a mesh
    h = 5
    padDist = np.ones((3, 2)) * 100
    mesh = mesh_builder_xyz(
        points.vertices,
        [h] * 3,
        depth_core=100.0,
        padding_distance=padDist,
        mesh_type="TREE",
    )
    mesh = refine_tree_xyz(
        mesh,
        topo,
        method="surface",
        octree_levels=refinement,
        octree_levels_padding=refinement,
        finalize=True,
    )
    octree = treemesh_2_octree(workspace, mesh, name="mesh")
    active = active_from_xyz(mesh, surf.vertices, grid_reference="N")
    # Model
    model = utils.model_builder.addBlock(
        mesh.gridCC,
        np.zeros(mesh.nC),
        np.r_[-20, -20, -20],
        np.r_[20, 20, 20],
        phys_prop,
    )
    model[~active] = np.nan
    octree.add_data({"model": {"values": model[mesh._ubc_order]}})
    octree.copy()  # Keep a copy around for ref
    return workspace
