#  Copyright (c) 2022 Mira Geoscience Ltd.
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
from geoh5py.objects import (
    CurrentElectrode,
    MTReceivers,
    Points,
    PotentialElectrode,
    Surface,
    TipperReceivers,
)
from geoh5py.ui_json import InputFile
from geoh5py.workspace import Workspace
from scipy.spatial import Delaunay
from SimPEG import utils

from geoapps.utils import treemesh_2_octree


class Geoh5Tester:
    """Create temp workspace, copy entities, and setup params class."""

    def __init__(self, geoh5, path, name, ui=None, params_class=None):

        self.geoh5 = geoh5
        self.tmp_path = os.path.join(path, name)

        if params_class is not None:
            self.ws = Workspace(self.tmp_path)
            self.params = params_class(validate=False, geoh5=self.ws)
            self.has_params = True

        else:
            self.ws = Workspace(self.tmp_path)
            self.has_params = False

    def copy_entity(self, uid):
        entity = self.ws.get_entity(uid)
        if not entity or entity[0] is None:
            return self.geoh5.get_entity(uid)[0].copy(parent=self.ws)
        return entity[0]

    def set_param(self, param, value):
        if self.has_params:
            try:
                uid = UUID(value)
                entity = self.copy_entity(uid)
                setattr(self.params, param, entity)
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


def setup_inversion_workspace(
    work_dir,
    background=None,
    anomaly=None,
    n_electrodes=20,
    n_lines=5,
    refinement=(4, 6),
    inversion_type="other",
    flatten=False,
):

    project = os.path.join(work_dir, "inversion_test.geoh5")
    geoh5 = Workspace(project)
    # Topography
    xx, yy = np.meshgrid(np.linspace(-200.0, 200.0, 50), np.linspace(-200.0, 200.0, 50))
    b = 100
    A = 50
    if flatten:
        zz = np.zeros_like(xx)
    else:
        zz = A * np.exp(-0.5 * ((xx / b) ** 2.0 + (yy / b) ** 2.0))
    topo = np.c_[utils.mkvc(xx), utils.mkvc(yy), utils.mkvc(zz)]
    triang = Delaunay(topo[:, :2])
    surf = Surface.create(
        geoh5, vertices=topo, cells=triang.simplices, name="topography"
    )
    # Observation points
    n_electrodes = (
        4 if (inversion_type == "dcip") & (n_electrodes < 4) else n_electrodes
    )
    xr = np.linspace(-100.0, 100.0, n_electrodes)
    yr = np.linspace(-100.0, 100.0, n_lines)
    X, Y = np.meshgrid(xr, yr)
    if flatten:
        Z = np.zeros_like(X)
    else:
        Z = A * np.exp(-0.5 * ((X / b) ** 2.0 + (Y / b) ** 2.0)) + 5.0

    vertices = np.c_[utils.mkvc(X.T), utils.mkvc(Y.T), utils.mkvc(Z.T)]

    if inversion_type == "dcip":

        parts = np.repeat(np.arange(n_lines), n_electrodes).astype("int32")
        currents = CurrentElectrode.create(
            geoh5, name="survey (currents)", vertices=vertices, parts=parts
        )
        currents.add_default_ab_cell_id()
        potentials = PotentialElectrode.create(geoh5, name="survey", vertices=vertices)
        potentials.current_electrodes = currents
        currents.potential_electrodes = potentials

        N = 6
        dipoles = []
        current_id = []
        potentials_parts = []
        for val in currents.ab_cell_id.values:  # For each source dipole
            cell_id = int(currents.ab_map[val]) - 1  # Python 0 indexing
            line = currents.parts[currents.cells[cell_id, 0]]
            for m_n in range(N):
                dipole_ids = (currents.cells[cell_id, :] + 2 + m_n).astype(
                    "uint32"
                )  # Skip two poles

                # Shorten the array as we get to the end of the line
                if any(dipole_ids > (potentials.n_vertices - 1)) or any(
                    currents.parts[dipole_ids] != line
                ):
                    continue
                potentials_parts += [line] * len(dipole_ids)
                dipoles += [dipole_ids]  # Save the receiver id
                current_id += [val]  # Save the source id

        potentials.cells = np.vstack(dipoles).astype("uint32")
        potentials.ab_cell_id = np.asarray(current_id).astype("int32")

    elif inversion_type == "magnetotellurics":
        components = [
            "Zxx (real)",
            "Zxx (imag)",
            "Zxy (real)",
            "Zxy (imag)",
            "Zyx (real)",
            "Zyx (imag)",
            "Zyy (real)",
            "Zyy (imag)",
        ]
        mt_receivers = MTReceivers.create(
            geoh5,
            vertices=vertices,
            name="survey",
            components=components,
            channels=[10.0, 100.0, 1000.0],
        )

    elif inversion_type == "tipper":
        components = [
            "Txz (real)",
            "Txz (imag)",
            "Tyz (real)",
            "Tyz (imag)",
        ]
        tipper_receivers = TipperReceivers.create(
            geoh5,
            vertices=vertices,
            name="survey",
            components=components,
            channels=[10.0, 100.0, 1000.0],
        )

    else:

        points = Points.create(
            geoh5,
            vertices=vertices,
            name="survey",
        )

    # Create a mesh
    h = 5
    padDist = np.ones((3, 2)) * 100
    mesh = mesh_builder_xyz(
        vertices - h / 2.0,
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
    octree = treemesh_2_octree(geoh5, mesh, name="mesh")
    active = active_from_xyz(mesh, surf.vertices, grid_reference="N")
    # Model
    if flatten:
        model = utils.model_builder.addBlock(
            mesh.gridCC,
            background * np.ones(mesh.nC),
            np.r_[-20, -20, -30],
            np.r_[20, 20, -70],
            anomaly,
        )
    else:
        model = utils.model_builder.addBlock(
            mesh.gridCC,
            background * np.ones(mesh.nC),
            np.r_[-20, -20, -20],
            np.r_[20, 20, 25],
            anomaly,
        )
    model[~active] = np.nan
    octree.add_data({"model": {"values": model[mesh._ubc_order]}})
    # octree.add_data({"active": {"values": active.astype(int)[mesh._ubc_order]}})
    octree.copy()  # Keep a copy around for ref
    return geoh5
