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

from __future__ import annotations

import os
import warnings
from uuid import UUID

import numpy as np
from discretize.utils import mesh_builder_xyz, refine_tree_xyz
from geoh5py.objects import (
    CurrentElectrode,
    MTReceivers,
    Points,
    PotentialElectrode,
    Surface,
    TipperBaseStations,
    TipperReceivers,
)
from geoh5py.workspace import Workspace
from scipy.spatial import Delaunay
from SimPEG import utils

from geoapps.driver_base.utils import active_from_xyz, treemesh_2_octree
from geoapps.utils.models import get_drape_model
from geoapps.utils.surveys import survey_lines


class Geoh5Tester:
    """Create temp workspace, copy entities, and setup params class."""

    def __init__(self, geoh5, path, name, params_class=None):

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
            except (AttributeError, ValueError):
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
    cell_size=(5.0, 5.0, 5.0),
    n_electrodes=20,
    n_lines=5,
    refinement=(4, 6),
    padding_distance=100,
    drape_height=5.0,
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
    topography = Surface.create(
        geoh5,
        vertices=topo,
        cells=triang.simplices,  # pylint: disable=E1101
        name="topography",
    )
    # Observation points
    n_electrodes = (
        4
        if (inversion_type in ["dcip", "dcip_2d"]) & (n_electrodes < 4)
        else n_electrodes
    )
    xr = np.linspace(-100.0, 100.0, n_electrodes)
    yr = np.linspace(-100.0, 100.0, n_lines)
    X, Y = np.meshgrid(xr, yr)
    if flatten:
        Z = np.zeros_like(X)
    else:
        Z = A * np.exp(-0.5 * ((X / b) ** 2.0 + (Y / b) ** 2.0)) + drape_height

    vertices = np.c_[utils.mkvc(X.T), utils.mkvc(Y.T), utils.mkvc(Z.T)]

    if inversion_type in ["dcip", "dcip_2d"]:

        ab_vertices = np.c_[
            X[:, :-2].flatten(), Y[:, :-2].flatten(), Z[:, :-2].flatten()
        ]
        mn_vertices = np.c_[X[:, 2:].flatten(), Y[:, 2:].flatten(), Z[:, 2:].flatten()]

        parts = np.repeat(np.arange(n_lines), n_electrodes - 2).astype("int32")
        currents = CurrentElectrode.create(
            geoh5, name="survey (currents)", vertices=ab_vertices, parts=parts
        )
        currents.add_default_ab_cell_id()

        N = 6
        dipoles = []
        current_id = []
        for val in currents.ab_cell_id.values:  # For each source dipole
            cell_id = int(currents.ab_map[val]) - 1  # Python 0 indexing
            line = currents.parts[currents.cells[cell_id, 0]]
            for m_n in range(N):
                dipole_ids = (currents.cells[cell_id, :] + m_n).astype(
                    "uint32"
                )  # Skip two poles

                # Shorten the array as we get to the end of the line
                if any(dipole_ids > (len(mn_vertices) - 1)) or any(
                    currents.parts[dipole_ids] != line
                ):
                    continue

                dipoles += [dipole_ids]  # Save the receiver id
                current_id += [val]  # Save the source id

        survey = PotentialElectrode.create(
            geoh5,
            name="survey",
            vertices=mn_vertices,
            cells=np.vstack(dipoles).astype("uint32"),
        )
        survey.current_electrodes = currents
        survey.ab_cell_id = np.asarray(current_id).astype("int32")
        currents.potential_electrodes = survey

    elif inversion_type == "magnetotellurics":
        survey = MTReceivers.create(
            geoh5,
            vertices=vertices,
            name="survey",
            components=[
                "Zxx (real)",
                "Zxx (imag)",
                "Zxy (real)",
                "Zxy (imag)",
                "Zyx (real)",
                "Zyx (imag)",
                "Zyy (real)",
                "Zyy (imag)",
            ],
            channels=[10.0, 100.0, 1000.0],
        )

    elif inversion_type == "tipper":
        survey = TipperReceivers.create(
            geoh5,
            vertices=vertices,
            name="survey",
            components=[
                "Txz (real)",
                "Txz (imag)",
                "Tyz (real)",
                "Tyz (imag)",
            ],
        )
        survey.base_stations = TipperBaseStations.create(geoh5)
        survey.channels = [10.0, 100.0, 1000.0]
        dist = np.linalg.norm(
            survey.vertices[survey.cells[:, 0], :]
            - survey.vertices[survey.cells[:, 1], :],
            axis=1,
        )
        # survey.cells = survey.cells[dist < 100.0, :]
        survey.remove_cells(np.where(dist > (200.0 / (n_electrodes - 1)))[0])

    else:
        survey = Points.create(
            geoh5,
            vertices=vertices,
            name="survey",
        )

    # Create a mesh

    if "2d" in inversion_type:

        locs = np.unique(np.vstack([ab_vertices, mn_vertices]), axis=0)
        lines = survey_lines(locs, [-100, -100])

        entity, mesh, permutation = get_drape_model(  # pylint: disable=W0632
            geoh5,
            "Models",
            locs[lines == 2],
            [cell_size[0], cell_size[2]],
            100.0,
            [padding_distance] * 2 + [padding_distance] * 2,
            1.1,
            parent=None,
            return_colocated_mesh=True,
            return_sorting=True,
        )
        active = active_from_xyz(
            entity, topography.vertices, grid_reference="cell_centers"
        )

    else:
        padDist = np.ones((3, 2)) * padding_distance
        mesh = mesh_builder_xyz(
            vertices - np.r_[cell_size] / 2.0,
            cell_size,
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
        entity = treemesh_2_octree(geoh5, mesh, name="mesh")
        active = active_from_xyz(mesh, topography.vertices, grid_reference="top_nodes")
        permutation = mesh._ubc_order  # pylint: disable=W0212

    # Model
    if flatten:

        if "2d" in inversion_type:
            p0 = np.r_[80, -30]
            p1 = np.r_[120, -70]
        else:
            p0 = np.r_[-20, -20, -30]
            p1 = np.r_[20, 20, -70]

        model = utils.model_builder.addBlock(
            mesh.gridCC,
            background * np.ones(mesh.nC),
            p0,
            p1,
            anomaly,
        )
    else:

        if "2d" in inversion_type:
            p0 = np.r_[80, -20]
            p1 = np.r_[120, 25]
        else:
            p0 = np.r_[-20, -20, -20]
            p1 = np.r_[20, 20, 25]

        model = utils.model_builder.addBlock(
            mesh.gridCC,
            background * np.ones(mesh.nC),
            p0,
            p1,
            anomaly,
        )

    model[
        ~(active[np.argsort(permutation)] if "2d" in inversion_type else active)
    ] = np.nan
    model = entity.add_data({"model": {"values": model[permutation]}})
    return geoh5, entity, model, survey, topography


def check_target(output: dict, target: dict, tolerance=0.1):
    """
    Check inversion output metrics against hard-valued target.
    :param output: Dictionary containing keys for 'data', 'phi_d' and 'phi_m'.
    :param target: Dictionary containing keys for 'data_norm', 'phi_d' and 'phi_m'.\
    :param tolerance: Tolerance between output and target measured as: |a-b|/b
    """
    if any(np.isnan(output["data"])):
        warnings.warn(
            "Skipping data norm comparison due to nan (used to bypass lone faulty test run in GH actions)."
        )
    else:
        np.testing.assert_array_less(
            np.abs(np.linalg.norm(output["data"]) - target["data_norm"])
            / target["data_norm"],
            tolerance,
        )
    print(np.abs(output["phi_m"][1] - target["phi_m"]) / target["phi_m"])
    print(tolerance)
    np.testing.assert_array_less(
        np.abs(output["phi_m"][1] - target["phi_m"]) / target["phi_m"], tolerance
    )
    np.testing.assert_array_less(
        np.abs(output["phi_d"][1] - target["phi_d"]) / target["phi_d"], tolerance
    )


def get_output_workspace(tmp_dir):
    """
    Extract the output geoh5 from the 'Temp' directory.
    """
    files = [
        file
        for file in os.listdir(os.path.join(tmp_dir, "Temp"))
        if file.endswith("geoh5")
    ]
    if len(files) != 1:
        raise UserWarning("Could not find a unique output workspace.")
    return os.path.join(tmp_dir, "Temp", files[0])
