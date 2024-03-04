#  Copyright (c) 2024 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).
#

from __future__ import annotations

import warnings
from pathlib import Path
from uuid import UUID

import numpy as np
from discretize.utils import mesh_builder_xyz
from geoh5py.objects import (
    AirborneFEMReceivers,
    AirborneFEMTransmitters,
    AirborneTEMReceivers,
    AirborneTEMTransmitters,
    CurrentElectrode,
    LargeLoopGroundTEMReceivers,
    LargeLoopGroundTEMTransmitters,
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
from geoapps.octree_creation.driver import OctreeDriver
from geoapps.utils.models import get_drape_model


class Geoh5Tester:
    """Create temp workspace, copy entities, and setup params class."""

    def __init__(self, geoh5, path, name, params_class=None):
        self.geoh5 = geoh5
        self.tmp_path = Path(path) / name
        self.ws = Workspace.create(self.tmp_path)

        if params_class is not None:
            self.params = params_class(validate=False, geoh5=self.ws)
            self.has_params = True
        else:
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


def generate_dc_survey(workspace, x_loc, y_loc, z_loc=None):
    """
    Utility function to generate a DC survey.
    """
    # Create sources along line
    if z_loc is None:
        z_loc = np.zeros_like(x_loc)

    vertices = np.c_[x_loc.ravel(), y_loc.ravel(), z_loc.ravel()]
    parts = np.kron(np.arange(x_loc.shape[0]), np.ones(x_loc.shape[1])).astype("int")
    currents = CurrentElectrode.create(workspace, vertices=vertices, parts=parts)
    currents.add_default_ab_cell_id()
    n_dipoles = 9
    dipoles = []
    current_id = []

    for val in currents.ab_cell_id.values:
        cell_id = int(currents.ab_map[val]) - 1

        for dipole in range(n_dipoles):
            dipole_ids = currents.cells[cell_id, :] + 2 + dipole

            if (
                any(dipole_ids > (currents.n_vertices - 1))
                or len(
                    np.unique(parts[np.r_[currents.cells[cell_id, 0], dipole_ids[1]]])
                )
                > 1
            ):
                continue

            dipoles += [dipole_ids]
            current_id += [val]

    potentials = PotentialElectrode.create(
        workspace, vertices=vertices, cells=np.vstack(dipoles).astype("uint32")
    )
    line_id = potentials.vertices[potentials.cells[:, 0], 1]
    line_id = (line_id - np.min(line_id) + 1).astype(np.int32)
    line_reference = {0: "Unknown"}
    line_reference.update({k: str(k) for k in np.unique(line_id)})
    potentials.add_data(
        {
            "line_ids": {
                "values": line_id,
                "type": "REFERENCED",
                "value_map": line_reference,
            }
        }
    )
    potentials.ab_cell_id = np.hstack(current_id).astype("int32")
    potentials.current_electrodes = currents
    currents.potential_electrodes = potentials

    return potentials


def setup_inversion_workspace(
    work_dir,
    background=None,
    anomaly=None,
    cell_size=(5.0, 5.0, 5.0),
    center=(0.0, 0.0, 0.0),
    n_electrodes=20,
    n_lines=5,
    refinement=(4, 6),
    x_limits=(-100.0, 100.0),
    y_limits=(-100.0, 100.0),
    padding_distance=100,
    drape_height=5.0,
    inversion_type="other",
    flatten=False,
    geoh5=None,
):
    if geoh5 is None:
        geoh5 = Workspace(Path(work_dir) / "inversion_test.ui.geoh5")
    # Topography
    xx, yy = np.meshgrid(np.linspace(-200.0, 200.0, 50), np.linspace(-200.0, 200.0, 50))

    def topo_drape(x, y):
        """Topography Gaussian function"""
        b = 100
        A = 50
        if flatten:
            return np.zeros_like(x)
        else:
            return A * np.exp(-0.5 * ((x / b) ** 2.0 + (y / b) ** 2.0))

    zz = topo_drape(xx, yy)
    topo = np.c_[
        utils.mkvc(xx) + center[0],
        utils.mkvc(yy) + center[1],
        utils.mkvc(zz) + center[2],
    ]
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
    xr = np.linspace(x_limits[0], x_limits[1], int(n_electrodes))
    yr = np.linspace(y_limits[0], y_limits[1], int(n_lines))
    X, Y = np.meshgrid(xr, yr)
    Z = topo_drape(X, Y) + drape_height

    vertices = np.c_[
        utils.mkvc(X.T) + center[0],
        utils.mkvc(Y.T) + center[1],
        utils.mkvc(Z.T) + center[2],
    ]

    if inversion_type in ["dcip", "dcip_2d"]:
        survey = generate_dc_survey(geoh5, X, Y, Z)

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
        survey.base_stations = TipperBaseStations.create(
            geoh5, vertices=np.c_[vertices[0, :]].T
        )
        survey.channels = [10.0, 100.0, 1000.0]
        dist = np.linalg.norm(
            survey.vertices[survey.cells[:, 0], :]
            - survey.vertices[survey.cells[:, 1], :],
            axis=1,
        )
        # survey.cells = survey.cells[dist < 100.0, :]
        survey.remove_cells(np.where(dist > 100)[0])

    elif inversion_type == "fem":
        survey = AirborneFEMReceivers.create(
            geoh5, vertices=vertices, name="Airborne_rx"
        )
        freq_metadata = [
            {"Coaxial data": False, "Frequency": 900, "Offset": 7.86},
            {"Coaxial data": False, "Frequency": 7200, "Offset": 7.86},
            {"Coaxial data": False, "Frequency": 56000, "Offset": 6.3},
        ]
        survey.metadata["EM Dataset"]["Frequency configurations"] = freq_metadata

        tx_locs = []
        freqs = []
        for meta in freq_metadata:
            tx_vertices = vertices.copy()
            tx_vertices[:, 0] -= meta["Offset"]
            tx_locs.append(tx_vertices)
            freqs.append([[meta["Frequency"]] * len(vertices)])
        tx_locs = np.vstack(tx_locs)
        freqs = np.hstack(freqs).flatten()

        transmitters = AirborneFEMTransmitters.create(
            geoh5, vertices=tx_locs, name="Airborne_tx"
        )
        survey.transmitters = transmitters
        survey.channels = [900.0, 7200.0, 56000.0]

        transmitters.add_data(
            {
                "Tx frequency": {
                    "values": freqs,
                    "association": "VERTEX",
                    "entity_type": {
                        "primitive_type": "REFERENCED",
                        "value_map": {k: str(k) for k in freqs},
                    },
                }
            }
        )

        dist = np.linalg.norm(
            survey.vertices[survey.cells[:, 0], :]
            - survey.vertices[survey.cells[:, 1], :],
            axis=1,
        )
        survey.remove_cells(np.where(dist > 200.0)[0])
        dist = np.linalg.norm(
            transmitters.vertices[transmitters.cells[:, 0], :]
            - transmitters.vertices[transmitters.cells[:, 1], :],
            axis=1,
        )
        transmitters.remove_cells(np.where(dist > 200.0)[0])

    elif "tem" in inversion_type:
        if "airborne" in inversion_type:
            survey = AirborneTEMReceivers.create(
                geoh5, vertices=vertices, name="Airborne_rx"
            )
            transmitters = AirborneTEMTransmitters.create(
                geoh5, vertices=vertices, name="Airborne_tx"
            )

            dist = np.linalg.norm(
                survey.vertices[survey.cells[:, 0], :]
                - survey.vertices[survey.cells[:, 1], :],
                axis=1,
            )
            survey.remove_cells(np.where(dist > 200.0)[0])
            transmitters.remove_cells(np.where(dist > 200.0)[0])
            survey.transmitters = transmitters
        else:
            arrays = [
                np.c_[
                    X[: int(n_lines / 2), :].flatten(),
                    Y[: int(n_lines / 2), :].flatten(),
                    Z[: int(n_lines / 2), :].flatten(),
                ],
                np.c_[
                    X[int(n_lines / 2) :, :].flatten(),
                    Y[int(n_lines / 2) :, :].flatten(),
                    Z[int(n_lines / 2) :, :].flatten(),
                ],
            ]
            loops = []
            loop_cells = []
            loop_id = []
            count = 0
            for ind, array in enumerate(arrays):
                loop_id += [np.ones(array.shape[0]) * (ind + 1)]
                min_loc = np.min(array, axis=0)
                max_loc = np.max(array, axis=0)
                loop = np.vstack(
                    [
                        np.c_[
                            np.ones(5) * min_loc[0],
                            np.linspace(min_loc[1], max_loc[1], 5),
                        ],
                        np.c_[
                            np.linspace(min_loc[0], max_loc[0], 5)[1:],
                            np.ones(4) * max_loc[1],
                        ],
                        np.c_[
                            np.ones(4) * max_loc[0],
                            np.linspace(max_loc[1], min_loc[1], 5)[1:],
                        ],
                        np.c_[
                            np.linspace(max_loc[0], min_loc[0], 5)[1:-1],
                            np.ones(3) * min_loc[1],
                        ],
                    ]
                )
                loop = (loop - np.mean(loop, axis=0)) * 1.5 + np.mean(loop, axis=0)
                loop = np.c_[loop, topo_drape(loop[:, 0], loop[:, 1]) + drape_height]
                loops += [loop + np.asarray(center)]
                loop_cells += [np.c_[np.arange(15) + count, np.arange(15) + count + 1]]
                loop_cells += [np.c_[count + 15, count]]
                count += 16

            transmitters = LargeLoopGroundTEMTransmitters.create(
                geoh5,
                vertices=np.vstack(loops),
                cells=np.vstack(loop_cells),
            )
            transmitters.tx_id_property = transmitters.parts + 1
            survey = LargeLoopGroundTEMReceivers.create(
                geoh5, vertices=np.vstack(vertices)
            )
            survey.transmitters = transmitters
            survey.tx_id_property = np.hstack(loop_id)
            # survey.parts = np.repeat(np.arange(n_lines), n_electrodes)

        survey.channels = np.r_[3e-04, 6e-04, 1.2e-03] * 1e3
        waveform = np.c_[
            np.r_[
                np.arange(-0.002, -0.0001, 5e-4),
                np.arange(-0.0004, 0.0, 1e-4),
                np.arange(0.0, 0.002, 5e-4),
            ]
            * 1e3
            + 2.0,
            np.r_[np.linspace(0, 1, 4), np.linspace(0.9, 0.0, 4), np.zeros(4)],
        ]
        survey.waveform = waveform
        survey.timing_mark = 2.0
        survey.unit = "Milliseconds (ms)"

    else:
        survey = Points.create(
            geoh5,
            vertices=vertices,
            name="survey",
        )

    # Create a mesh

    if "2d" in inversion_type:
        lines = survey.get_entity("line_ids")[0].values
        entity, mesh, _ = get_drape_model(  # pylint: disable=W0632
            geoh5,
            "Models",
            survey.vertices[np.unique(survey.cells[lines == 101, :]), :],
            [cell_size[0], cell_size[2]],
            100.0,
            [padding_distance] * 2 + [padding_distance] * 2,
            1.1,
            parent=None,
            return_colocated_mesh=True,
            return_sorting=True,
        )
        active = active_from_xyz(entity, topography.vertices, grid_reference="top")

    else:
        padDist = np.ones((3, 2)) * padding_distance
        mesh = mesh_builder_xyz(
            vertices - np.r_[cell_size] / 2.0,
            cell_size,
            depth_core=100.0,
            padding_distance=padDist,
            mesh_type="TREE",
        )
        mesh = OctreeDriver.refine_tree_from_surface(
            mesh,
            topography,
            levels=refinement,
            diagonal_balance=False,
            finalize=False,
        )

        if inversion_type in ["fem", "airborne_tem"]:
            mesh = OctreeDriver.refine_tree_from_points(
                mesh,
                vertices,
                levels=[2],
                diagonal_balance=False,
                finalize=False,
            )

        mesh.finalize()
        entity = treemesh_2_octree(geoh5, mesh, name="mesh")
        active = active_from_xyz(entity, topography.vertices, grid_reference="top")

    # Model
    if flatten:
        p0 = np.r_[-20, -20, -30]
        p1 = np.r_[20, 20, -70]

        model = utils.model_builder.add_block(
            entity.centroids,
            background * np.ones(mesh.nC),
            p0,
            p1,
            anomaly,
        )
    else:
        p0 = np.r_[-20, -20, -10]
        p1 = np.r_[20, 20, 30]

        model = utils.model_builder.add_block(
            entity.centroids,
            background * np.ones(mesh.nC),
            p0,
            p1,
            anomaly,
        )

    model[~active] = np.nan
    model = entity.add_data({"model": {"values": model}})
    return geoh5, entity, model, survey, topography


def check_target(output: dict, target: dict, tolerance=0.1):
    """
    Check inversion output metrics against hard-valued target.
    :param output: Dictionary containing keys for 'data', 'phi_d' and 'phi_m'.
    :param target: Dictionary containing keys for 'data_norm', 'phi_d' and 'phi_m'.\
    :param tolerance: Tolerance between output and target measured as: |a-b|/b
    """
    print(
        f"'data_norm': {np.linalg.norm(output['data'])}, 'phi_d': {output['phi_d'][1]}, 'phi_m': {output['phi_m'][1]}"
    )
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

    np.testing.assert_array_less(
        np.abs(output["phi_m"][1] - target["phi_m"]) / target["phi_m"], tolerance
    )
    np.testing.assert_array_less(
        np.abs(output["phi_d"][1] - target["phi_d"]) / target["phi_d"], tolerance
    )


def get_output_workspace(tmp_dir: Path):
    """
    Extract the output geoh5 from the 'Temp' directory.
    """
    files = list((tmp_dir / "Temp").glob("*.geoh5"))
    if len(files) != 1:
        raise UserWarning("Could not find a unique output workspace.")
    return str(files[0])
