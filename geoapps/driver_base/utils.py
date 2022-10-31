#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import numpy as np
from geoh5py.objects import DrapeModel, Octree
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import Delaunay, cKDTree


def running_mean(
    values: np.array, width: int = 1, method: str = "centered"
) -> np.array:
    """
    Compute a running mean of an array over a defined width.

    :param values: Input values to compute the running mean over
    :param width: Number of neighboring values to be used
    :param method: Choice between 'forward', 'backward' and ['centered'] averaging.

    :return mean_values: Averaged array values of shape(values, )
    """
    # Averaging vector (1/N)
    weights = np.r_[np.zeros(width + 1), np.ones_like(values)]
    sum_weights = np.cumsum(weights)

    mean = np.zeros_like(values)

    # Forward averaging
    if method in ["centered", "forward"]:
        padd = np.r_[np.zeros(width + 1), values]
        cumsum = np.cumsum(padd)
        mean += (cumsum[(width + 1) :] - cumsum[: (-width - 1)]) / (
            sum_weights[(width + 1) :] - sum_weights[: (-width - 1)]
        )

    # Backward averaging
    if method in ["centered", "backward"]:
        padd = np.r_[np.zeros(width + 1), values[::-1]]
        cumsum = np.cumsum(padd)
        mean += (
            (cumsum[(width + 1) :] - cumsum[: (-width - 1)])
            / (sum_weights[(width + 1) :] - sum_weights[: (-width - 1)])
        )[::-1]

    if method == "centered":
        mean /= 2.0

    return mean


def treemesh_2_octree(workspace, treemesh, **kwargs):

    index_array, levels = getattr(treemesh, "_ubc_indArr")
    ubc_order = getattr(treemesh, "_ubc_order")

    index_array = index_array[ubc_order] - 1
    levels = levels[ubc_order]

    origin = treemesh.x0.copy()
    origin[2] += treemesh.h[2].size * treemesh.h[2][0]
    mesh_object = Octree.create(
        workspace,
        origin=origin,
        u_count=treemesh.h[0].size,
        v_count=treemesh.h[1].size,
        w_count=treemesh.h[2].size,
        u_cell_size=treemesh.h[0][0],
        v_cell_size=treemesh.h[1][0],
        w_cell_size=-treemesh.h[2][0],
        octree_cells=np.c_[index_array, levels],
        **kwargs,
    )

    return mesh_object


def active_from_xyz(
    mesh, xyz, grid_reference="cell_centers", method="linear", logical="all"
):
    """Returns an active cell index array below a surface

    **** ADAPTED FROM discretize.utils.mesh_utils.active_from_xyz ****

    """

    if isinstance(mesh, DrapeModel) and grid_reference != "cell_centers":
        raise ValueError("Drape models must use cell_centers grid reference.")

    if method == "linear":
        delaunay_2d = Delaunay(xyz[:, :2])
        z_interpolate = LinearNDInterpolator(delaunay_2d, xyz[:, 2])
    else:
        z_interpolate = NearestNDInterpolator(xyz[:, :2], xyz[:, 2])

    if grid_reference == "cell_centers":
        # this should work for all 4 mesh types...
        locations = (
            mesh.centroids if isinstance(mesh, (DrapeModel, Octree)) else mesh.gridCC
        )

    elif grid_reference == "top_nodes":
        locations = np.vstack(
            [
                mesh.gridCC
                + (np.c_[-1, 1, 1][:, None] * mesh.h_gridded / 2.0).squeeze(),
                mesh.gridCC
                + (np.c_[-1, -1, 1][:, None] * mesh.h_gridded / 2.0).squeeze(),
                mesh.gridCC
                + (np.c_[1, 1, 1][:, None] * mesh.h_gridded / 2.0).squeeze(),
                mesh.gridCC
                + (np.c_[1, -1, 1][:, None] * mesh.h_gridded / 2.0).squeeze(),
            ]
        )
    elif grid_reference == "bottom_nodes":
        locations = np.vstack(
            [
                mesh.gridCC
                + (np.c_[-1, 1, -1][:, None] * mesh.h_gridded / 2.0).squeeze(),
                mesh.gridCC
                + (np.c_[-1, -1, -1][:, None] * mesh.h_gridded / 2.0).squeeze(),
                mesh.gridCC
                + (np.c_[1, 1, -1][:, None] * mesh.h_gridded / 2.0).squeeze(),
                mesh.gridCC
                + (np.c_[1, -1, -1][:, None] * mesh.h_gridded / 2.0).squeeze(),
            ]
        )

    # Interpolate z values on CC or N
    z_xyz = z_interpolate(locations[:, :-1]).squeeze()

    # Apply nearest neighbour if in extrapolation
    ind_nan = np.isnan(z_xyz)
    if any(ind_nan):
        tree = cKDTree(xyz)
        _, ind = tree.query(locations[ind_nan, :])
        z_xyz[ind_nan] = xyz[ind, -1]

    # Create an active bool of all True

    n_cells = mesh.n_cells if isinstance(mesh, (DrapeModel, Octree)) else mesh.nC
    active = getattr(np, logical)(
        (locations[:, -1] < z_xyz).reshape((n_cells, -1), order="F"), axis=1
    )

    return active.ravel()
