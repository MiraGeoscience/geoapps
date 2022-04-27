#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import numpy as np
from geoh5py.objects import Octree
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

    indArr, levels = treemesh._ubc_indArr
    ubc_order = treemesh._ubc_order

    indArr = indArr[ubc_order] - 1
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
        octree_cells=np.c_[indArr, levels],
        **kwargs,
    )

    return mesh_object


def active_from_xyz(
    mesh, xyz, grid_reference="cell_centers", method="linear", logical="all"
):
    """Returns an active cell index array below a surface

    **** ADAPTED FROM discretize.utils.mesh_utils.active_from_xyz ****


    """
    if method == "linear":
        tri2D = Delaunay(xyz[:, :2])
        z_interpolate = LinearNDInterpolator(tri2D, xyz[:, 2])
    else:
        z_interpolate = NearestNDInterpolator(xyz[:, :2], xyz[:, 2])

    if grid_reference == "cell_centers":
        # this should work for all 4 mesh types...
        locations = mesh.gridCC

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
    active = getattr(np, logical)(
        (locations[:, -1] < z_xyz).reshape((mesh.nC, -1), order="F"), axis=1
    )

    return active.ravel()
