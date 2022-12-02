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
        padded = np.r_[np.zeros(width + 1), values]
        cumsum = np.cumsum(padded)
        mean += (cumsum[(width + 1) :] - cumsum[: (-width - 1)]) / (
            sum_weights[(width + 1) :] - sum_weights[: (-width - 1)]
        )

    # Backward averaging
    if method in ["centered", "backward"]:
        padded = np.r_[np.zeros(width + 1), values[::-1]]
        cumsum = np.cumsum(padded)
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


def cell_size_z(drape_model: DrapeModel) -> np.ndarray:
    """Compute z cell sizes of drape model."""
    hz = []
    for prism in drape_model.prisms:
        top_z, top_layer, n_layers = prism[2:]
        bottoms = drape_model.layers[
            range(int(top_layer), int(top_layer + n_layers)), 2
        ]
        z = np.hstack([top_z, bottoms])
        hz.append(z[:-1] - z[1:])
    return np.hstack(hz)


def active_from_xyz(
    mesh: DrapeModel | Octree,
    topo: np.ndarray,
    grid_reference="center",
    method="linear",
):
    """Returns an active cell index array below a surface

    :param mesh: Mesh object
    :param topo: Array of xyz locations
    :param grid_reference: Cell reference. Must be "center", "top", or "bottom"
    :param method: Interpolation method. Must be "linear", or "nearest"
    """

    mesh_dim = 2 if isinstance(mesh, DrapeModel) else 3
    locations = mesh.centroids.copy()

    if method == "linear":
        delaunay_2d = Delaunay(topo[:, :-1])
        z_interpolate = LinearNDInterpolator(delaunay_2d, topo[:, -1])
    elif method == "nearest":
        z_interpolate = NearestNDInterpolator(topo[:, :-1], topo[:, -1])
    else:
        raise ValueError("Method must be 'linear', or 'nearest'")

    if mesh_dim == 2:
        z_offset = cell_size_z(mesh) / 2.0
    else:
        z_offset = mesh.octree_cells["NCells"] * np.abs(mesh.w_cell_size) / 2

    # Shift cell center location to top or bottom of cell
    if grid_reference == "top":
        locations[:, -1] += z_offset
    elif grid_reference == "bottom":
        locations[:, -1] -= z_offset
    elif grid_reference == "center":
        pass
    else:
        raise ValueError("'grid_reference' must be one of 'center', 'top', or 'bottom'")

    z_locations = z_interpolate(locations[:, :2])

    # Apply nearest neighbour if in extrapolation
    ind_nan = np.isnan(z_locations)
    if any(ind_nan):
        tree = cKDTree(topo)
        _, ind = tree.query(locations[ind_nan, :])
        z_locations[ind_nan] = topo[ind, -1]

    # fill_nan(locations, z_locations, filler=topo[:, -1])

    # Return the active cell array
    return locations[:, -1] < z_locations
