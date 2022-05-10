#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import geoh5py
import numpy as np
from geoh5py.objects import (
    BlockModel,
)
from scipy.interpolate import interp1d
from skimage.measure import marching_cubes

from geoapps.utils import rotate_xy, weighted_average


def iso_surface(
    entity,
    values,
    levels,
    resolution=100,
    max_distance=np.inf,
):
    """
    Generate 3D iso surface from an entity vertices or centroids and values.

    Parameters
    ----------
    entity: geoh5py.objects
        Any entity with 'vertices' or 'centroids' attribute.

    values: numpy.ndarray
        Array of values to create iso-surfaces from.

    levels: list of floats
        List of iso values

    max_distance: float, default=numpy.inf
        Maximum distance from input data to generate iso surface.
        Only used for input entities other than BlockModel.

    resolution: int, default=100
        Grid size used to generate the iso surface.
        Only used for input entities other than BlockModel.

    Returns
    -------
    surfaces: list of numpy.ndarrays
        List of surfaces (one per levels) defined by
        vertices and cell indices.
        [(vertices, cells)_level_1, ..., (vertices, cells)_level_n]
    """
    if getattr(entity, "vertices", None) is not None:
        locations = entity.vertices
    elif getattr(entity, "centroids", None) is not None:
        locations = entity.centroids
    else:
        print("Input 'entity' must have 'vertices' or 'centroids'.")
        return None

    if isinstance(entity, BlockModel):
        values = values.reshape(
            (entity.shape[2], entity.shape[0], entity.shape[1]), order="F"
        ).transpose((1, 2, 0))
        grid = [
            entity.u_cell_delimiters,
            entity.v_cell_delimiters,
            entity.z_cell_delimiters,
        ]

    else:
        grid = []
        for ii in range(3):
            grid += [
                np.arange(
                    locations[:, ii].min(),
                    locations[:, ii].max() + resolution,
                    resolution,
                )
            ]

        y, x, z = np.meshgrid(grid[1], grid[0], grid[2])
        values = weighted_average(
            locations,
            np.c_[x.flatten(), y.flatten(), z.flatten()],
            [values],
            threshold=1e-1,
            n=8,
            max_distance=max_distance,
        )
        values = values[0].reshape(x.shape)

    surfaces = []
    for level in levels:
        try:
            verts, faces, _, _ = marching_cubes(values, level=level)

            # Remove all vertices and cells with nan
            nan_verts = np.any(np.isnan(verts), axis=1)
            rem_cells = np.any(nan_verts[faces], axis=1)

            active = np.arange(nan_verts.shape[0])
            active[nan_verts] = nan_verts.shape[0]
            _, inv_map = np.unique(active, return_inverse=True)

            verts = verts[nan_verts == False, :]
            faces = faces[rem_cells == False, :]
            faces = inv_map[faces].astype("uint32")

            vertices = []
            for ii in range(3):
                F = interp1d(
                    np.arange(grid[ii].shape[0]), grid[ii], fill_value="extrapolate"
                )
                vertices += [F(verts[:, ii])]

            if isinstance(entity, BlockModel):
                vertices = rotate_xy(np.vstack(vertices).T, [0, 0, 0], entity.rotation)
                vertices[:, 0] += entity.origin["x"]
                vertices[:, 1] += entity.origin["y"]
                vertices[:, 2] += entity.origin["z"]

            else:
                vertices = np.vstack(vertices).T
        except RuntimeError:
            vertices, faces = [], []

        surfaces += [[vertices, faces]]

    return surfaces