#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import warnings

import numpy as np
from discretize import TreeMesh
from scipy.spatial import ConvexHull, cKDTree
from SimPEG.electromagnetics.frequency_domain.sources import (
    LineCurrent as FEMLineCurrent,
)
from SimPEG.electromagnetics.time_domain.sources import LineCurrent as TEMLineCurrent
from SimPEG.survey import BaseSurvey
from SimPEG.utils import mkvc

from geoapps.utils.surveys import get_intersecting_cells, get_unique_locations


def calculate_2D_trend(
    points: np.ndarray, values: np.ndarray, order: int = 0, method: str = "all"
):
    """
    detrend2D(points, values, order=0, method='all')

    Function to remove a trend from 2D scatter points with values

    Parameters:
    ----------

    points: array or floats, shape(*, 2)
        Coordinates of input points

    values: array of floats, shape(*,)
        Values to be de-trended

    order: Order of the polynomial to be used

    method: str
        Method to be used for the detrending
            "all": USe all points
            "perimeter": Only use points on the convex hull


    Returns
    -------

    trend: array of floats, shape(*,)
        Calculated trend

    coefficients: array of floats, shape(order+1)
        Coefficients for the polynomial describing the trend

        trend = c[0] + points[:, 0] * c[1] +  points[:, 1] * c[2]
    """
    if not isinstance(order, int) or order < 0:
        raise ValueError(
            "Polynomial 'order' should be an integer > 0. "
            f"Value of {order} provided."
        )

    ind_nan = ~np.isnan(values)
    loc_xy = points[ind_nan, :]
    values = values[ind_nan]

    if method == "perimeter":
        hull = ConvexHull(loc_xy[:, :2])
        # Extract only those points that make the ConvexHull
        loc_xy = loc_xy[hull.vertices, :2]
        values = values[hull.vertices]
    elif not method == "all":
        raise ValueError(
            "'method' must be either 'all', or 'perimeter'. " f"Value {method} provided"
        )

    # Compute center of mass
    center_x = np.sum(loc_xy[:, 0] * np.abs(values)) / np.sum(np.abs(values))
    center_y = np.sum(loc_xy[:, 1] * np.abs(values)) / np.sum(np.abs(values))

    polynomial = []
    xx, yy = np.triu_indices(order + 1)
    for x, y in zip(xx, yy):
        polynomial.append(
            (loc_xy[:, 0] - center_x) ** float(x)
            * (loc_xy[:, 1] - center_y) ** float(y - x)
        )
    polynomial = np.vstack(polynomial).T

    if polynomial.shape[0] <= polynomial.shape[1]:
        raise ValueError(
            "The number of input values must be greater than the number of coefficients in the polynomial. "
            f"Provided {polynomial.shape[0]} values for a {order}th order polynomial with {polynomial.shape[1]} coefficients."
        )

    params, _, _, _ = np.linalg.lstsq(polynomial, values, rcond=None)
    data_trend = np.zeros(points.shape[0])
    for count, (x, y) in enumerate(zip(xx, yy)):
        data_trend += (
            params[count]
            * (points[:, 0] - center_x) ** float(x)
            * (points[:, 1] - center_y) ** float(y - x)
        )
    print(
        f"Removed {order}th order polynomial trend with mean: {np.mean(data_trend):.6g}"
    )
    return data_trend, params


def create_nested_mesh(
    survey: BaseSurvey,
    base_mesh: TreeMesh,
    padding_cells: int = 8,
    minimum_level: int = 3,
    finalize: bool = True,
):
    """
    Create a nested mesh with the same extent as the input global mesh.
    Refinement levels are preserved only around the input locations (local survey).

    Parameters
    ----------

    locations: Array of coordinates for the local survey shape(*, 3).
    base_mesh: Input global TreeMesh object.
    padding_cells: Used for 'method'= 'padding_cells'. Number of cells in each concentric shell.
    minimum_level: Minimum octree level to preserve everywhere outside the local survey area.
    finalize: Return a finalized local treemesh.
    """
    locations = get_unique_locations(survey)
    nested_mesh = TreeMesh(
        [base_mesh.h[0], base_mesh.h[1], base_mesh.h[2]], x0=base_mesh.x0
    )
    base_level = base_mesh.max_level - minimum_level
    base_refinement = base_mesh.cell_levels_by_index(np.arange(base_mesh.nC))
    base_refinement[base_refinement > base_level] = base_level
    nested_mesh.insert_cells(
        base_mesh.gridCC,
        base_refinement,
        finalize=False,
    )
    base_cell = np.min([base_mesh.h[0][0], base_mesh.h[1][0]])
    tx_loops = []
    for source in survey.source_list:
        if isinstance(source, (TEMLineCurrent, FEMLineCurrent)):
            mesh_indices = get_intersecting_cells(source.location, base_mesh)
            tx_loops.append(base_mesh.cell_centers[mesh_indices, :])

    if tx_loops:
        locations = np.vstack([locations] + tx_loops)

    tree = cKDTree(locations[:, :2])
    rad, _ = tree.query(base_mesh.gridCC[:, :2])
    pad_distance = 0.0
    for ii in range(minimum_level):
        pad_distance += base_cell * 2**ii * padding_cells
        indices = np.where(rad < pad_distance)[0]
        levels = base_mesh.cell_levels_by_index(indices)
        levels[levels > (base_mesh.max_level - ii)] = base_mesh.max_level - ii
        nested_mesh.insert_cells(
            base_mesh.gridCC[indices, :],
            levels,
            finalize=False,
        )

    if finalize:
        nested_mesh.finalize()

    return nested_mesh


def tile_locations(
    locations,
    n_tiles,
    minimize=True,
    method="kmeans",
    bounding_box=False,
    count=False,
    unique_id=False,
):
    """
    Function to tile a survey points into smaller square subsets of points

    :param numpy.ndarray locations: n x 2 array of locations [x,y]
    :param integer n_tiles: number of tiles (for 'cluster'), or number of
        refinement steps ('other')
    :param Bool minimize: shrink tile sizes to minimum
    :param string method: set to 'kmeans' to use better quality clustering, or anything
        else to use more memory efficient method for large problems
    :param bounding_box: bool [False]
        Return the SW and NE corners of each tile.
    :param count: bool [False]
        Return the number of locations in each tile.
    :param unique_id: bool [False]
        Return the unique identifiers of all tiles.

    RETURNS:
    :param list: Return a list of arrays with the for the SW and NE
                        limits of each tiles
    :param integer binCount: Number of points in each tile
    :param list labels: Cluster index of each point n=0:(nTargetTiles-1)
    :param numpy.array tile_numbers: Vector of tile numbers for each count in binCount

    NOTE: All X Y and xy products are legacy now values, and are only used
    for plotting functions. They are not used in any calculations and could
    be dropped from the return calls in future versions.


    """

    if method == "kmeans":
        # Best for smaller problems

        np.random.seed(0)
        # Cluster
        # TODO turn off filter once sklearn has dealt with the issue causing the warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            from sklearn.cluster import KMeans

            cluster = KMeans(n_clusters=n_tiles, n_init="auto")
            cluster.fit_predict(locations[:, :2])

        labels = cluster.labels_

        # nData in each tile
        binCount = np.zeros(int(n_tiles))

        # x and y limits on each tile
        X1 = np.zeros_like(binCount)
        X2 = np.zeros_like(binCount)
        Y1 = np.zeros_like(binCount)
        Y2 = np.zeros_like(binCount)

        for ii in range(int(n_tiles)):
            mask = cluster.labels_ == ii
            X1[ii] = locations[mask, 0].min()
            X2[ii] = locations[mask, 0].max()
            Y1[ii] = locations[mask, 1].min()
            Y2[ii] = locations[mask, 1].max()
            binCount[ii] = mask.sum()

        xy1 = np.c_[X1[binCount > 0], Y1[binCount > 0]]
        xy2 = np.c_[X2[binCount > 0], Y2[binCount > 0]]

        # Get the tile numbers that exist, for compatibility with the next method
        tile_id = np.unique(cluster.labels_)

    else:
        # Works on larger problems
        # Initialize variables
        # Test each refinement level for maximum space coverage
        nTx = 1
        nTy = 1
        for ii in range(int(n_tiles + 1)):
            nTx += 1
            nTy += 1

            testx = np.percentile(locations[:, 0], np.arange(0, 100, 100 / nTx))
            testy = np.percentile(locations[:, 1], np.arange(0, 100, 100 / nTy))

            # if ii > 0:
            dx = testx[:-1] - testx[1:]
            dy = testy[:-1] - testy[1:]

            if np.mean(dx) > np.mean(dy):
                nTx -= 1
            else:
                nTy -= 1

            print(nTx, nTy)
        tilex = np.percentile(locations[:, 0], np.arange(0, 100, 100 / nTx))
        tiley = np.percentile(locations[:, 1], np.arange(0, 100, 100 / nTy))

        X1, Y1 = np.meshgrid(tilex, tiley)
        X2, Y2 = np.meshgrid(
            np.r_[tilex[1:], locations[:, 0].max()],
            np.r_[tiley[1:], locations[:, 1].max()],
        )

        # Plot data and tiles
        X1, Y1, X2, Y2 = mkvc(X1), mkvc(Y1), mkvc(X2), mkvc(Y2)
        binCount = np.zeros_like(X1)
        labels = np.zeros_like(locations[:, 0])
        for ii in range(X1.shape[0]):
            mask = (
                (locations[:, 0] >= X1[ii])
                * (locations[:, 0] <= X2[ii])
                * (locations[:, 1] >= Y1[ii])
                * (locations[:, 1] <= Y2[ii])
            ) == 1

            # Re-adjust the window size for tight fit
            if minimize:
                if mask.sum():
                    X1[ii], X2[ii] = (
                        locations[:, 0][mask].min(),
                        locations[:, 0][mask].max(),
                    )
                    Y1[ii], Y2[ii] = (
                        locations[:, 1][mask].min(),
                        locations[:, 1][mask].max(),
                    )

            labels[mask] = ii
            binCount[ii] = mask.sum()

        xy1 = np.c_[X1[binCount > 0], Y1[binCount > 0]]
        xy2 = np.c_[X2[binCount > 0], Y2[binCount > 0]]

        # Get the tile numbers that exist
        # Since some tiles may have 0 data locations, and are removed by
        # [binCount > 0], the tile numbers are no longer contiguous 0:nTiles
        tile_id = np.unique(labels)

    tiles = []
    for tid in tile_id.tolist():
        tiles += [np.where(labels == tid)[0]]

    out = [tiles]

    if bounding_box:
        out.append([xy1, xy2])

    if count:
        out.append(binCount[binCount > 0])

    if unique_id:
        out.append(tile_id)

    if len(out) == 1:
        return out[0]
    return tuple(out)
