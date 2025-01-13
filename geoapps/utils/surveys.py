# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024-2025 Mira Geoscience Ltd.                                '
#                                                                              '
#  This file is part of geoapps.                                               '
#                                                                              '
#  geoapps is distributed under the terms and conditions of the MIT License    '
#  (see LICENSE file at the root of this source code package).                 '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from __future__ import annotations

from typing import TYPE_CHECKING

from SimPEG.survey import BaseSurvey


if TYPE_CHECKING:
    from simpeg_drivers.components import InversionData

from collections.abc import Callable

import numpy as np
from discretize import TensorMesh, TreeMesh
from geoh5py.objects import PotentialElectrode
from geoh5py.objects.surveys.electromagnetics.base import LargeLoopGroundEMSurvey
from geoh5py.workspace import Workspace
from scipy.spatial import cKDTree


def get_containing_cells(
    mesh: TreeMesh | TensorMesh, data: InversionData
) -> np.ndarray:
    """
    Find indices of cells that contain data locations

    :param mesh: Computational mesh object
    :param data: Inversion data object
    """
    if isinstance(mesh, TreeMesh):
        inds = mesh._get_containing_cell_indexes(  # pylint: disable=protected-access
            data.locations
        )

        if isinstance(data.entity, LargeLoopGroundEMSurvey):
            line_ind = []
            transmitters = data.entity.transmitters
            for cell in transmitters.cells:
                line_ind.append(
                    mesh.get_cells_along_line(
                        transmitters.vertices[cell[0], :],
                        transmitters.vertices[cell[1], :],
                    )
                )
            inds = np.unique(np.r_[inds, np.hstack(line_ind)])

    elif isinstance(mesh, TensorMesh):
        locations = data.drape_locations(np.unique(data.locations, axis=0))
        xi = np.searchsorted(mesh.nodes_x, locations[:, 0]) - 1
        yi = np.searchsorted(mesh.nodes_y, locations[:, -1]) - 1
        inds = xi * mesh.shape_cells[1] + yi

    else:
        raise TypeError("Mesh must be 'TreeMesh' or 'TensorMesh'")

    return inds


def new_neighbors(distances: np.ndarray, neighbors: np.ndarray, nodes: list[int]):
    """
    Index into neighbor arrays excluding zero distance and past neighbors.

    :param: distances: previously computed distances
    :param: neighbors: Possible neighbors
    :param: nodes: Traversed point ids.
    """
    ind = [
        i in nodes if distances[neighbors.tolist().index(i)] != 0 else False
        for i in neighbors
    ]
    return np.where(ind)[0].tolist()


def next_neighbor(tree: cKDTree, point: list[float], nodes: list[int], n: int = 3):
    """
    Returns smallest distance neighbor that has not yet been traversed.

    :param: tree: kd-tree computed for the point cloud of possible neighbors.
    :param: point: Current point being traversed.
    :param: nodes: Traversed point ids.
    """
    distances, neighbors = tree.query(point, n)
    new_ids = new_neighbors(distances, neighbors, nodes)
    if any(new_ids):
        distances = distances[new_ids]
        neighbors = neighbors[new_ids]
        next_id = np.argmin(distances)
        return distances[next_id], neighbors[next_id]

    else:
        return next_neighbor(tree, point, nodes, n + 3)


def find_unique_tops(locations: np.ndarray):
    """
    Finds the uppermost trace of a 2D section of points.

    :param: locations: Cartesian coordinates for points lying in a 2d section.
    """

    locs, inds = np.unique(locations[:, :2], axis=0, return_inverse=True)

    if len(locs) != len(locations):
        tops = []
        for i in range(len(locs)):
            tops.append(locations[:, 2][np.where(inds == i)[0]].max())

        locations = np.c_[locs, tops]

    return locations


def traveling_salesman(locs: np.ndarray) -> np.ndarray:
    """
    Finds the order of a roughly linear point set.

    Uses the point furthest from the mean location as the starting point.

    :param: locs: Cartesian coordinates of points lying either roughly within a plane or a line.
    :param: return_index: Return the indices of the end points in the original array.
    """
    mean = locs[:, :2].mean(axis=0)
    current = np.argmax(np.linalg.norm(locs[:, :2] - mean, axis=1))
    order = [current]
    mask = np.ones(locs.shape[0], dtype=bool)
    mask[current] = False

    for _ in range(locs.shape[0] - 1):
        remaining = np.where(mask)[0]
        ind = np.argmin(np.linalg.norm(locs[current, :2] - locs[remaining, :2], axis=1))
        current = remaining[ind]
        order.append(current)
        mask[current] = False

    return np.asarray(order)


def compute_alongline_distance(points: np.ndarray, ordered: bool = True):
    """
    Convert from cartesian (x, y, values) points to (distance, values) locations.

    :param: points: Cartesian coordinates of points lying either roughly within a
        plane or a line.
    """
    if not ordered:
        order = traveling_salesman(points)
        points = points[order, :]

    distances = np.cumsum(
        np.r_[0, np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)]
    )
    if points.shape[1] == 3:
        distances = np.c_[distances, points[:, 2:]]

    return distances


def slice_and_map(obj: np.ndarray, slicer: np.ndarray | Callable):
    """
    Slice an array and return both sliced array and global to local map.

    :param object: Array to be sliced.
    :param slicer: Boolean index array, Integer index array,  or callable
        that provides a condition to keep or remove each row of object.
    :return: Sliced array.
    :return: Dictionary map from global to local indices.
    """

    if isinstance(slicer, np.ndarray):
        if slicer.dtype == bool:
            sliced_object = obj[slicer]
            g2l = dict(zip(np.where(slicer)[0], np.arange(len(obj)), strict=False))
        else:
            sliced_object = obj[slicer]
            g2l = dict(zip(slicer, np.arange(len(slicer)), strict=False))

    elif callable(slicer):
        slicer = np.array([slicer(k) for k in obj])
        sliced_object = obj[slicer]
        g2l = dict(zip(np.where(slicer)[0], np.arange(len(obj)), strict=False))

    return sliced_object, g2l


def extract_dcip_survey(
    workspace: Workspace, survey: PotentialElectrode, cell_mask: np.ndarray
):
    """
    Returns a survey containing data from a single line.

    :param workspace: geoh5py workspace containing a valid DCIP survey.
    :param survey: PotentialElectrode object.
    :param cell_mask: Boolean array of M-N pairs to include in the new survey.
    """

    if not np.any(cell_mask):
        raise ValueError("No cells found in the mask.")

    active_poles = np.zeros(survey.n_vertices, dtype=bool)
    active_poles[survey.cells[cell_mask, :].ravel()] = True
    potentials = survey.copy(parent=workspace, mask=active_poles, cell_mask=cell_mask)

    return potentials


def split_dcip_survey(
    survey: PotentialElectrode,
    lines: np.ndarray,
    workspace: Workspace | None = None,
):
    """
    Split survey into sub-surveys each containing a single line of data.

    :param: survey: PotentialElectrode object.
    :param: lines: Line indexer for survey.
    :param: name: Name prefix to assign to the new survey object (to be
        suffixed with the line number).
    :param: workspace: destination for the resulting survey object (if
        different from the parent of the survey argument).
    """

    ws = workspace if workspace is not None else survey.workspace

    with ws.open(mode="r+") as ws:
        line_surveys = []
        for line_id in np.unique(lines):
            cell_mask = lines == line_id
            line_survey = extract_dcip_survey(ws, survey, cell_mask)
            line_surveys.append(line_survey)

    return line_surveys


def get_unique_locations(survey: BaseSurvey) -> np.ndarray:
    """
    Get unique locations from a survey including sources and receivers when
    applicable.

    :param: survey: SimPEG survey object.

    :return: Array of unique locations.
    """
    if survey.source_list:
        locations = []
        for source in survey.source_list:
            source_location = source.location
            if source_location is not None:
                if not isinstance(source_location, list):
                    locations += [[source_location]]
                else:
                    locations += [source_location]
            locations += [receiver.locations for receiver in source.receiver_list]
        locations = np.vstack([np.vstack(np.atleast_2d(*locs)) for locs in locations])
    else:
        locations = survey.receiver_locations

    return np.unique(locations, axis=0)


def get_intersecting_cells(locations: np.ndarray, mesh: TreeMesh) -> np.ndarray:
    """
    Find cells that intersect with a set of segments.

    :param: locations: Locations making a line path.
    :param: mesh: TreeMesh object.

    :return: Array of unique cell indices.
    """
    cell_index = []
    for ind in range(locations.shape[0] - 1):
        cell_index.append(mesh.get_cells_along_line(locations[ind], locations[ind + 1]))

    return np.unique(np.hstack(cell_index))
