#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).
from __future__ import annotations

from typing import Callable

import numpy as np
from geoh5py.data import FloatData
from geoh5py.objects import CurrentElectrode, PotentialElectrode
from scipy.spatial import cKDTree

from geoapps.utils.statistics import is_outlier


def new_neighbors(distances, neighbors, nodes):
    """index into neighbor arrays excluding zero distance and past neighbors."""
    ind = [
        i in nodes if distances[neighbors.tolist().index(i)] != 0 else False
        for i in neighbors
    ]
    return np.where(ind)[0].tolist()


def next_neighbor(tree, point, nodes, n=3):
    """Returns smallest distance neighbor that has not yet been traversed"""
    d, id = tree.query(point, n)
    new_id = new_neighbors(d, id, nodes)
    if any(new_id):
        d = d[new_id]
        id = id[new_id]
        next = np.argmin(d)
        return d[next], id[next]

    else:
        return next_neighbor(tree, point, nodes, n + 3)


def find_endpoints(locs, ends=None, start_index=0):
    """Finds the end locations of a roughly linear point set."""

    ends = [] if ends is None else ends
    start = locs[start_index, :2]
    dist = np.linalg.norm(start - locs[:, :2], axis=1)
    end_id = np.where(dist == dist.max())[0]
    ends.append(locs[end_id].squeeze().tolist())
    if len(ends) < 2:
        ends = find_endpoints(locs, ends, start_index=end_id)
    return ends


def compute_alongline_distance(points):
    """Convert from cartesian (x, y, z) points to (distance, z) locations."""
    endpoints = find_endpoints(points)
    return np.linalg.norm(endpoints[0] - points, axis=1)


def survey_lines(survey, start_loc, save=False):
    """Build an array of line ids for a survey layed out in a line biased grid."""

    # extract xy locations and create linear indexing
    locs = survey.vertices[:, :2]
    nodes = np.arange(len(locs)).tolist()

    # find the id of the closest point to the starting location
    start_id = np.argmin(
        np.sqrt(((locs[:, 0] - start_loc[0]) ** 2) + ((locs[:, 1]) - start_loc[1]) ** 2)
    )

    # pop the starting location and index out of their respective lists
    locs = locs.tolist()
    loc = locs[start_id]
    _ = nodes.pop(start_id)

    # compute the tree of the remaining points and begin to traverse the tree
    # in the direction of closest neighbors.  Label points with same line id
    # until an outlier is detected in the distance to the next closest point,
    # then increase the line id.
    tree = cKDTree(locs)
    line_id = 1  # zero is reserved
    lines = []
    distances = []
    while nodes:

        lines.append(line_id)
        d, id = next_neighbor(tree, loc, nodes)

        outlier = False
        if len(distances) > 1:
            if all(d == distances):
                outlier = False
            else:
                outlier = is_outlier(distances, d)

        if outlier:
            line_id += 1
            distances = []
        else:
            distances.append(d)

        nodes.pop(nodes.index(id))
        loc = locs[id]

    lines += [line_id]  # nodes run out before last id assigned

    if save:
        survey.add_data(
            {
                "Line ID": {
                    "values": lines,
                    "association": "CELL",
                    "entity_type": {
                        "primitive_type": "REFERENCED",
                        "value_map": {k: str(k) for k in lines},
                    },
                }
            }
        )

    return np.array(lines)


def slice_and_map(object: np.ndarray, slicer: np.ndarray | Callable):
    """
    Slice an array and return both sliced array and global to local map.

    :param object: Array to be sliced.
    :param slicer: Boolean index array, Integer index array,  or callable
        that provides a condition to keep or remove each row of object.

    :returns: Sliced array.
    :returns: Dictionary map from global to local indices.
    """

    if isinstance(slicer, np.ndarray):

        if slicer.dtype == bool:
            sliced_object = object[slicer]
            g2l = dict(zip(np.where(slicer)[0], np.arange(len(object))))
        else:
            sliced_object = object[slicer]
            g2l = dict(zip(slicer, np.arange(len(slicer))))

    elif callable(slicer):

        slicer = np.array([slicer(k) for k in object])
        sliced_object = object[slicer]
        g2l = dict(zip(np.where(slicer)[0], np.arange(len(object))))

    return sliced_object, g2l


def extract_dcip_survey(workspace, survey, lines, line_id, name="Line"):
    """Returns a survey containing data from a single line."""

    current = survey.current_electrodes

    # Extract line locations and store map into full survey
    survey_locs, survey_loc_map = slice_and_map(survey.vertices, lines == line_id)

    # Use line locations to slice cells and store map into full survey
    func = lambda c: (c[0] in survey_loc_map) & (c[1] in survey_loc_map)
    survey_cells, survey_cell_map = slice_and_map(survey.cells, func)
    survey_cells = [[survey_loc_map[i] for i in c] for c in survey_cells]

    # Use line cells to slice ab_cell_ids
    ab_cell_ids = survey.ab_cell_id.values[list(survey_cell_map)]
    ab_cell_ids = np.array(ab_cell_ids, dtype=int) - 1

    # Use line ab_cell_ids to slice current cells
    current_cells, current_cell_map = slice_and_map(
        current.cells, np.unique(ab_cell_ids)
    )

    # Use line current cells to slice current locs
    current_locs, current_loc_map = slice_and_map(
        current.vertices, np.unique(current_cells.ravel())
    )

    # Remap global ids to local counterparts
    ab_cell_ids = np.array([current_cell_map[i] for i in ab_cell_ids])
    current_cells = [[current_loc_map[i] for i in c] for c in current_cells]

    # Save objects
    line_name = f"{name} {line_id}"
    currents = CurrentElectrode.create(
        workspace, name=f"{line_name} (currents)", vertices=current_locs
    )
    currents.cells = np.array(current_cells)
    currents.add_default_ab_cell_id()

    potentials = PotentialElectrode.create(
        workspace, name=line_name, vertices=survey_locs
    )
    potentials.cells = np.array(survey_cells)

    # Add ab_cell_id as referenced data object
    value_map = {k + 1: str(k + 1) for k in ab_cell_ids}
    value_map.update({0: "Unknown"})
    ab_cell_id = potentials.add_data(
        {
            "A-B Cell ID": {
                "values": ab_cell_ids + 1,
                "association": "CELL",
                "entity_type": {
                    "primitive_type": "REFERENCED",
                    "value_map": value_map,
                },
            }
        }
    )

    # Attach current and potential objects and copy data slice into line survey
    potentials.current_electrodes = currents
    for c in survey.children:
        if isinstance(c, FloatData) and "Pseudo" not in c.name:
            potentials.add_data({c.name: {"values": c.values[list(survey_cell_map)]}})

    return potentials


def split_dcip_survey(survey, lines, name="Line", workspace=None):
    """Split survey into sub-surveys each containing a single line of data."""

    ws = workspace if workspace is not None else survey.workspace

    with ws.open(mode="r+") as ws:
        line_surveys = []
        for line_id in np.unique(lines):
            line_survey = extract_dcip_survey(
                survey.workspace, survey, lines, line_id, name
            )
            line_surveys.append(survey)

    return line_surveys
