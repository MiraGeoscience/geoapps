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


def new_neighbors(d, id, nodes):
    """index into neighbor arrays excluding zero distance and past neighbors."""
    ind = [i in nodes if d[id.tolist().index(i)] != 0 else False for i in id]
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


def survey_lines(survey, start_loc):

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


def split_dcip_survey(survey, lines, on="A"):

    with survey.workspace.open(mode="r+"):

        current = survey.current_electrodes
        ab = survey.ab_cell_id
        survey_locs = survey.vertices
        survey_cells = survey.cells

        line_id = 1
        survey_locs, survey_loc_map = slice_and_map(survey_locs, lines == line_id)
        func = lambda c: (c[0] in survey_loc_map) & (c[1] in survey_loc_map)
        survey_cells, survey_cell_map = slice_and_map(survey_cells, func)
        survey_cells = np.array([[survey_loc_map[i] for i in c] for c in survey_cells])

        ab_cell_ids = np.array(ab.values[list(survey_cell_map)], dtype=int)
        current_cells, current_cell_map = slice_and_map(
            current.cells, np.unique(ab_cell_ids)
        )
        current_locs, current_loc_map = slice_and_map(
            current.vertices, np.unique(current_cells.ravel())
        )
        ab_cell_ids = np.array([current_cell_map[i] for i in ab_cell_ids])
        current_cells = np.array(
            [[current_loc_map[i] for i in c] for c in current_cells]
        )

        name = f"Line {line_id}"
        currents = CurrentElectrode.create(
            survey.workspace, name=name, vertices=current_locs
        )
        currents.cells = current_cells
        currents.add_default_ab_cell_id()

        potentials = PotentialElectrode.create(
            survey.workspace, name=name + "_rx", vertices=survey_locs
        )
        potentials.cells = survey_cells
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
        potentials.current_electrodes = currents
        for c in survey.children:
            if isinstance(c, FloatData) and "Pseudo" not in c.name:
                potentials.add_data(
                    {c.name: {"values": c.values[list(survey_cell_map)]}}
                )

    return potentials
