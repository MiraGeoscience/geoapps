#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from geoapps.utils.statistics import is_outlier


def new_neighbors(dist, neighbor, nodes):
    """index into neighbor arrays excluding zero distance and past neighbors."""
    ind = [
        i in nodes if dist[neighbor.tolist().index(i)] != 0 else False for i in neighbor
    ]
    return np.where(ind)[0].tolist()


def next_neighbor(tree, point, nodes, n=3):
    """Returns smallest distance neighbor that has not yet been traversed"""
    distances, ids = tree.query(point, n)
    new_id = new_neighbors(distances, ids, nodes)
    if any(new_id):
        distances = distances[new_id]
        ids = ids[new_id]
        next_id = np.argmin(distances)
        return distances[next_id], ids[next_id]

    else:
        return next_neighbor(tree, point, nodes, n + 3)


def survey_lines(survey, start_loc):
    """Generate line ids for a survey object."""

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
        dist, neighbor_id = next_neighbor(tree, loc, nodes)

        outlier = False
        if len(distances) > 1:
            if all(dist == distances):
                outlier = False
            else:
                outlier = is_outlier(distances, dist)

        if outlier:
            line_id += 1
            distances = []
        else:
            distances.append(dist)

        nodes.pop(nodes.index(neighbor_id))
        loc = locs[neighbor_id]

    lines += [line_id]  # nodes run out before last id assigned

    return np.array(lines)
