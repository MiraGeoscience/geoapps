#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import numpy as np
from geoh5py.objects import Grid2D
from scipy.interpolate import NearestNDInterpolator

from geoapps.utils import filter_xy, rotate_xy


def get_topography(workspace, params, mesh, window):

    topography_object = workspace.get_entity(params.topography_object)[0]
    if isinstance(topography_object, Grid2D):
        topo_locs = topography_object.centroids
    else:
        topo_locs = topography_object.vertices

    if workspace.list_entities_name[params.topography] != "Z":
        topo_locs[:, 2] = workspace.get_entity(params.topography)[0].values

    if window is not None:
        topo_window = window.copy()
        topo_window["size"] = [ll * 2 for ll in window["size"]]
        ind = filter_xy(
            topo_locs[:, 0],
            topo_locs[:, 1],
            params.resolution / 2,
            window=topo_window,
            angle=-mesh.rotation["angle"],
        )
        xy_rot = rotate_xy(
            topo_locs[ind, :2],
            mesh.rotation["origin"],
            -mesh.rotation["angle"],
        )
        topo_locs = np.c_[xy_rot, topo_locs[ind, 2]]

    topo_interp_function = NearestNDInterpolator(topo_locs[:, :2], topo_locs[:, 2])

    return topo_locs, topo_interp_function
