#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from uuid import UUID

import numpy as np
from geoh5py.objects import Grid2D
from scipy.interpolate import LinearNDInterpolator

from geoapps.utils import rotate_xy


class InversionLocations:
    """
    Retrieve topography data from workspace and apply transformations.

    Methods
    -------
    get_locs: Returns locations of data object centroids or vertices.
    filter: Apply accumulated self.mask to array, or dict of arrays.
    rotate: Un-rotate data using origin and angle assigned to inversion mesh.

    """

    def __init__(self, workspace, params, window):
        """
        :param: workspace: Geoh5py workspace object containing location based data.
        :param: params: Params object containing location based data parameters.
        :param: window: Center and size defining window for data, topography, etc.
        :param: mask: Mask that stores cumulative filtering actions.
        :param: origin: Rotation origin.
        :param: angle: Rotation angle.
        :param: is_rotated: True if locations have been rotated.
        :param: locs: xyz locations.
        """
        self.workspace = workspace
        self.params = params
        self.window = window
        self.mask = None
        self.origin = None
        self.angle = None
        self.is_rotated = False
        self.locs = None

        mesh = workspace.get_entity(params.mesh)[0]
        if mesh.rotation is not None:
            self.origin = np.asarray(mesh.origin.tolist())
            self.angle = -mesh.rotation[0]
            self.is_rotated = True

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, v):
        if v is None:
            self._mask = v
            return
        if np.all([n in [0, 1] for n in np.unique(v)]):
            v = np.array(v, dtype=bool)
        else:
            msg = f"Badly formed mask array {v}"
            raise (ValueError(msg))
        self._mask = v

    def get_locs(self, uid: UUID) -> np.ndarray:
        """
        Returns locations of data object centroids or vertices.

        :param: uid: UUID of geoh5py object containing centroid or
            vertex location data
        :returns: locs: N by 3 array of x, y, z location data

        """

        data_object = self.workspace.get_entity(uid)[0]

        if isinstance(data_object, Grid2D):
            locs = data_object.centroids
        else:
            locs = data_object.vertices

        if locs is None:
            msg = f"Workspace object {data_object} 'vertices' attribute is None."
            msg += " Object type should be Grid2D or point-like."
            raise (ValueError(msg))

        if data_object.uid == self.params.topography_object:
            if self.params.topography is not None:
                elev = self.workspace.get_entity(self.params.topography)[0].values
                if not np.all(locs[:, 2] == elev):
                    locs[:, 2] = elev

        return locs

    def filter(self, a):
        """ Apply accumulated self.mask to array, or dict of arrays. """
        if isinstance(a, dict):
            return {k: v[self.mask] for k, v in a.items()}
        else:
            return a[self.mask]

    def rotate(self, locs: np.ndarray) -> np.ndarray:
        """ Un-rotate data using origin and angle assigned to inversion mesh. """
        xy = rotate_xy(locs[:, :2], self.origin, self.angle)
        return np.c_[xy, locs[:, 2]]

    def z_from_topo(self, locs: np.ndarray):
        """ interpolate locations z data from topography. """

        topo = self.get_locs(self.params.topography_object)

        xyz = locs.copy()
        topo_interpolator = LinearNDInterpolator(topo[:, :2], topo[:, 2])
        z_topo = topo_interpolator(xyz[:, :2])
        if np.any(np.isnan(z_topo)):
            tree = cKDTree(topo[:, :2])
            _, ind = tree.query(xyz[np.isnan(z_topo), :2])
            z_topo[np.isnan(z_topo)] = topo[ind, 2]
        xyz[:, 2] = z_topo

        return xyz
