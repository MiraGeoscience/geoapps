#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


from __future__ import annotations

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from uuid import UUID
    from geoh5py.workspace import Workspace
    from geoapps.io.params import Params

import numpy as np
from geoh5py.objects import Grid2D

from geoapps.utils import rotate_xy


class InversionLocations:
    """
    Retrieve topography data from workspace and apply transformations.

    Parameters
    ----------
    mask :
        Mask that stores cumulative filtering actions.
    origin :
        Rotation origin.
    angle :
        Rotation angle.
    is_rotated :
        True if locations have been rotated.
    locations :
        xyz locations.

    Methods
    -------
    get_locations() :
        Returns locations of data object centroids or vertices.
    filter() :
        Apply accumulated self.mask to array, or dict of arrays.
    rotate() :
        Un-rotate data using origin and angle assigned to inversion mesh.

    """

    def __init__(self, workspace: Workspace, params: Params, window: Dict[str, Any]):
        """
        :param: workspace: Geoh5py workspace object containing location based data.
        :param: params: Params object containing location based data parameters.
        :param: window: Center and size defining window for data, topography, etc.

        """
        self.workspace = workspace
        self.params = params
        self.window = window
        self.mask: np.ndarray = None
        self.origin: List[float] = None
        self.angle: float = None
        self.is_rotated: bool = False
        self.locations: np.ndarray = None

        mesh = workspace.get_entity(params.mesh)[0]
        if mesh.rotation is not None:
            self.origin = np.asarray(mesh.origin.tolist())
            self.angle = -1 * mesh.rotation[0]
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

    def get_locations(self, uid: UUID) -> np.ndarray:
        """
        Returns locations of data object centroids or vertices.

        :param: uid: UUID of geoh5py object containing centroid or
            vertex location data
        :return: locs: N by 3 array of x, y, z location data

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

        return locs

    def filter(self, a: Union[Dict[str, np.ndarray], np.ndarray]):
        """
        Apply accumulated self.mask to array, or dict of arrays.

        If argument a is a dictionary filter will be applied to all key/values.

        :param: a: Object containing dat ato filter.

        :return: Filtered data.

        """
        if isinstance(a, dict):
            return {k: v[self.mask] for k, v in a.items()}
        else:
            return a[self.mask]

    def rotate(self, locs: np.ndarray) -> np.ndarray:
        """
        Rotate data using origin and angle assigned to inversion mesh.

        Since rotation attribute is stored with a negative sign the applied
        rotation will restore locations to an East-West, North-South orientation.

        :param: locs: Array of xyz locations.
        """
        xy = rotate_xy(locs[:, :2], self.origin, self.angle)
        return np.c_[xy, locs[:, 2]]
