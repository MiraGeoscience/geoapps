#  Copyright (c) 2024 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geoh5py.workspace import Workspace
    from geoapps.inversion.params import InversionBaseParams


import numpy as np
from geoh5py.objects import Points
from geoh5py.shared import Entity
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree

from geoapps.shared_utils.utils import get_locations as get_locs


class InversionLocations:
    """
    Retrieve topography data from workspace and apply transformations.

    Parameters
    ----------
    mask :
        Mask that stores cumulative filtering actions.
    locations :
        xyz locations.

    Methods
    -------
    get_locations() :
        Returns locations of data object centroids or vertices.
    filter() :
        Apply accumulated self.mask to array, or dict of arrays.

    """

    def __init__(self, workspace: Workspace, params: InversionBaseParams):
        """
        :param workspace: Geoh5py workspace object containing location based data.
        :param params: Params object containing location based data parameters.
        """
        self.workspace = workspace
        self._params: InversionBaseParams = params
        self.mask: np.ndarray | None = None
        self.locations: np.ndarray | None = None

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

    def create_entity(self, name, locs: np.ndarray, geoh5_object=Points):
        """Create Data group and Points object with observed data."""
        entity = geoh5_object.create(
            self.workspace,
            name=name,
            vertices=locs,
            parent=self.params.out_group,
        )

        return entity

    def get_locations(self, obj) -> np.ndarray:
        """
        Returns locations of data object centroids or vertices.

        :param uid: UUID of geoh5py object containing centroid or
            vertex location data

        :return: Array shape(*, 3) of x, y, z location data

        """

        locs = get_locs(self.workspace, obj)

        if locs is None:
            msg = f"Workspace object {obj} 'vertices' attribute is None."
            msg += " Object type should be Grid2D or point-like."
            raise (ValueError(msg))

        return locs

    def _filter(self, a, mask):
        for k, v in a.items():
            if not isinstance(v, np.ndarray):
                a.update({k: self._filter(v, mask)})
            else:
                a.update({k: v[mask]})
        return a

    def _none_dict(self, a):
        is_none = []
        for v in a.values():
            if isinstance(v, dict):
                v = None if self._none_dict(v) else 1
            is_none.append(v is None)
        return all(is_none)

    def filter(self, a: dict[str, np.ndarray] | np.ndarray, mask=None):
        """
        Apply accumulated self.mask to array, or dict of arrays.

        If argument a is a dictionary filter will be applied to all key/values.

        :param a: Object containing data to filter.

        :return: Filtered data.

        """

        mask = self.mask if mask is None else mask

        if isinstance(a, dict):
            if self._none_dict(a):
                return a
            else:
                return self._filter(a, mask)
        else:
            if a is None:
                return None
            else:
                return a[mask]

    def set_z_from_topo(self, locs: np.ndarray):
        """interpolate locations z data from topography."""

        if locs is None:
            return None

        topo = self.get_locations(self.params.topography_object)
        if self.params.topography is not None:
            if isinstance(self.params.topography, Entity):
                z = self.params.topography.values
            else:
                z = np.ones_like(topo[:, 2]) * self.params.topography

            topo[:, 2] = z

        xyz = locs.copy()
        topo_interpolator = LinearNDInterpolator(topo[:, :2], topo[:, 2])
        z_topo = topo_interpolator(xyz[:, :2])
        if np.any(np.isnan(z_topo)):
            tree = cKDTree(topo[:, :2])
            _, ind = tree.query(xyz[np.isnan(z_topo), :2])
            z_topo[np.isnan(z_topo)] = topo[ind, 2]
        xyz[:, 2] = z_topo

        return xyz

    @property
    def params(self):
        """Associated parameters."""
        return self._params
