#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geoh5py.workspace import Workspace
    from geoapps.io import Params
    from . import InversionMesh

from copy import deepcopy

import numpy as np
from discretize.utils import active_from_xyz

from geoapps.utils import filter_xy

from .locations import InversionLocations


class InversionTopography(InversionLocations):
    """
    Retrieve topography data from workspace and apply transformations.

    Parameters
    ----------
    locations :
        Topography locations.
    mask :
        Mask created by windowing operation and applied to locations
        and data on initialization.

    Methods
    -------
    active_cells(mesh) :
        Return mask that restricts models to active (earth) cells.

    """

    def __init__(self, workspace: Workspace, params: Params, window: dict[str, Any]):
        """
        :param: workspace: Geoh5py workspace object containing location based data.
        :param: params: Params object containing location based data parameters.
        :param: window: Center and size defining window for data, topography, etc.
        """
        super().__init__(workspace, params, window)
        self.locations: np.ndarray = None
        self.mask: np.ndarray = None
        self._initialize()

    def _initialize(self):

        self.locations = self.get_locations(self.params.topography_object)

        self.mask = np.ones(len(self.locations), dtype=bool)

        topo_window = deepcopy(self.window)
        topo_window["size"] = [2 * s for s in topo_window["size"]]
        self.mask = filter_xy(
            self.locations[:, 0],
            self.locations[:, 1],
            window=topo_window,
            angle=self.angle,
            mask=self.mask,
        )

        self.locations = super().filter(self.locations)

        if self.is_rotated:
            self.locations = super().rotate(self.locations)

        self.entity = self.write_entity()

    def active_cells(self, mesh: InversionMesh) -> np.ndarray:
        """
        Return mask that restricts models to set of earth cells.

        :param: mesh: Inversion mesh.
        :return: active_cells: Mask that restricts a model to the set of
            earth cells that are active in the inversion (beneath topography).
        """
        active_cells = active_from_xyz(mesh.mesh, self.locations, grid_reference="N")
        mesh.entity.add_data(
            {
                "active_cells": {
                    "values": active_cells[mesh.octree_permutation].astype("float64")
                }
            }
        )

        return active_cells

    def get_locations(self, uid: UUID) -> np.ndarray:
        """
        Returns locations of data object centroids or vertices.

        :param uid: UUID of geoh5py object containing centroid or
            vertex location data

        :return: Array shape(*, 3) of x, y, z location data

        """

        locs = super().get_locations(uid)

        if self.params.topography is not None:
            elev = self.workspace.get_entity(self.params.topography)[0].values
            if not np.all(locs[:, 2] == elev):
                locs[:, 2] = elev

        return locs

    def write_entity(self):
        """Write out the survey to geoh5"""

        entity = super().create_entity("Topo", self.locations)

        return entity
