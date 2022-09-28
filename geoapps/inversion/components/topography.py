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
    from geoapps.drivers import BaseParams
    from . import InversionMesh
    from typing import Any

from copy import deepcopy

import numpy as np
from geoh5py.objects import Curve
from geoh5py.shared import Entity

from geoapps.driver_base.utils import active_from_xyz
from geoapps.inversion.natural_sources.magnetotellurics.params import (
    MagnetotelluricsParams,
)
from geoapps.shared_utils.utils import filter_xy

from .data import InversionData
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

    def __init__(
        self,
        workspace: Workspace,
        params: BaseParams,
        inversion_data: InversionData,
        window: dict[str, Any],
    ):
        """
        :param: workspace: Geoh5py workspace object containing location based data.
        :param: params: Params object containing location based data parameters.
        :param: window: Center and size defining window for data, topography, etc.
        """
        super().__init__(workspace, params, window)
        self.inversion_data = inversion_data
        self.locations: np.ndarray = None
        self.mask: np.ndarray = None
        self._initialize()

    def _initialize(self):
        self.locations = self.get_locations(self.params.topography_object)
        self.mask = np.ones(len(self.locations), dtype=bool)
        topo_window = deepcopy(self.window)

        if topo_window is not None:
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

    def active_cells(self, mesh: InversionMesh, data: InversionData) -> np.ndarray:
        """
        Return mask that restricts models to set of earth cells.

        :param: mesh: inversion mesh.
        :return: active_cells: Mask that restricts a model to the set of
            earth cells that are active in the inversion (beneath topography).
        """
        if isinstance(self.params, MagnetotelluricsParams):
            active_cells = active_from_xyz(
                mesh.mesh, self.locations, grid_reference="bottom_nodes", logical="any"
            )
            active_cells[
                mesh.mesh._get_containing_cell_indexes(  # pylint: disable=protected-access
                    data.locations
                )
            ] = True
        else:
            mesh_object = (
                mesh.entity if "2d" in self.params.inversion_type else mesh.mesh
            )
            active_cells = active_from_xyz(
                mesh_object, self.locations, grid_reference="cell_centers"
            )

        if "2d" in self.params.inversion_type:
            ac_model = active_cells.astype("float64")
            active_cells = active_cells[np.argsort(mesh.permutation)]
        else:
            ac_model = active_cells[mesh.permutation].astype("float64")

        mesh.entity.add_data({"active_cells": {"values": ac_model}})

        return active_cells

    def get_locations(self, obj: Entity) -> np.ndarray:
        """
        Returns locations of data object centroids or vertices.

        :param obj: geoh5py object containing centroid or
            vertex location data

        :return: Array shape(*, 3) of x, y, z location data

        """

        locs = super().get_locations(obj)

        if self.params.topography is not None:
            if isinstance(self.params.topography, Entity):
                elev = self.params.topography.values
            elif isinstance(self.params.topography, (int, float)):
                elev = np.ones_like(locs[:, 2]) * self.params.topography
            else:
                elev = self.params.topography.values  # Must be FloatData at this point

            if not np.all(locs[:, 2] == elev):
                locs[:, 2] = elev

        return locs

    def write_entity(self):
        """Write out the survey to geoh5"""

        if "2d" in self.params.inversion_type:
            locs = self.inversion_data._survey.unique_locations  # pylint: disable=W0212
            entity = super().create_entity("Topo", locs, Curve)
        else:
            entity = super().create_entity("Topo", self.locations)

        return entity
