#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from copy import deepcopy

import numpy as np
from discretize.utils import active_from_xyz
from SimPEG import maps

from geoapps.utils import filter_xy

from .locations import InversionLocations


class InversionTopography(InversionLocations):
    """ Retrieve topography data from workspace and apply transformations. """

    def __init__(self, workspace, params, window):
        super().__init__(workspace, params, window)
        self._initialize()

    def _initialize(self):

        self.locs = super().get_locs(self.params.topography_object)
        self.mask = np.ones(len(self.locs), dtype=bool)

        topo_window = deepcopy(self.window)
        topo_window["size"] = [2 * s for s in topo_window["size"]]
        self.mask = filter_xy(
            self.locs[:, 0],
            self.locs[:, 1],
            window=topo_window,
            angle=self.angle,
            mask=self.mask,
        )

        self.locs = super().filter(self.locs)

        if self.is_rotated:
            self.locs = super().rotate(self.locs)

    def active_cells(self, mesh):
        active_cells = active_from_xyz(mesh, self.locs, grid_reference="N")

        return active_cells
