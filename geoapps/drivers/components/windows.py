#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from typing import Dict

import numpy as np
from geoh5py.objects import Grid2D


class InversionWindow:
    """
    Retrieve window from workspace.

    Methods
    -------
    is_empty: Check if window data is empty.

    Notes
    -----
    If params contains no window data, the window will initialize with from the
    data extents.

    """

    def __init__(self, workspace, params):
        """
        :param: workspace: Geoh5py workspace object containing window data.
        :param: params: Params object containing window parameters.
        :param: window: Center and size defining window for data, topography, etc.
        """
        self.workspace = workspace
        self.params = params
        self.window: Dict[str, float] = None
        self._initialize()

    def _initialize(self) -> None:
        """ Extract data from workspace using params data. """

        self.window = self.params.window()

        if self.is_empty():

            data_object = self.workspace.get_entity(self.params.data_object)[0]
            if isinstance(data_object, Grid2D):
                locs = data_object.centroids
            else:
                locs = data_object.vertices

            if locs is None:
                msg = f"Object {data_object} is not Grid2D object and doesn't contain vertices."
                raise (ValueError(msg))

            xmin = np.min(locs[:, 0])
            xmax = np.max(locs[:, 0])
            ymin = np.min(locs[:, 1])
            ymax = np.max(locs[:, 1])

            self.window = {
                "center": [np.mean([xmin, xmax]), np.mean([ymin, ymax])],
                "size": [xmax - xmin, ymax - ymin],
            }

    def is_empty(self) -> bool:
        """ Check if window data is empty. """
        center_x_null = True if self.window["center"][0] is None else False
        center_y_null = True if self.window["center"][1] is None else False
        size_x_null = True if self.window["size"][0] is None else False
        size_y_null = True if self.window["size"][1] is None else False
        return center_x_null & center_y_null & size_x_null & size_y_null
