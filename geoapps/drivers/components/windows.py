#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from geoh5py.objects import Grid2D


class InversionWindow:
    def __init__(self, workspace, params):
        self.workspace = workspace
        self.params = params
        self._initialize()

    def _initialize(self):

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

    def is_empty(self):
        center_x_null = True if self.window["center"][0] is none else False
        center_y_null = True if self.window["center"][1] is none else False
        size_x_null = True if self.window["size"][0] is none else False
        size_y_null = True if self.window["size"][1] is none else False
