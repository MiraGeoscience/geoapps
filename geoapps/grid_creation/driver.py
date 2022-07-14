#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).
from __future__ import annotations

import os
import sys
from os import path

import numpy as np
from geoh5py.ui_json import InputFile, monitored_directory_copy
from scipy.spatial import cKDTree

from geoapps.grid_creation.params import GridCreationParams
from geoapps.interpolation.application import DataInterpolation
from geoapps.shared_utils.utils import get_locations


class GridCreationDriver:
    def __init__(self, params: GridCreationParams):
        self.params: GridCreationParams = params

    def run(self):
        xyz = get_locations(self.params.geoh5, self.params.objects)
        if xyz is None:
            raise ValueError("Input object has no centroids or vertices.")

        tree = cKDTree(xyz)

        xyz_ref = get_locations(self.params.geoh5, self.params.xy_reference)
        if xyz_ref is None:
            print("No object selected for 'Lateral Extent'. Defaults to input object.")
            xyz_ref = xyz

        # Find extent of grid
        h = [self.params.cell_size_x, self.params.cell_size_y, self.params.cell_size_z]
        # pads: W, E, S, N, D, U
        pads = [
            self.params.horizontal_padding,
            self.params.horizontal_padding,
            self.params.horizontal_padding,
            self.params.horizontal_padding,
            self.params.bottom_padding,
            0.0,
        ]

        object_out = DataInterpolation.get_block_model(
            self.params.geoh5,
            self.params.new_grid,
            xyz_ref,
            h,
            self.params.depth_core,
            pads,
            self.params.expansion_fact,
        )

        # Try to recenter on nearest
        # Find nearest cells
        rad, ind = tree.query(object_out.centroids)
        ind_nn = np.argmin(rad)

        d_xyz = object_out.centroids[ind_nn, :] - xyz[ind[ind_nn], :]

        object_out.origin = np.r_[object_out.origin.tolist()] - d_xyz

        if self.params.monitoring_directory is not None and path.exists(
            os.path.abspath(self.params.monitoring_directory)
        ):
            monitored_directory_copy(
                os.path.abspath(self.params.monitoring_directory), object_out
            )


if __name__ == "__main__":
    print("Loading geoh5 file . . .")
    file = sys.argv[1]
    ifile = InputFile.read_ui_json(file)
    params = GridCreationParams(ifile)
    driver = GridCreationDriver(params)
    print("Loaded. Creating block model . . .")
    driver.run()
    print("Saved to " + params.geoh5.h5file)
