#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from geoh5py.objects import BlockModel
from geoh5py.workspace import Workspace
import numpy as np
import os

def test_centroids():
    ws = Workspace(os.path.abspath("C:/Users/JamieB/Documents/GIT/geoapps/assets/iso_test.geoh5"))

    # Generate a 3D array
    nx, ny, nz = 16, 16, 16

    x = np.linspace(0, 10, nx)
    y = np.linspace(0, 10, ny)
    z = np.linspace(0, 10, nz)

    origin = [0.0, 0.0, 0.0]

    block_model = BlockModel.create(
        ws,
        origin=origin,
        u_cell_delimiters=x,
        v_cell_delimiters=y,
        z_cell_delimiters=z,
        name="test_block_model",
        allow_move=False,
    )

    # https://stackoverflow.com/questions/53326570/how-to-create-a-sphere-inside-an-ndarray/53339684#53339684
    size = (nx-1, ny-1, nz-1)
    center = (7, 7, 7)
    distance = np.linalg.norm(np.subtract(np.indices(size).T, np.asarray(center)), axis=len(center))
    sphere = np.ones(size) * (distance <= 3)
    print(len(block_model.centroids))
    print(len(sphere.flatten()))

    data = block_model.add_data(
        {
            "DataValues": {
                "association": "CELL",
                "values": sphere.flatten(),
            }
        }
    )


test_centroids()
