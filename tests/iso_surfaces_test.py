#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from geoh5py.objects import BlockModel, Surface
from geoh5py.workspace import Workspace
import numpy as np
import os
from skimage.measure import marching_cubes
from geoapps.iso_surfaces.driver import IsoSurfacesDriver
import random

def test_centroids():
    ws = Workspace(os.path.abspath("C:/Users/JamieB/Documents/GIT/geoapps/assets/iso_test.geoh5"))

    # Generate a 3D array
    nx, ny, nz = 70, 70, 70 #np.random.randint(25, 100, size=3) #70, 70, 70 #(n-1)^3 points

    x = np.linspace(0, random.randint(5, 15), nx)
    y = np.linspace(0, random.randint(5, 15), ny)
    z = np.linspace(0, random.randint(5, 15), nz)

    origin = np.random.randint(-100, 100, 3)

    # Create test block model
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
    # Sphere test data for the block model
    n = min([nx, ny, nz])
    size = (n-1, n-1, n-1)#(nx-1, ny-1, nz-1)
    center = ((nx-2)/2, (ny-2)/2, (nz-2)/2) + np.random.randint(-10, 10, 3)
    #center = ((nx-2)/2, (ny-2)/2, (nz-2)/2)
    distance = np.linalg.norm(np.subtract(np.indices(size).T, np.asarray(center)), axis=len(center))
    sphere = np.ones(size) * (distance <= random.randint(1, 20))
    sphere = np.swapaxes(sphere, 1, 2) #.transpose((1, 2, 0))

    data = block_model.add_data(
        {
            "DataValues": {
                "association": "CELL",
                "values": sphere.flatten("F"),
            }
        }
    )

    func_surface = IsoSurfacesDriver.iso_surface(block_model, sphere, [0], resolution=100.0, max_distance=np.inf)

    Surface.create(
        ws,
        name="function surface",
        vertices=func_surface[0][0],
        cells=func_surface[0][1]
    )


test_centroids()
