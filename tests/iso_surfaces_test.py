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

def test_centroids():
    ws = Workspace(os.path.abspath("C:/Users/JamieB/Documents/GIT/geoapps/assets/iso_test.geoh5"))

    # Generate a 3D array
    nx, ny, nz = 64, 64, 64 #(n-1)^3 points

    x = np.linspace(0, 10, nx)
    y = np.linspace(0, 10, ny)
    z = np.linspace(0, 10, nz)

    origin = [0.0, 0.0, 0.0]

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
    size = (nx-1, ny-1, nz-1)
    center = (15, 7, 25)
    #center = ((nx-2)/2, (ny-2)/2, (nz-2)/2)
    distance = np.linalg.norm(np.subtract(np.indices(size).T, np.asarray(center)), axis=len(center))
    sphere = np.ones(size) * (distance <= 6)

    data = block_model.add_data(
        {
            "DataValues": {
                "association": "CELL",
                "values": sphere.flatten("F"), #C, F, A, K
            }
        }
    )

    '''
    # Apply marching cubes
    verts, faces, _, _ = marching_cubes(sphere, level=0)
    
    # Create test surface
    surface = Surface.create(
        ws,
        vertices=verts,
        cells=faces,
        name="test_surface")
    '''

    func_surface = IsoSurfacesDriver.iso_surface(block_model, sphere, [0], resolution=100.0, max_distance=np.inf)

    Surface.create(
        ws,
        name="function surface",
        vertices=func_surface[0][0],
        cells=func_surface[0][1]
    )




test_centroids()
