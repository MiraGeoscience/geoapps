#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from geoh5py.objects import BlockModel, Surface, Points
from geoh5py.workspace import Workspace
import numpy as np
import os
from geoapps.iso_surfaces.driver import IsoSurfacesDriver
import random
from scipy import spatial


def test_centroids():
    ws = Workspace(os.path.abspath("C:/Users/JamieB/Documents/GIT/geoapps/assets/iso_test.geoh5"))

    # Generate a 3D array
    n = 70
    length = 10

    x = np.linspace(0, length, n)
    y = np.linspace(0, length, n)
    z = np.linspace(0, length, n)

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
    size = (n-1, n-1, n-1)
    offset = np.random.randint(-10, 10, 3)
    sphere_center = ((n-2)/2, (n-2)/2, (n-2)/2) + offset
    sphere_center_real = (length/2, length/2, length/2) + offset*(length/(n-1))

    distance = np.linalg.norm(np.subtract(np.indices(size).T, np.asarray(sphere_center)), axis=len(sphere_center))

    radius = random.randint(15, 30)
    radius_real = radius*(length/(n-1))

    sphere = np.ones(size) * (distance <= radius)
    sphere = np.swapaxes(sphere, 1, 2)

    data = block_model.add_data(
        {
            "DataValues": {
                "association": "CELL",
                "values": sphere.flatten("F"),
            }
        }
    )

    # Generate surface
    func_surface = IsoSurfacesDriver.iso_surface(block_model, sphere, [0], resolution=100.0, max_distance=np.inf)

    surface = Surface.create(
        ws,
        name="surface",
        vertices=func_surface[0][0],
        cells=func_surface[0][1]
    )

    # Compare surface center with sphere center
    surf_center = np.mean(surface.vertices, axis=0)
    center_error = np.abs(((sphere_center_real + origin) - surf_center)/(sphere_center_real + origin))
    print(np.all(center_error < 0.01))

    # Radius of sphere
    surf_distance = np.linalg.norm(np.subtract(surface.vertices, surf_center), axis=1)
    surf_radius = np.mean(surf_distance, axis=0)
    radius_error = np.abs((surf_radius-radius_real)/radius_real)
    print(radius_error < 0.05)


def test_vertices():
    ws = Workspace(os.path.abspath("C:/Users/JamieB/Documents/GIT/geoapps/assets/iso_test.geoh5"))

    verts = np.random.randint(0, 100, (100, 3))

    points = Points.create(
        ws,
        name="test_points",
        vertices=verts,
    )

    values = np.random.randint(1, 5, (100, 3))

    data = points.add_data(
        {
            "DataValues": {
                "association": "CELL",
                "values": values.flatten("F"),
            }
        }
    )

    func_surface = IsoSurfacesDriver.iso_surface(points, verts, [3], resolution=10.0, max_distance=np.inf)

    surface = Surface.create(
        ws,
        name="surface",
        vertices=func_surface[0][0],
        cells=func_surface[0][1]
    )


test_vertices()
