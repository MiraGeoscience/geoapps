#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import random

import numpy as np
from geoh5py.objects import BlockModel, Points, Surface
from geoh5py.workspace import Workspace

from geoapps.iso_surfaces.driver import IsoSurfacesDriver


def test_centroids():
    """
    Test iso_surface with a block model. Data values are the distance from a point.
    """
    ws = Workspace("./iso_test.geoh5")

    n = 70
    length = 10

    # Axes for block model
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

    # Sphere test data for the block model
    size = (n - 1, n - 1, n - 1)
    offset = np.random.random_sample(3) * n * 0.05
    sphere_center = ((n - 2) / 2, (n - 2) / 2, (n - 2) / 2) + offset
    # Convert to world units using the length of the block model axes
    sphere_center_real = sphere_center * (length / (n - 1))

    # The value at each point is its distance from the center of the sphere
    values = np.linalg.norm(
        np.subtract(np.indices(size).T, np.asarray(sphere_center)),
        axis=len(sphere_center),
    )
    values = np.swapaxes(values, 1, 2)

    sphere_radius = random.uniform(n * 0.15, n * 0.3)
    sphere_radius_real = sphere_radius * (length / (n - 1))

    data = block_model.add_data(
        {
            "DataValues": {
                "association": "CELL",
                "values": values.flatten("F"),
            }
        }
    )

    # Generate surface
    func_surface = IsoSurfacesDriver.iso_surface(
        block_model, values, [sphere_radius], resolution=100.0, max_distance=np.inf
    )

    surface = Surface.create(
        ws, name="surface", vertices=func_surface[0][0], cells=func_surface[0][1]
    )

    # Compare surface center with sphere center
    surf_center = np.mean(surface.vertices, axis=0)
    center_error = np.abs(
        ((sphere_center_real + origin) - surf_center) / (sphere_center_real + origin)
    )

    assert np.all(center_error < 0.01)

    # Radius of sphere
    surf_distance = np.linalg.norm(np.subtract(surface.vertices, surf_center), axis=1)
    surf_radius = np.mean(surf_distance, axis=0)
    radius_error = np.abs((surf_radius - sphere_radius_real) / sphere_radius_real)

    assert radius_error < 0.05


def test_vertices():
    """
    Test iso_surface with a points object. Data values are the distance from a point.
    """
    ws = Workspace("./iso_test.geoh5")

    length = 10
    origin = np.random.randint(-100, 100, 3)
    verts = np.random.randint(0, length, (1000, 3)) + origin
    offset = np.random.random_sample(3) * length * 0.05
    sphere_center = [length / 2, length / 2, length / 2] + offset

    values = np.linalg.norm(
        np.subtract(verts, np.asarray(origin + sphere_center)), axis=1
    ).flatten("F")
    sphere_radius = random.uniform(length * 0.15, length * 0.3)

    points = Points.create(
        ws,
        name="test_points",
        vertices=verts,
    )

    data = points.add_data(
        {
            "DataValues": {
                "association": "CELL",
                "values": values,
            }
        }
    )

    func_surface = IsoSurfacesDriver.iso_surface(
        points, values, [sphere_radius], resolution=(length / 100), max_distance=np.inf
    )

    surface = Surface.create(
        ws, name="surface", vertices=func_surface[0][0], cells=func_surface[0][1]
    )

    # Compare surface center with sphere center
    surf_center = np.mean(surface.vertices, axis=0)
    center_error = np.abs(
        ((sphere_center + origin) - surf_center) / (sphere_center + origin)
    )

    assert np.all(center_error < 0.05)

    # Radius of sphere
    surf_distance = np.linalg.norm(np.subtract(surface.vertices, surf_center), axis=1)
    surf_radius = np.mean(surf_distance, axis=0)
    radius_error = np.abs((surf_radius - sphere_radius) / sphere_radius)

    assert radius_error < 0.05
