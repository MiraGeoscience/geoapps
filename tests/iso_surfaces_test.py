# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024 Mira Geoscience Ltd.                                     '
#                                                                              '
#  This file is part of geoapps.                                               '
#                                                                              '
#  geoapps is distributed under the terms and conditions of the MIT License    '
#  (see LICENSE file at the root of this source code package).                 '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from __future__ import annotations

from pathlib import Path

import numpy as np
from geoh5py.objects import BlockModel, Points, Surface
from geoh5py.workspace import Workspace

from geoapps.iso_surfaces.driver import IsoSurfacesDriver


def test_centroids(tmp_path: Path):
    """
    Test iso_surface with a block model. Data values are the distance from a point.
    """
    ws = Workspace(tmp_path / "iso_test.geoh5")
    np.random.seed(0)
    n = 70
    length = 10

    # Axes for block model
    x = np.linspace(0, length, n)
    y = np.linspace(0, length, n)
    z = np.linspace(0, length, n)

    origin = np.random.uniform(-100, 100, 3)

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
    sphere_radius = np.random.uniform(length * 0.15, length * 0.3)
    offset = np.random.uniform(0, (length / 2) - sphere_radius, 3)
    sphere_center = ((length - 2) / 2, (length - 2) / 2, (length - 2) / 2) + offset

    # The value at each point is its distance from the center of the sphere
    x_coords, y_coords, z_coords = np.meshgrid(
        np.linspace(0, length, n - 1),
        np.linspace(0, length, n - 1),
        np.linspace(0, length, n - 1),
    )
    verts = np.stack(
        (x_coords.flatten(), y_coords.flatten(), z_coords.flatten()), axis=1
    )

    values = np.linalg.norm(
        np.subtract(verts, np.asarray(sphere_center)),
        axis=1,
    )

    # Generate surface
    func_surface = IsoSurfacesDriver.iso_surface(
        block_model, values, [sphere_radius], max_distance=np.inf
    )

    # Compare surface center with sphere center
    surf_center = np.mean(func_surface[0][0], axis=0)
    center_error = np.abs(
        ((sphere_center + origin) - surf_center) / (sphere_center + origin)
    )

    assert np.all(center_error < 0.02)

    # Radius of sphere
    surf_distance = np.linalg.norm(np.subtract(func_surface[0][0], surf_center), axis=1)
    surf_radius = np.mean(surf_distance, axis=0)
    radius_error = np.abs((surf_radius - sphere_radius) / sphere_radius)

    assert radius_error < 0.02

    # For user validation only
    Surface.create(
        ws, name="surface", vertices=func_surface[0][0], cells=func_surface[0][1]
    )
    block_model.add_data(
        {
            "DataValues": {
                "values": values,
            }
        }
    )
    ws.close()


def test_vertices(tmp_path: Path):
    """
    Test iso_surface with a points object. Data values are the distance from a point.
    """
    ws = Workspace(tmp_path / "iso_test.geoh5")
    np.random.seed(0)
    length = 10
    origin = np.random.uniform(-100, 100, 3)
    verts = np.random.randn(5000, 3) * length + origin
    sphere_radius = np.random.uniform(length * 0.2, length * 0.5, 1)
    offset = np.random.uniform(0, (length / 2), 3)
    sphere_center = origin + offset

    values = np.linalg.norm(verts - sphere_center, axis=1)

    points = Points.create(
        ws,
        name="test_points",
        vertices=verts,
    )

    func_surface = IsoSurfacesDriver.iso_surface(
        points,
        values,
        [sphere_radius],
        resolution=sphere_radius / 8.0,
        max_distance=np.inf,
    )

    # For user validation only
    Surface.create(
        ws, name="surface", vertices=func_surface[0][0], cells=func_surface[0][1]
    )
    points.add_data(
        {
            "DataValues": {
                "values": values,
            }
        }
    )
    ws.close()

    # Compare surface center with sphere center
    surf_center = np.mean(func_surface[0][0], axis=0)
    center_error = np.abs((sphere_center - surf_center) / (sphere_center))

    assert np.all(center_error < 0.25)

    # Radius of sphere
    surf_distance = np.linalg.norm(np.subtract(func_surface[0][0], surf_center), axis=1)
    surf_radius = np.mean(surf_distance, axis=0)
    radius_error = np.abs((surf_radius - sphere_radius) / sphere_radius)

    assert radius_error < 0.06
