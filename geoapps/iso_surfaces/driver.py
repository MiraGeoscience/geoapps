#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


from __future__ import annotations

import os
import sys

import numpy as np
from geoh5py.groups import ContainerGroup
from geoh5py.objects import BlockModel, ObjectBase, Surface
from geoh5py.ui_json import InputFile
from geoh5py.ui_json.utils import monitored_directory_copy
from scipy.interpolate import interp1d
from skimage.measure import marching_cubes

from geoapps.iso_surfaces.params import IsoSurfacesParams
from geoapps.shared_utils.utils import get_contours, rotate_xyz, weighted_average
from geoapps.utils.formatters import string_name


class IsoSurfacesDriver:
    def __init__(self, params: IsoSurfacesParams):
        self.params: IsoSurfacesParams = params

    def run(self):
        """
        Create iso surfaces from input values.
        """
        levels = get_contours(
            self.params.interval_min,
            self.params.interval_max,
            self.params.interval_spacing,
            self.params.fixed_contours,
        )

        if len(levels) < 1:

            return

        print("Starting the isosurface creation.")
        surfaces = self.iso_surface(
            self.params.objects,
            self.params.data.values,
            levels,
            resolution=self.params.resolution,
            max_distance=self.params.max_distance,
        )

        container = ContainerGroup.create(self.params.geoh5, name="Isosurface")
        result = []
        for surface, level in zip(surfaces, levels):
            if len(surface[0]) > 0 and len(surface[1]) > 0:
                result += [
                    Surface.create(
                        self.params.geoh5,
                        name=string_name(self.params.export_as + f"_{level:.2e}"),
                        vertices=surface[0],
                        cells=surface[1],
                        parent=container,
                    )
                ]
        if self.params.monitoring_directory is not None and os.path.exists(
            self.params.monitoring_directory
        ):
            monitored_directory_copy(self.params.monitoring_directory, container)

        print("Isosurface completed. " f"-> {len(surfaces)} surface(s) created.")

        return result

    @staticmethod
    def iso_surface(
        entity: ObjectBase,
        values: np.ndarray,
        levels: list,
        resolution: float = 100,
        max_distance: float = np.inf,
    ):
        """
        Generate 3D iso surface from an entity vertices or centroids and values.

        Parameters
        ----------
        entity: geoh5py.objects
            Any entity with 'vertices' or 'centroids' attribute.

        values: numpy.ndarray
            Array of values to create iso-surfaces from.

        levels: list of floats
            List of iso values

        max_distance: float, default=numpy.inf
            Maximum distance from input data to generate iso surface.
            Only used for input entities other than BlockModel.

        resolution: int, default=100
            Grid size used to generate the iso surface.
            Only used for input entities other than BlockModel.

        Returns
        -------
        surfaces: list of numpy.ndarrays
            List of surfaces (one per levels) defined by
            vertices and cell indices.
            [(vertices, cells)_level_1, ..., (vertices, cells)_level_n]
        """
        if getattr(entity, "vertices", None) is not None:
            locations = entity.vertices
        elif getattr(entity, "centroids", None) is not None:
            locations = entity.centroids
        else:
            print("Input 'entity' must have 'vertices' or 'centroids'.")
            return None

        if isinstance(entity, BlockModel):
            values = values.reshape(
                (entity.shape[2], entity.shape[0], entity.shape[1]), order="F"
            ).transpose((1, 2, 0))

            grid = []
            for i in ["u", "v", "z"]:
                cell_delimiters = getattr(entity, i + "_cell_delimiters")
                dx = cell_delimiters[1:] - cell_delimiters[:-1]
                grid.append(cell_delimiters[:-1] + dx / 2)

        else:
            grid = []
            for i in range(3):
                grid += [
                    np.arange(
                        locations[:, i].min(),
                        locations[:, i].max() + resolution,
                        resolution,
                    )
                ]

            y, x, z = np.meshgrid(grid[1], grid[0], grid[2])
            values = weighted_average(
                locations,
                np.c_[x.flatten(), y.flatten(), z.flatten()],
                [values],
                threshold=resolution / 2.0,
                n=8,
                max_distance=max_distance,
            )
            values = values[0].reshape(x.shape)

        surfaces = []
        for level in levels:
            try:
                verts, faces, _, _ = marching_cubes(values, level=level)

                # Remove all vertices and cells with nan
                nan_verts = np.any(np.isnan(verts), axis=1)
                rem_cells = np.any(nan_verts[faces], axis=1)

                active = np.arange(nan_verts.shape[0])
                active[nan_verts] = nan_verts.shape[0]
                _, inv_map = np.unique(active, return_inverse=True)

                verts = verts[nan_verts == False, :]
                faces = faces[rem_cells == False, :]
                faces = inv_map[faces].astype("uint32")

                vertices = []
                for i in range(3):
                    F = interp1d(
                        np.arange(grid[i].shape[0]), grid[i], fill_value="extrapolate"
                    )
                    vertices += [F(verts[:, i])]

                if isinstance(entity, BlockModel):
                    vertices = rotate_xyz(
                        np.vstack(vertices).T, [0, 0, 0], entity.rotation
                    )
                    vertices[:, 0] += entity.origin["x"]
                    vertices[:, 1] += entity.origin["y"]
                    vertices[:, 2] += entity.origin["z"]

                else:
                    vertices = np.vstack(vertices).T
            except RuntimeError:
                vertices, faces = [], []

            surfaces += [[vertices, faces]]

        return surfaces


if __name__ == "__main__":
    print("Loading geoh5 file . . .")
    file = sys.argv[1]
    params_class = IsoSurfacesParams(InputFile.read_ui_json(file))
    driver = IsoSurfacesDriver(params_class)

    print("Loaded. Running iso surface creation . . .")
    with params_class.geoh5.open(mode="r+"):
        driver.run()
    print("Done.")
