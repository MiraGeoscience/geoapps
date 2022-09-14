#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


from __future__ import annotations

import sys
from os import path

import geoh5py.data
import geoh5py.objects
import numpy as np
from geoh5py.groups import ContainerGroup
from geoh5py.objects import Curve, Grid2D
from geoh5py.ui_json import InputFile
from geoh5py.ui_json.utils import monitored_directory_copy
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line

from geoapps.edge_detection.params import EdgeDetectionParams
from geoapps.shared_utils.utils import filter_xy
from geoapps.utils.formatters import string_name


class EdgeDetectionDriver:
    def __init__(self, params: EdgeDetectionParams):
        self.params: EdgeDetectionParams = params

    def run(self):
        """
        Driver for Grid2D objects for the automated detection of line features.
        The application relies on the Canny and Hough transforms from the
        Scikit-Image library.
        """
        vertices, cells = EdgeDetectionDriver.get_edges(*self.params.edge_args())

        if vertices is not None:
            name = string_name(self.params.export_as)

            out_entity = ContainerGroup.create(
                workspace=self.params.geoh5,
                name=self.params.ga_group_name,
            )
            Curve.create(
                workspace=self.params.geoh5,
                name=name,
                vertices=vertices,
                cells=cells,
                parent=out_entity,
            )

            if self.params.monitoring_directory is not None and path.exists(
                self.params.monitoring_directory
            ):
                monitored_directory_copy(self.params.monitoring_directory, out_entity)

    @staticmethod
    def get_edges(
        grid: geoh5py.objects,
        data: geoh5py.data,
        sigma: float,
        line_length: int,
        threshold: int,
        line_gap: int,
        window_size: int,
        window_center_x: float,
        window_center_y: float,
        window_width: float,
        window_height: float,
        window_azimuth: float,
        resolution: float,
    ) -> [list[float], list[float]]:
        """
        Get indices within window.

        :params grid: A Grid2D object.
        :params data: Input data.
        :params sigma: Standard deviation of the Gaussian filter. (Canny)
        :params line_length: Minimum accepted pixel length of detected lines. (Hough)
        :params threshold: Value threshold. (Hough)
        :params line_gap: Maximum gap between pixels to still form a line. (Hough)
        :params window_size: Window size.
        :params window_center_x: Easting position of the selection box.
        :params window_center_y: Northing position of the selection box.
        :params window_width: Width (m) of the selection box.
        :params window_height: Height (m) of the selection box.
        :params window_azimuth: Rotation angle of the selection box.
        :params resolution: Minimum data separation (m).

        :returns : n x 3 array. Vertices of edges.
        :returns : list
            n x 2 float array. Cells of edges.

        """
        vertices, cells, xy = None, None, None

        x = grid.centroids[:, 0].reshape(grid.shape, order="F")
        y = grid.centroids[:, 1].reshape(grid.shape, order="F")
        z = grid.centroids[:, 2].reshape(grid.shape, order="F")
        grid_data = data.values.reshape(grid.shape, order="F")

        indices = np.ones_like(grid_data, dtype="bool")

        ind_x, ind_y = (
            np.any(indices, axis=1),
            np.any(indices, axis=0),
        )
        x = x[ind_x, :][:, ind_y]
        y = y[ind_x, :][:, ind_y]
        z = z[ind_x, :][:, ind_y]
        grid_data = grid_data[ind_x, :][:, ind_y]
        grid_data -= np.nanmin(grid_data)
        grid_data /= np.nanmax(grid_data)
        grid_data[np.isnan(grid_data)] = 0

        if np.any(grid_data):
            # Find edges
            edges = canny(grid_data, sigma=sigma, use_quantiles=True)
            shape = edges.shape
            # Cycle through tiles of square size
            max_l = np.min([window_size, shape[0], shape[1]])
            half = np.floor(max_l / 2)
            overlap = 1.25

            n_cell_y = (shape[0] - 2 * half) * overlap / max_l
            n_cell_x = (shape[1] - 2 * half) * overlap / max_l

            if n_cell_x > 0:
                cnt_x = np.linspace(
                    half, shape[1] - half, 2 + int(np.round(n_cell_x)), dtype=int
                ).tolist()
                half_x = half
            else:
                cnt_x = [np.ceil(shape[1] / 2)]
                half_x = np.ceil(shape[1] / 2)

            if n_cell_y > 0:
                cnt_y = np.linspace(
                    half, shape[0] - half, 2 + int(np.round(n_cell_y)), dtype=int
                ).tolist()
                half_y = half
            else:
                cnt_y = [np.ceil(shape[0] / 2)]
                half_y = np.ceil(shape[0] / 2)

            coords = []
            for cx in cnt_x:
                for cy in cnt_y:

                    i_min, i_max = int(cy - half_y), int(cy + half_y)
                    j_min, j_max = int(cx - half_x), int(cx + half_x)
                    lines = probabilistic_hough_line(
                        edges[i_min:i_max, j_min:j_max],
                        line_length=line_length,
                        threshold=threshold,
                        line_gap=line_gap,
                        seed=0,
                    )

                    if np.any(lines):
                        coord = np.vstack(lines)
                        coords.append(
                            np.c_[
                                x[i_min:i_max, j_min:j_max][coord[:, 1], coord[:, 0]],
                                y[i_min:i_max, j_min:j_max][coord[:, 1], coord[:, 0]],
                                z[i_min:i_max, j_min:j_max][coord[:, 1], coord[:, 0]],
                            ]
                        )

            if coords:
                coord = np.vstack(coords)
                object_lines = coord
                indices = EdgeDetectionDriver.get_indices(
                    grid,
                    window_center_x,
                    window_center_y,
                    window_width,
                    window_height,
                    window_azimuth,
                    resolution,
                    object_lines,
                )
                xy = object_lines[indices, :2]
                if np.any(xy):
                    vertices = np.vstack(object_lines[indices, :])
                    cells = (
                        np.arange(vertices.shape[0]).astype("uint32").reshape((-1, 2))
                    )

            return vertices, cells

    @staticmethod
    def get_indices(
        grid: geoh5py.objects,
        window_center_x: float,
        window_center_y: float,
        window_width: float,
        window_height: float,
        window_azimuth: float,
        resolution: float,
        object_lines: list[float],
    ) -> list[bool]:
        """
        Get indices within window.

        :param grid: A Grid2D object.

        :param window_center_x: Easting position of the selection box.

        :param window_center_y: Northing position of the selection box.

        :param window_width: Width (m) of the selection box.

        :param window_height: Height (m) of the selection box.

        :param window_azimuth: Rotation angle of the selection box.

        :param resolution: Minimum data separation (m).

        :param object_lines: n x 3 array. Full list of edges.

        :return : n x 1 array. Indices within the window bounds.
        """
        # Fetch vertices in the project
        lim_x = [1e8, -1e8]
        lim_y = [1e8, -1e8]

        if isinstance(grid, Grid2D):
            lim_x[0], lim_x[1] = grid.centroids[:, 0].min(), grid.centroids[:, 0].max()
            lim_y[0], lim_y[1] = grid.centroids[:, 1].min(), grid.centroids[:, 1].max()
        else:
            return

        width = lim_x[1] - lim_x[0]
        height = lim_y[1] - lim_y[0]

        if window_center_x is None:
            window_center_x = np.mean(lim_x)
        if window_center_y is None:
            window_center_y = np.mean(lim_y)
        if window_width is None:
            window_width = width * 1.2
        if window_height is None:
            window_height = height * 1.2

        xy = object_lines
        indices_1 = filter_xy(
            xy[1::2, 0],
            xy[1::2, 1],
            resolution,
            window={
                "center": [
                    window_center_x,
                    window_center_y,
                ],
                "size": [
                    window_width,
                    window_height,
                ],
                "azimuth": window_azimuth,
            },
        )
        indices_2 = filter_xy(
            xy[::2, 0],
            xy[::2, 1],
            resolution,
            window={
                "center": [
                    window_center_x,
                    window_center_y,
                ],
                "size": [
                    window_width,
                    window_height,
                ],
                "azimuth": window_azimuth,
            },
        )

        indices = np.kron(
            np.any(np.c_[indices_1, indices_2], axis=1),
            np.ones(2),
        ).astype(bool)

        return indices


if __name__ == "__main__":
    print("Loading geoh5 file . . .")
    file = sys.argv[1]
    ifile = InputFile.read_ui_json(file)
    params_class = EdgeDetectionParams(ifile)
    driver = EdgeDetectionDriver(params_class)
    print("Loaded. Running edge detection . . .")
    with params_class.geoh5.open(mode="r+"):
        driver.run()
    print("Saved to " + ifile.path)
