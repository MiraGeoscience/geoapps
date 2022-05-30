#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


from __future__ import annotations

import sys

import numpy as np
from geoh5py.groups import ContainerGroup
from geoh5py.objects import Curve, Grid2D, Points, Surface
from geoh5py.ui_json import InputFile
from geoh5py.workspace import Workspace
from matplotlib import collections
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line

from geoapps.edge_detection.params import EdgeDetectionParams
from geoapps.utils.formatters import string_name
from geoapps.utils.utils import filter_xy


class EdgeDetectionDriver:
    def __init__(self, params: EdgeDetectionParams):
        self.params: EdgeDetectionParams = params
        self.collections = None
        self.trigger_vertices = None
        self.trigger_cells = None
        self._unique_object = {}
        self.indices = None
        self.object_lines = None

    def run(self):
        """ """

        grid = self.params.objects

        x = grid.centroids[:, 0].reshape(grid.shape, order="F")
        y = grid.centroids[:, 1].reshape(grid.shape, order="F")
        z = grid.centroids[:, 2].reshape(grid.shape, order="F")
        grid_data = self.params.data.values.reshape(grid.shape, order="F")

        indices = self.indices
        if indices is None:
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
            edges = canny(grid_data, sigma=self.params.sigma, use_quantiles=True)
            shape = edges.shape
            # Cycle through tiles of square size
            max_l = np.min([self.params.window_size, shape[0], shape[1]])
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
                        line_length=self.params.line_length,
                        threshold=self.params.threshold,
                        line_gap=self.params.line_gap,
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
                self.object_lines = coord
                self.plot_store_lines()
            else:
                self.object_lines = None

    def plot_store_lines(self):

        if hasattr(self.params, "resolution"):
            resolution = self.params.resolution
        else:
            resolution = 50

        # Fetch vertices in the project
        lim_x = [1e8, -1e8]
        lim_y = [1e8, -1e8]

        obj = self.params.objects
        if isinstance(obj, Grid2D):
            lim_x[0], lim_x[1] = obj.centroids[:, 0].min(), obj.centroids[:, 0].max()
            lim_y[0], lim_y[1] = obj.centroids[:, 1].min(), obj.centroids[:, 1].max()
        elif isinstance(obj, (Points, Curve, Surface)):
            lim_x[0], lim_x[1] = obj.vertices[:, 0].min(), obj.vertices[:, 0].max()
            lim_y[0], lim_y[1] = obj.vertices[:, 1].min(), obj.vertices[:, 1].max()
        else:
            return

        width = lim_x[1] - lim_x[0]
        height = lim_y[1] - lim_y[0]

        if self.params.window_center_x is None:
            self.params.window_center_x = np.mean(lim_x)
        if self.params.window_center_y is None:
            self.params.window_center_y = np.mean(lim_y)
        if self.params.window_width is None:
            self.params.window_width = (width * 1.2) / 2.0
        if self.params.window_height is None:
            self.params.window_height = (height * 1.2) / 2.0

        xy = self.object_lines
        indices_1 = filter_xy(
            xy[1::2, 0],
            xy[1::2, 1],
            resolution,
            window={
                "center": [
                    self.params.window_center_x,
                    self.params.window_center_y,
                ],
                "size": [
                    self.params.window_width,
                    self.params.window_height,
                ],
                "azimuth": self.params.window_azimuth,
            },
        )
        indices_2 = filter_xy(
            xy[::2, 0],
            xy[::2, 1],
            resolution,
            window={
                "center": [
                    self.params.window_center_x,
                    self.params.window_center_y,
                ],
                "size": [
                    self.params.window_width,
                    self.params.window_height,
                ],
                "azimuth": self.params.window_azimuth,
            },
        )

        indices = np.kron(
            np.any(np.c_[indices_1, indices_2], axis=1),
            np.ones(2),
        ).astype(bool)

        xy = self.object_lines[indices, :2]
        self.collections = [
            collections.LineCollection(
                np.reshape(xy, (-1, 2, 2)), colors="k", linewidths=2
            )
        ]

        if np.any(xy):
            vertices = np.vstack(self.object_lines[indices, :])
            cells = np.arange(vertices.shape[0]).astype("uint32").reshape((-1, 2))
            if np.any(cells):
                self.trigger_vertices = vertices
                self.trigger_cells = cells
        else:
            self.trigger_vertices = None
            self.trigger_cells = None

    def export(self, workspace):
        # entity, _ = self.get_selected_entities()
        entity = self.params.objects
        if self.trigger_vertices is not None:
            name = string_name(self.params.export_as)

            out_entity = ContainerGroup.create(
                workspace,
                name=self.params.ga_group_name,
                uid=self._unique_object.get(self.params.ga_group_name, None),
            )
            curve = Curve.create(
                workspace,
                name=name,
                vertices=self.trigger_vertices,
                cells=self.trigger_cells,
                parent=out_entity,
                uid=self._unique_object.get(name, None),
            )
            self._unique_object[name] = curve.uid
            self._unique_object[self.params.ga_group_name] = out_entity.uid


if __name__ == "__main__":
    file = sys.argv[1]
    ifile = InputFile.read_ui_json(file)
    params = EdgeDetectionParams(ifile)
    driver = EdgeDetectionDriver(params)
    driver.run()
    driver.export(ifile.workspace)
