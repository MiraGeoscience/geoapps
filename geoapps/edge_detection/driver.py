#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


from __future__ import annotations

import sys

import numpy as np
from geoh5py.ui_json import InputFile
from matplotlib import collections
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line

from geoapps.edge_detection.params import EdgeDetectionParams
from geoapps.utils.utils import filter_xy


class EdgeDetectionDriver:
    def __init__(self, params: EdgeDetectionParams):
        self.params: EdgeDetectionParams = params
        self.collections = None

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
                self.params.objects.lines = coord
                self.plot_store_lines()
            else:
                self.params.objects.lines = None

    def plot_store_lines(self):

        if hasattr(self.params, "resolution"):
            resolution = self.params.resolution.value
        else:
            resolution = 50

        xy = self.params.objects.lines
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

        xy = self.params.objects.lines[indices, :2]
        self.collections = [
            collections.LineCollection(
                np.reshape(xy, (-1, 2, 2)), colors="k", linewidths=2
            )
        ]

        if np.any(xy):
            vertices = np.vstack(self.params.objects.lines[indices, :])
            cells = np.arange(vertices.shape[0]).astype("uint32").reshape((-1, 2))
            if np.any(cells):
                self.trigger.vertices = vertices
                self.trigger.cells = cells
        else:
            self.trigger.vertices = None
            self.trigger.cells = None

    def export(self, _):
        """
        entity, _ = self.get_selected_entities()
        if getattr(self.trigger, "vertices", None) is not None:
            name = string_name(self.export_as.value)
            temp_geoh5 = f"{string_name(self.export_as.value)}_{time():.3f}.geoh5"
            with self.get_output_workspace(
                self.export_directory.selected_path, temp_geoh5
            ) as workspace:
                out_entity = ContainerGroup.create(
                    workspace,
                    name=self.ga_group_name.value,
                    uid=self._unique_object.get(self.ga_group_name.value, None),
                )
                curve = Curve.create(
                    workspace,
                    name=name,
                    vertices=self.trigger.vertices,
                    cells=self.trigger.cells,
                    parent=out_entity,
                    uid=self._unique_object.get(name, None),
                )
                self._unique_object[name] = curve.uid
                self._unique_object[self.ga_group_name.value] = out_entity.uid
        if self.live_link.value:
            monitored_directory_copy(self.export_directory.selected_path, out_entity)
        """


if __name__ == "__main__":
    file = sys.argv[1]
    params = EdgeDetectionParams(InputFile.read_ui_json(file))
    driver = EdgeDetectionDriver(params)
    driver.run()
    driver.export()
