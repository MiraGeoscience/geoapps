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
from geoh5py.objects import Curve, Points, Surface
from geoh5py.ui_json import InputFile
from matplotlib.pyplot import axes
from scipy.interpolate import LinearNDInterpolator

from geoapps.contours.params import ContoursParams
from geoapps.utils.formatters import string_name
from geoapps.utils.plotting import plot_plan_data_selection
from geoapps.utils.utils import get_contours


class ContoursDriver:
    def __init__(self, params: ContoursParams):
        self.params: ContoursParams = params
        self._unique_object = {}

    def run(self):
        entity = self.params.objects
        data = self.params.data

        _, _, _, _, contour_set = plot_plan_data_selection(
            entity,
            data,
            **{
                "axis": axes(),
                "resolution": self.params.resolution,
                "window": {
                    "center": [
                        self.params.window_center_x,
                        self.params.window_center_y,
                    ],
                    "size": [self.params.window_width, self.params.window_height],
                    "azimuth": self.params.window_azimuth,
                },
                "contours": get_contours(
                    self.params.interval_min,
                    self.params.interval_max,
                    self.params.interval_spacing,
                    self.params.fixed_contours,
                ),
            },
        )

        if contour_set is not None:
            vertices, cells, values = [], [], []
            count = 0
            for segs, level in zip(contour_set.allsegs, contour_set.levels):
                for poly in segs:
                    n_v = len(poly)
                    vertices.append(poly)
                    cells.append(
                        np.c_[
                            np.arange(count, count + n_v - 1),
                            np.arange(count + 1, count + n_v),
                        ]
                    )
                    values.append(np.ones(n_v) * level)
                    count += n_v
            if vertices:
                vertices = np.vstack(vertices)
                if self.params.z_value:
                    vertices = np.c_[vertices, np.hstack(values)]
                else:
                    if isinstance(entity, (Points, Curve, Surface)):
                        z_interp = LinearNDInterpolator(
                            entity.vertices[:, :2], entity.vertices[:, 2]
                        )
                        vertices = np.c_[vertices, z_interp(vertices)]
                    else:
                        vertices = np.c_[
                            vertices,
                            np.ones(vertices.shape[0]) * entity.origin["z"],
                        ]

            curve = Curve.create(
                self.params.geoh5,
                name=string_name(self.params.export_as),
                vertices=vertices,
                cells=np.vstack(cells).astype("uint32"),
            )
            out_entity = curve
            if len(self.params.ga_group_name) > 0:
                out_entity = ContainerGroup.create(
                    self.params.geoh5, name=string_name(self.params.ga_group_name)
                )
                curve.parent = out_entity

            curve.add_data(
                {
                    self.get_contour_string(
                        self.params.interval_min,
                        self.params.interval_max,
                        self.params.interval_spacing,
                        self.params.fixed_contours,
                    ): {"values": np.hstack(values)}
                }
            )

    @staticmethod
    def get_contour_string(min, max, step, fixed_contours):
        contour_string = (
            str(min)
            + ":"
            + str(max)
            + ":"
            + str(step)
            + ","
            + str(fixed_contours.replace(" ", ""))
        )
        return contour_string


if __name__ == "__main__":
    print("Loading geoh5 file . . .")
    file = sys.argv[1]
    ifile = InputFile.read_ui_json(file)
    params = ContoursParams(ifile)
    driver = ContoursParams(params)
    print("Loaded. Running edge detection . . .")
    driver.run()
    print("Saved to " + ifile.path)
