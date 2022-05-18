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

import plotly.graph_objects as go

from geoh5py.ui_json import InputFile

from geoapps.base.selection import ObjectDataSelection
from geoapps.utils.plotting import format_axis, normalize
from geoapps.utils.utils import random_sampling, symlog

from geoapps.scatter_plot.params import ScatterPlotParams


class ScatterPlotDriver:
    def __init__(self, params: ScatterPlotParams):
        self.params: ScatterPlotParams = params

    def run(self):
        """
        Create a scatter plot from input values
        """
        figure = go.Figure()

        x_axis, y_axis, z_axis = None, None, None

        if (self.params.x is not None) & (self.params.y is not None):

            indices = self.get_indices()

            if self.params.size is not None:
                vals = self.params.size.values[indices]
                min = self.params.size_min
                max = self.params.size_max
                if min is None:
                    min = np.nanmin(vals)
                if max is None:
                    max = np.nanmax(vals)
                inbound = (vals > min) * (vals < max)
                vals[~inbound] = np.nan
                size = normalize(vals)

                if self.params.size_log:
                    size = symlog(size, self.params.size_thresh)
                size *= self.params.size_markers
            else:
                size = None

            if self.params.color is not None:
                vals = self.params.color.values[indices]
                min = self.params.color_min
                max = self.params.color_max
                if min is None:
                    min = np.nanmin(vals)
                if max is None:
                    max = np.nanmax(vals)
                inbound = (vals > min) * (vals < max)
                vals[~inbound] = np.nan
                color = normalize(vals)

                if self.params.color_log:
                    color = symlog(color, self.params.color_thresh)
            else:
                color = "black"

            x_axis = self.params.x.values[indices]
            y_axis = self.params.y.values[indices]

            min = self.params.x_min
            max = self.params.x_max
            if min is None:
                min = np.nanmin(x_axis)
            if max is None:
                max = np.nanmax(x_axis)
            inbound = (x_axis >= min) * (x_axis <= max)
            x_axis[~inbound] = np.nan
            x_axis, x_label, x_ticks, x_ticklabels = format_axis(
                "label x", x_axis, self.params.x_log, self.params.x_thresh
            )

            min = self.params.y_min
            max = self.params.y_max
            if min is None:
                min = np.nanmin(y_axis)
            if max is None:
                max = np.nanmax(y_axis)
            inbound = (y_axis >= min) * (y_axis <= max)
            y_axis[~inbound] = np.nan
            y_axis, y_label, y_ticks, y_ticklabels = format_axis(
                "label y", y_axis, self.params.y_log, self.params.y_thresh
            )

            if self.params.z is not None:
                z_axis = self.params.z.values[indices]

                min = self.params.z_min
                max = self.params.z_max
                if min is None:
                    min = np.nanmin(z_axis)
                if max is None:
                    max = np.nanmax(z_axis)
                inbound = (z_axis >= min) * (z_axis <= max)
                z_axis[~inbound] = np.nan
                z_axis, z_label, z_ticks, z_ticklabels = format_axis(
                    "label z", z_axis, self.params.z_log, self.params.z_thresh
                )

                # 3D Scatter

                plot = go.Scatter3d(
                    x=x_axis,
                    y=y_axis,
                    z=z_axis,
                    mode="markers",
                    marker={"color": color, "size": size, "colorscale": self.params.color_maps},
                )

                layout = {
                    "margin": dict(l=0, r=0, b=0, t=0),
                    "scene": {
                        "xaxis_title": x_label,
                        "yaxis_title": y_label,
                        "zaxis_title": z_label,
                        "xaxis": {
                            "tickvals": x_ticks,
                        },
                        "yaxis": {
                            "tickvals": y_ticks,
                        },
                        "zaxis": {
                            "tickvals": z_ticks,
                        },
                    },
                }

            else:
            # 2D Scatter
                plot = go.Scatter(
                    x=x_axis,
                    y=y_axis,
                    mode="markers",
                    marker={"color": color, "size": size, "colorscale": self.params.color_maps},
                )

                layout = {
                    "margin": dict(l=0, r=0, b=0, t=0),
                    "xaxis": {
                        "tickvals": x_ticks,
                        "exponentformat": "e",
                        "title": x_label,
                    },
                    "yaxis": {
                        "tickvals": y_ticks,
                        "exponentformat": "e",
                        "title": y_label,
                    },
                }

            figure.add_trace(plot)
            figure.update_layout(layout)

        figure.show()
        #figure.write_html("path")

    def get_indices(self):

        values = []
        for axis in [self.params.x, self.params.y, self.params.z]:
            if axis is not None:
                values.append(np.asarray(axis.values, dtype=float))

        values = np.vstack(values)
        # Normalize all columns
        values = (values - np.nanmin(values, axis=1)[:, None]) / (
            np.nanmax(values, axis=1) - np.nanmin(values, axis=1)
        )[:, None]

        if self.params.downsampling is None:
            percent = 1
        else:
            percent = self.params.downsampling/100

        indices = random_sampling(
            values.T,
            int(percent * np.size(values, 1)),
            bandwidth=2.0,
            rtol=1e0,
            method="histogram",
        )

        return indices


if __name__ == "__main__":
    file = sys.argv[1]
    params = ScatterPlotParams(InputFile.read_ui_json(file))
    driver = ScatterPlotDriver(params)
    driver.run()
