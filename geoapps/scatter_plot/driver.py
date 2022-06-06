#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


from __future__ import annotations

import sys

import numpy as np
import plotly.graph_objects as go
from geoh5py.ui_json import InputFile

from geoapps.scatter_plot.params import ScatterPlotParams
from geoapps.utils.plotting import format_axis, normalize
from geoapps.utils.utils import random_sampling, symlog


class ScatterPlotDriver:
    def __init__(self, params: ScatterPlotParams):
        self.params: ScatterPlotParams = params

    def run(self):
        """
        Create a scatter plot from input values
        """
        figure = go.Figure()

        if (self.params.x is not None) & (self.params.y is not None):

            if (self.params.downsampling == 100) | self.params.downsampling is None:
                indices = np.full(len(self.params.x.values), True)
            else:
                indices = self.get_indices()

            size = None
            if self.params.size is not None:
                vals = self.params.size.values[indices]
                min = self.params.size_min
                max = self.params.size_max
                if min is None:
                    min = np.nanmin(vals)
                if max is None:
                    max = np.nanmax(vals)
                inbound = (vals > min) * (vals < max)
                if np.sum(inbound) > 0:
                    vals[~inbound] = np.nan
                    size = normalize(vals)
                    if self.params.size_log:
                        size = symlog(size, self.params.size_thresh)
                    if self.params.size_markers is not None:
                        size *= self.params.size_markers

            color = "black"
            if self.params.color is not None:
                vals = self.params.color.values[indices]
                min = self.params.color_min
                max = self.params.color_max
                if min is None:
                    min = np.nanmin(vals)
                if max is None:
                    max = np.nanmax(vals)
                inbound = (vals > min) * (vals < max)
                if np.sum(inbound) > 0:
                    vals[~inbound] = np.nan
                    color = normalize(vals)
                    if self.params.color_log:
                        color = symlog(color, self.params.color_thresh)

            x_axis = self.params.x.values[indices]
            y_axis = self.params.y.values[indices]

            min = self.params.x_min
            max = self.params.x_max
            if min is None:
                min = np.nanmin(x_axis)
            if max is None:
                max = np.nanmax(x_axis)
            inbound = (x_axis >= min) * (x_axis <= max)
            if np.sum(inbound) > 0:
                x_axis[~inbound] = np.nan
                x_axis, x_label, x_ticks, x_ticklabels = format_axis(
                    self.params.x.name, x_axis, self.params.x_log, self.params.x_thresh
                )
            else:
                return figure

            min = self.params.y_min
            max = self.params.y_max
            if min is None:
                min = np.nanmin(y_axis)
            if max is None:
                max = np.nanmax(y_axis)
            inbound = (y_axis >= min) * (y_axis <= max)
            if np.sum(inbound) > 0:
                y_axis[~inbound] = np.nan
                y_axis, y_label, y_ticks, y_ticklabels = format_axis(
                    self.params.y.name, y_axis, self.params.y_log, self.params.y_thresh
                )
            else:
                return figure

            if self.params.z is not None:
                z_axis = self.params.z.values[indices]

                min = self.params.z_min
                max = self.params.z_max
                if min is None:
                    min = np.nanmin(z_axis)
                if max is None:
                    max = np.nanmax(z_axis)
                inbound = (z_axis >= min) * (z_axis <= max)
                if np.sum(inbound) > 0:
                    z_axis[~inbound] = np.nan
                    z_axis, z_label, z_ticks, z_ticklabels = format_axis(
                        self.params.z.name,
                        z_axis,
                        self.params.z_log,
                        self.params.z_thresh,
                    )
                else:
                    return figure

                # 3D Scatter
                plot = go.Scatter3d(
                    x=x_axis,
                    y=y_axis,
                    z=z_axis,
                    mode="markers",
                    marker={
                        "color": color,
                        "size": size,
                        "colorscale": self.params.color_maps,
                    },
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
                    marker={
                        "color": color,
                        "size": size,
                        "colorscale": self.params.color_maps,
                    },
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

        return figure

    def get_indices(self):

        values = []
        non_nan = []
        for axis in [self.params.x, self.params.y, self.params.z]:
            if axis is not None:
                values.append(np.asarray(axis.values, dtype=float))
                non_nan.append(~np.isnan(axis.values))

        values = np.vstack(values)
        non_nan = np.vstack(non_nan)

        # Normalize all columns
        values = (values - np.nanmin(values, axis=1)[:, None]) / (
            np.nanmax(values, axis=1) - np.nanmin(values, axis=1)
        )[:, None]

        percent = self.params.downsampling / 100

        # Number of values that are not nan along all three axes
        size = np.sum(np.all(non_nan, axis=0))

        indices = random_sampling(
            values.T,
            int(percent * size),
            bandwidth=2.0,
            rtol=1e0,
            method="histogram",
        )

        return indices


if __name__ == "__main__":
    print("Loading geoh5 file . . .")
    file = sys.argv[1]
    ifile = InputFile.read_ui_json(file)
    params = ScatterPlotParams(ifile)
    driver = ScatterPlotDriver(params)
    print("Loaded. Building the plotly scatterplot . . .")
    figure = driver.run()
    figure.show()
    if params.save:
        figure.write_html(ifile.path + "/Crossplot.html")
        print("Figure saved to " + ifile.path + "/Crossplot.html")
    print("Done")
