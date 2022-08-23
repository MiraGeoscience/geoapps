#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from geoapps.scatter_plot.params import ScatterPlotParams
from geoapps.utils.plotting import format_axis, normalize, symlog
from geoapps.utils.statistics import random_sampling


class ScatterPlotDriver:
    def __init__(self, params: ScatterPlotParams):
        self.params: ScatterPlotParams = params

    def run(self):
        """
        Create a scatter plot from input values
        """
        figure = go.Figure()

        if (self.params.x is not None) & (self.params.y is not None):

            if self.params.downsampling is None:
                self.params.downsampling = 100
            indices = self.get_indices()

            size = None
            if self.params.size is not None:
                vals = self.params.size.values[indices]
                size_min = self.params.size_min
                size_max = self.params.size_max
                if size_min is None:
                    size_min = np.nanmin(vals)
                if size_max is None:
                    size_max = np.nanmax(vals)
                inbound = (vals >= size_min) * (vals <= size_max)
                if np.sum(inbound) > 0:
                    vals[~inbound] = np.nan
                    size = normalize(vals)
                    if self.params.size_log:
                        size = symlog(size, self.params.size_thresh)
                    if self.params.size_markers is not None:
                        size *= self.params.size_markers

            color = "black"
            if self.params.color is not None:
                if self.params.color.name == "kmeans":
                    color = self.params.color.values
                else:
                    vals = self.params.color.values[indices]
                    color_min = self.params.color_min
                    color_max = self.params.color_max
                    if color_min is None:
                        color_min = np.nanmin(vals)
                    if color_max is None:
                        color_max = np.nanmax(vals)
                    inbound = (vals >= color_min) * (vals <= color_max)
                    if np.sum(inbound) > 0:
                        vals[~inbound] = np.nan
                        color = normalize(vals)
                        if self.params.color_log:
                            color = symlog(color, self.params.color_thresh)

            x_axis = self.params.x.values[indices]
            y_axis = self.params.y.values[indices]

            x_min = self.params.x_min
            x_max = self.params.x_max
            if x_min is None:
                x_min = np.nanmin(x_axis)
            if x_max is None:
                x_max = np.nanmax(x_axis)
            inbound = (x_axis >= x_min) * (x_axis <= x_max)
            if np.sum(inbound) > 0:
                x_axis[~inbound] = np.nan
                x_axis, x_label, x_ticks, _ = format_axis(
                    self.params.x.name, x_axis, self.params.x_log, self.params.x_thresh
                )
            else:
                return figure

            y_min = self.params.y_min
            y_max = self.params.y_max
            if y_min is None:
                y_min = np.nanmin(y_axis)
            if y_max is None:
                y_max = np.nanmax(y_axis)
            inbound = (y_axis >= y_min) * (y_axis <= y_max)
            if np.sum(inbound) > 0:
                y_axis[~inbound] = np.nan
                y_axis, y_label, y_ticks, _ = format_axis(
                    self.params.y.name, y_axis, self.params.y_log, self.params.y_thresh
                )
            else:
                return figure

            if self.params.z is not None:
                z_axis = self.params.z.values[indices]

                z_min = self.params.z_min
                z_max = self.params.z_max
                if z_min is None:
                    z_min = np.nanmin(z_axis)
                if z_max is None:
                    z_max = np.nanmax(z_axis)
                inbound = (z_axis >= z_min) * (z_axis <= z_max)
                if np.sum(inbound) > 0:
                    z_axis[~inbound] = np.nan
                    z_axis, z_label, z_ticks, _ = format_axis(
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
                        "line_width": 0,
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
                        "line_width": 0,
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

    def get_indices(self) -> np.ndarray:
        """
        Get indices of data to plot after downsampling.

        :return indices:
        """
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
        n_values = np.min([int(percent * size), 5000])

        indices = random_sampling(
            values.T,
            n_values,
            bandwidth=2.0,
            rtol=1e0,
            method="histogram",
        )
        return indices
