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

from geoapps.base.selection import ObjectDataSelection
from geoapps.utils.plotting import format_axis, normalize
from geoapps.utils.utils import random_sampling, symlog

from geoapps.scatter_plot.params import ScatterPlotParams


class ScatterPlotDriver:
    def __init__(self, params: ScatterPlotParams):
        self.params: ScatterPlotParams = params

    def run(self):
        """
        Create an octree mesh from input values
        """

        x_active = self.params.x.enabled
        y_active = self.params.y.enabled
        z_active = self.params.z.enabled
        color_active = self.params.color.enabled
        size_active = self.params.size.enabled

        #if not self.refresh.value or self.indices is None:
        #    return None
        if (
                self.params.downsampling != self.n_values
                and self.indices.shape[0] == self.n_values
        ):
            return self.update_downsampling(None)

        if self.get_channel(self.params.size) is not None and size_active:
            vals = self.get_channel(self.params.size)[self.indices]
            inbound = (vals > self.params.size_min) * (vals < self.params.size_max)
            vals[~inbound] = np.nan
            size = normalize(vals)

            if self.params.size_log:
                size = symlog(size, self.params.size_thresh)
            size *= self.params.size_markers
        else:
            size = None

        if self.get_channel(self.params.color) is not None and color_active:
            vals = self.get_channel(self.params.color)[self.indices]
            inbound = (vals >= self.params.color_min) * (vals <= self.params.color_max)
            vals[~inbound] = np.nan
            color = normalize(vals)
            if self.params.color_log:
                color = symlog(color, self.params.color_thresh)
        else:
            color = "black"

        x_axis, y_axis, z_axis = None, None, None

        if np.sum([x_active, y_active, z_active]) == 3:

            if x_active:
                x_axis = self.get_channel(x)
                if x_axis is None:
                    x_active = False
                else:
                    x_axis = x_axis[self.indices]

            if y_active:
                y_axis = self.get_channel(y)
                if y_axis is None:
                    y_active = False
                else:
                    y_axis = y_axis[self.indices]

            if z_active:
                z_axis = self.get_channel(z)
                if z_axis is None:
                    z_active = False
                else:
                    z_axis = z_axis[self.indices]
            '''
            if np.sum([axis is not None for axis in [x_axis, y_axis, z_axis]]) < 2:
                self.figure.data = []
                return
            '''

            if x_axis is not None:
                inbound = (x_axis >= self.params.x_min) * (x_axis <= self.params.x_max)
                x_axis[~inbound] = np.nan
                x_axis, x_label, x_ticks, x_ticklabels = format_axis(
                    self.data.uid_name_map[x], x_axis, self.params.x_log, self.params.x_thresh
                )
            else:
                inbound = (z_axis >= self.params.z_min) * (z_axis <= self.params.z_max)
                z_axis[~inbound] = np.nan
                x_axis, x_label, x_ticks, x_ticklabels = format_axis(
                    self.data.uid_name_map[z], z_axis, self.params.z_log, self.params.z_thresh
                )

            if y_axis is not None:
                inbound = (y_axis >= y_min) * (y_axis <= y_max)
                y_axis[~inbound] = np.nan
                y_axis, y_label, y_ticks, y_ticklabels = format_axis(
                    self.data.uid_name_map[y], y_axis, self.params.y_log, self.params.y_thresh
                )
            else:
                inbound = (z_axis >= self.params.z_min) * (z_axis <= self.params.z_max)
                z_axis[~inbound] = np.nan
                y_axis, y_label, y_ticks, y_ticklabels = format_axis(
                    self.data.uid_name_map[z], z_axis, self.params.z_log, self.params.z_thresh
                )

            if z_axis is not None:
                inbound = (z_axis >= self.params.z_min) * (z_axis <= self.params.z_max)
                z_axis[~inbound] = np.nan
                z_axis, z_label, z_ticks, z_ticklabels = format_axis(
                    self.data.uid_name_map[z], z_axis, self.params.z_log, self.params.z_thresh
                )
            '''
            if self.custom_colormap:
                color_maps = self.custom_colormap
            '''

            # 3D Scatter
            if np.sum([x_active, y_active, z_active]) == 3:

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
                            # "ticktext": [f"{label:.2e}" for label in x_ticklabels],
                        },
                        "yaxis": {
                            "tickvals": y_ticks,
                            # "ticktext": [f"{label:.2e}" for label in y_ticklabels],
                        },
                        "zaxis": {
                            "tickvals": z_ticks,
                            # "ticktext": [f"{label:.2e}" for label in z_ticklabels],
                        },
                    },
                }
            # 2D Scatter
            else:
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
                        # "ticktext": [f"{label:.2e}" for label in x_ticklabels],
                        "exponentformat": "e",
                        "title": x_label,
                    },
                    "yaxis": {
                        "tickvals": y_ticks,
                        # "ticktext": [f"{label:.2e}" for label in y_ticklabels],
                        "exponentformat": "e",
                        "title": y_label,
                    },
                }

            self.figure.data = []
            self.figure.add_trace(plot)
            self.figure.update_layout(layout)

        else:
            self.figure.data = []

    def get_channel(self, channel):
        obj, _ = self.get_selected_entities()

        if channel is None:
            return None

        if channel not in self.data_channels.keys():

            if self.workspace.get_entity(channel):
                values = np.asarray(
                    self.workspace.get_entity(channel)[0].values, dtype=float
                ).copy()
            elif channel in "XYZ":
                # Check number of points
                if hasattr(obj, "centroids"):
                    values = obj.centroids[:, "XYZ".index(channel)]
                elif hasattr(obj, "vertices"):
                    values = obj.vertices[:, "XYZ".index(channel)]
            else:
                return

            self.data_channels[channel] = values

        return self.data_channels[channel].copy()

    def set_channel_bounds(self, name):
        """
        Set the min and max values for the given axis channel
        """

        channel = getattr(self, "_" + name).value
        self.get_channel(channel)

        if channel in self.data_channels.keys():

            values = self.data_channels[channel]
            values = values[~np.isnan(values)]

            cmin = getattr(self, "_" + name + "_min")
            cmin.value = f"{np.min(values):.2e}"
            cmax = getattr(self, "_" + name + "_max")
            cmax.value = f"{np.max(values):.2e}"

    def update_downsampling(self, _):

        if not list(self.data_channels.values()):
            return

        self.refresh.value = False
        values = []
        for axis in [self.x, self.y, self.z]:
            vals = self.get_channel(axis.value)
            if vals is not None:
                values.append(np.asarray(vals, dtype=float))

        if len(values) < 2:
            return

        values = np.vstack(values)
        nans = np.isnan(values)
        values[nans] = 0
        # Normalize all columns
        values = (values - np.min(values, axis=1)[:, None]) / (
            np.max(values, axis=1) - np.min(values, axis=1)
        )[:, None]
        values[nans] = np.nan
        self._indices = random_sampling(
            values.T,
            self.downsampling.value,
            bandwidth=2.0,
            rtol=1e0,
            method="histogram",
        )


if __name__ == "__main__":
    file = sys.argv[1]
    params = OctreeParams(InputFile.read_ui_json(file))
    driver = OctreeDriver(params)
    driver.run()
