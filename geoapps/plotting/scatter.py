#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import os

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from ipywidgets import (
    Checkbox,
    Dropdown,
    FloatText,
    HBox,
    IntSlider,
    Label,
    Layout,
    ToggleButton,
    VBox,
    interactive_output,
)

from geoapps.plotting import format_axis, normalize
from geoapps.selection import ObjectDataSelection
from geoapps.utils.utils import random_sampling, symlog


class ScatterPlots(ObjectDataSelection):
    """
    Application for 2D and 3D crossplots of data using symlog scaling
    """

    defaults = {
        "h5file": "../../assets/FlinFlon.geoh5",
        "objects": "geochem",
        "data": ["Al2O3", "CaO", "V", "MgO", "Ba"],
        "x": "Al2O3",
        "y": "CaO",
        "z": "Ba",
        "y_log": True,
        "z_log": True,
        "z_active": True,
        "color": "V",
        "color_active": True,
        "color_log": True,
        "size": "MgO",
        "size_active": True,
        "color_maps": "inferno",
        "refresh": True,
        "refresh_trigger": True,
    }

    def __init__(self, static=False, **kwargs):
        self.static = static
        self.select_multiple = True
        self._add_groups = True
        self.custom_colormap = []
        self._indices = None

        def channel_bounds_setter(caller):
            self.set_channel_bounds(caller["owner"].name)

        self._downsampling = IntSlider(
            description="Population Downsampling:",
            min=1,
            style={"description_width": "initial"},
            continuous_update=False,
        )

        self._x = Dropdown(description="Data:")
        self._x.observe(channel_bounds_setter, names="value")
        self._x.name = "x"
        self._x_active = Checkbox(description="Active", value=True, indent=False)
        self._x_log = Checkbox(
            description="Log10",
            value=False,
            indent=False,
        )
        self._x_thresh = FloatText(
            description="Threshold",
            value=1e-1,
        )
        self._x_min = FloatText(
            description="Min",
        )
        self._x_max = FloatText(
            description="Max",
        )
        self._x_panel = VBox(
            [
                self._x_active,
                HBox([self._x]),
                HBox([self._x_log, self._x_thresh]),
                HBox([self._x_min, self._x_max]),
            ]
        )

        self._y = Dropdown(description="Data:")
        self._y.observe(channel_bounds_setter, names="value")
        self._y.name = "y"
        self._y_active = Checkbox(
            description="Active",
            value=True,
            indent=False,
        )
        self._y_log = Checkbox(description="Log10", value=False, indent=False)
        self._y_thresh = FloatText(
            description="Threshold",
            value=1e-1,
        )
        self._y_min = FloatText(
            description="Min",
        )
        self._y_max = FloatText(
            description="Max",
        )
        self._y_panel = VBox(
            [
                self._y_active,
                HBox([self._y]),
                HBox([self._y_log, self._y_thresh]),
                HBox([self._y_min, self._y_max]),
            ]
        )

        self._z = Dropdown(description="Data:")
        self._z.observe(channel_bounds_setter, names="value")
        self._z.name = "z"
        self._z_active = Checkbox(
            description="Active",
            value=False,
            indent=False,
        )
        self._z_log = Checkbox(
            description="Log10",
            value=False,
            indent=False,
        )
        self._z_thresh = FloatText(
            description="Threshold",
            value=1e-1,
        )
        self._z_min = FloatText(
            description="Min",
        )
        self._z_max = FloatText(
            description="Max",
        )
        self._z_panel = VBox(
            [
                self._z_active,
                HBox([self._z]),
                HBox([self._z_log, self._z_thresh]),
                HBox([self._z_min, self._z_max]),
            ]
        )
        self._color = Dropdown(description="Data:")
        self._color.observe(channel_bounds_setter, names="value")
        self._color.name = "color"
        self._color_log = Checkbox(
            description="Log10",
            value=False,
            indent=False,
        )
        self._color_thresh = FloatText(
            description="Threshold",
            value=1e-1,
        )
        self._color_active = Checkbox(
            description="Active",
            value=False,
            indent=False,
        )
        self._color_maps = Dropdown(
            description="Colormaps",
            options=px.colors.named_colorscales(),
            value="viridis",
        )
        self._color_min = FloatText(
            description="Min",
        )
        self._color_max = FloatText(
            description="Max",
        )
        self._color_panel = VBox(
            [
                self._color_active,
                HBox([self._color]),
                self._color_maps,
                HBox([self._color_log, self._color_thresh]),
                HBox([self._color_min, self._color_max]),
            ]
        )

        self._size = Dropdown(description="Data:")
        self._size.observe(channel_bounds_setter, names="value")
        self._size.name = "size"
        self._size_active = Checkbox(
            description="Active",
            value=False,
            indent=False,
        )
        self._size_log = Checkbox(
            description="Log10",
            value=False,
            indent=False,
        )
        self._size_thresh = FloatText(
            description="Threshold",
            value=1e-1,
        )
        self._size_markers = IntSlider(
            min=1, max=100, value=20, description="Marker size", continuous_update=False
        )
        self._size_min = FloatText(
            description="Min",
        )
        self._size_max = FloatText(
            description="Max",
        )

        self._size_panel = VBox(
            [
                self._size_active,
                HBox([self._size]),
                self._size_markers,
                HBox([self._size_log, self._size_thresh]),
                HBox([self._size_min, self._size_max]),
            ]
        )
        self._refresh_trigger = ToggleButton(description="Refresh Plot", value=False)

        # Wrap all axis panels into dropdown
        def axes_pannels_trigger(_):
            self.axes_pannels_trigger()

        self.panels = {
            "X-axis": self._x_panel,
            "Y-axis": self._y_panel,
            "Z-axis": self._z_panel,
            "Color": self._color_panel,
            "Size": self._size_panel,
        }
        self.axes_pannels = Dropdown(
            options=["X-axis", "Y-axis", "Z-axis", "Color", "Size"],
            layout=Layout(width="300px"),
        )
        self.axes_pannels.observe(axes_pannels_trigger, names="value")
        self.axes_options = VBox([self.axes_pannels, self._x_panel])
        self.data_channels = {}
        self.data.observe(self.update_choices, names="value")
        self.objects.observe(self.update_objects, names="value")
        self.downsampling.observe(self.update_downsampling, names="value")

        super().__init__(**self.apply_defaults(**kwargs))

        self.figure = go.FigureWidget()

        self.crossplot = interactive_output(
            self.plot_selection,
            {
                "x": self.x,
                "x_log": self.x_log,
                "x_active": self.x_active,
                "x_thresh": self.x_thresh,
                "x_min": self.x_min,
                "x_max": self.x_max,
                "y": self.y,
                "y_log": self.y_log,
                "y_thresh": self.y_thresh,
                "y_active": self.y_active,
                "y_min": self.y_min,
                "y_max": self.y_max,
                "z": self.z,
                "z_log": self.z_log,
                "z_thresh": self.z_thresh,
                "z_active": self.z_active,
                "z_min": self.z_min,
                "z_max": self.z_max,
                "color": self.color,
                "color_log": self.color_log,
                "color_thresh": self.color_thresh,
                "color_active": self.color_active,
                "color_maps": self.color_maps,
                "color_min": self.color_min,
                "color_max": self.color_max,
                "size": self.size,
                "size_log": self.size_log,
                "size_thresh": self.size_thresh,
                "size_active": self.size_active,
                "size_markers": self.size_markers,
                "size_min": self.size_min,
                "size_max": self.size_max,
                "refresh_trigger": self.refresh_trigger,
            },
        )

        def write_html(_):
            self.write_html()

        self.trigger.on_click(write_html)
        self.trigger.description = "Save HTML"

        if self.static:
            self._main = VBox(
                [
                    self.project_panel,
                    HBox([self.objects, self.data]),
                    self.downsampling,
                    self.axes_options,
                    self.trigger,
                ]
            )
        else:
            self._main = VBox(
                [
                    self.project_panel,
                    HBox([self.objects, self.data]),
                    VBox([Label("Downsampling"), self.downsampling]),
                    self.axes_options,
                    self.trigger,
                    self.figure,
                ]
            )

    @property
    def n_values(self):
        """
        Number of values contained by the current object
        """

        obj, _ = self.get_selected_entities()
        if obj is not None:
            # Check number of points
            if hasattr(obj, "centroids"):
                return obj.n_cells
            elif hasattr(obj, "vertices"):
                return obj.n_vertices
        return None

    @property
    def indices(self):
        """
        Bool or array of int
        Indices of data to be plotted
        """
        if getattr(self, "_indices", None) is None:
            if self.n_values is not None:
                self._indices = np.arange(self.n_values)
            else:
                return None

        return self._indices

    @property
    def color(self):
        """
        :obj:`ipywidgets.Dropdown`
        """
        return self._color

    @property
    def color_log(self):
        """
        :obj:`ipywidgets.Checkbox`
        """
        return self._color_log

    @property
    def color_thresh(self):
        """
        :obj:`ipywidgets.FloatText`
        """
        return self._color_thresh

    @property
    def color_active(self):
        """
        :obj:`ipywidgets.Checkbox`
        """
        return self._color_active

    @property
    def color_maps(self):
        """
        :obj:`ipywidgets.Dropdown`
        """
        return self._color_maps

    @property
    def color_min(self):
        """
        :obj:`ipywidgets.Text`
        """
        return self._color_min

    @property
    def color_max(self):
        """
        :obj:`ipywidgets.Text`
        """
        return self._color_max

    @property
    def downsampling(self):
        """
        :obj:`ipywidgets.IntSlider`
        """
        return self._downsampling

    @property
    def refresh_trigger(self):
        """"""
        return self._refresh_trigger

    @property
    def size(self):
        """
        :obj:`ipywidgets.Dropdown`
        """
        return self._size

    @property
    def size_active(self):
        """
        :obj:`ipywidgets.Checkbox`
        """
        return self._size_active

    @property
    def size_log(self):
        """
        :obj:`ipywidgets.Checkbox`
        """
        return self._size_log

    @property
    def size_thresh(self):
        """
        :obj:`ipywidgets.FloatText`
        """
        return self._size_thresh

    @property
    def size_markers(self):
        """
        :obj:`ipywidgets.IntSlider`
        """
        return self._size_markers

    @property
    def size_min(self):
        """
        :obj:`ipywidgets.Text`
        """
        return self._size_min

    @property
    def size_max(self):
        """
        :obj:`ipywidgets.Text`
        """
        return self._size_max

    @property
    def x(self):
        """
        :obj:`ipywidgets.Dropdown`
        """
        return self._x

    @property
    def x_active(self):
        """
        :obj:`ipywidgets.Checkbox`
        """
        return self._x_active

    @property
    def x_log(self):
        """
        :obj:`ipywidgets.Checkbox`
        """
        return self._x_log

    @property
    def x_thresh(self):
        """
        :obj:`ipywidgets.FloatText`
        """
        return self._x_thresh

    @property
    def x_min(self):
        """
        :obj:`ipywidgets.Text`
        """
        return self._x_min

    @property
    def x_max(self):
        """
        :obj:`ipywidgets.Text`
        """
        return self._x_max

    @property
    def y(self):
        """
        :obj:`ipywidgets.Dropdown`
        """
        return self._y

    @property
    def y_active(self):
        """
        :obj:`ipywidgets.Checkbox`
        """
        return self._y_active

    @property
    def y_log(self):
        """
        :obj:`ipywidgets.Checkbox`
        """
        return self._y_log

    @property
    def y_thresh(self):
        """
        :obj:`ipywidgets.FloatText`
        """
        return self._y_thresh

    @property
    def y_min(self):
        """
        :obj:`ipywidgets.Text`
        """
        return self._y_min

    @property
    def y_max(self):
        """
        :obj:`ipywidgets.Text`
        """
        return self._y_max

    @property
    def z(self):
        """
        :obj:`ipywidgets.Dropdown`
        """
        return self._z

    @property
    def z_active(self):
        """
        :obj:`ipywidgets.Checkbox`
        """
        return self._z_active

    @property
    def z_log(self):
        """
        :obj:`ipywidgets.Checkbox`
        """
        return self._z_log

    @property
    def z_thresh(self):
        """
        :obj:`ipywidgets.FloatText`
        """
        return self._z_thresh

    @property
    def z_min(self):
        """
        :obj:`ipywidgets.Text`
        """
        return self._z_min

    @property
    def z_max(self):
        """
        :obj:`ipywidgets.Text`
        """
        return self._z_max

    def axes_pannels_trigger(self):
        self.axes_options.children = [
            self.axes_pannels,
            self.panels[self.axes_pannels.value],
        ]

    def get_channel(self, channel):
        obj, _ = self.get_selected_entities()

        if channel is None:
            return None

        if channel not in self.data_channels.keys():

            if obj.get_data(channel):
                values = np.asarray(obj.get_data(channel)[0].values, dtype=float).copy()
                values[(values > 1e-38) * (values < 2e-38)] = np.nan
            elif channel == "Z":
                # Check number of points
                if hasattr(obj, "centroids"):
                    values = obj.centroids[:, 2]
                elif hasattr(obj, "vertices"):
                    values = obj.vertices[:, 2]
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

    def plot_selection(
        self,
        x,
        x_log,
        x_active,
        x_thresh,
        x_min,
        x_max,
        y,
        y_log,
        y_active,
        y_thresh,
        y_min,
        y_max,
        z,
        z_log,
        z_active,
        z_thresh,
        z_min,
        z_max,
        color,
        color_log,
        color_active,
        color_thresh,
        color_maps,
        color_min,
        color_max,
        size,
        size_log,
        size_active,
        size_thresh,
        size_markers,
        size_min,
        size_max,
        refresh_trigger,
    ):

        if (
            not self.refresh_trigger.value
            or not self.refresh.value
            or self.indices is None
        ):
            return None

        if (
            self.downsampling.value != self.n_values
            and self.indices.shape[0] == self.n_values
        ):
            return self.update_downsampling(None)

        if self.get_channel(size) is not None and size_active:
            vals = self.get_channel(size)[self.indices]
            inbound = (vals > size_min) * (vals < size_max)
            vals[~inbound] = np.nan
            size = normalize(vals)

            if size_log:
                size = symlog(size, size_thresh)
            size *= size_markers
        else:
            size = None

        if self.get_channel(color) is not None and color_active:
            vals = self.get_channel(color)[self.indices]
            inbound = (vals >= color_min) * (vals <= color_max)
            vals[~inbound] = np.nan
            color = normalize(vals)
            if color_log:
                color = symlog(color, color_thresh)
        else:
            color = "black"

        x_axis, y_axis, z_axis = None, None, None

        if np.sum([x_active, y_active, z_active]) > 1:

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

            if np.sum([axis is not None for axis in [x_axis, y_axis, z_axis]]) < 2:
                self.figure.data = []
                return

            if x_axis is not None:
                inbound = (x_axis >= x_min) * (x_axis <= x_max)
                x_axis[~inbound] = np.nan
                x_axis, x_label, x_ticks, x_ticklabels = format_axis(
                    x, x_axis, x_log, x_thresh
                )
            else:
                inbound = (z_axis >= z_min) * (z_axis <= z_max)
                z_axis[~inbound] = np.nan
                x_axis, x_label, x_ticks, x_ticklabels = format_axis(
                    z, z_axis, z_log, z_thresh
                )

            if y_axis is not None:
                inbound = (y_axis >= y_min) * (y_axis <= y_max)
                y_axis[~inbound] = np.nan
                y_axis, y_label, y_ticks, y_ticklabels = format_axis(
                    y, y_axis, y_log, y_thresh
                )
            else:
                inbound = (z_axis >= z_min) * (z_axis <= z_max)
                z_axis[~inbound] = np.nan
                y_axis, y_label, y_ticks, y_ticklabels = format_axis(
                    z, z_axis, z_log, z_thresh
                )

            if z_axis is not None:
                inbound = (z_axis >= z_min) * (z_axis <= z_max)
                z_axis[~inbound] = np.nan
                z_axis, z_label, z_ticks, z_ticklabels = format_axis(
                    z, z_axis, z_log, z_thresh
                )

            if self.custom_colormap:
                color_maps = self.custom_colormap

            # 3D Scatter
            if np.sum([x_active, y_active, z_active]) == 3:

                plot = go.Scatter3d(
                    x=x_axis,
                    y=y_axis,
                    z=z_axis,
                    mode="markers",
                    marker={"color": color, "size": size, "colorscale": color_maps},
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
                    marker={"color": color, "size": size, "colorscale": color_maps},
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

    def update_axes(self, refresh_plot=True):
        self.refresh_trigger.value = False
        for name in [
            "x",
            "y",
            "z",
            "color",
            "size",
        ]:
            widget = getattr(self, "_" + name)
            val = widget.value
            widget.options = list(self.data_channels.keys())

            if val in widget.options:
                widget.value = val
            else:
                widget.value = None
        if refresh_plot:
            self.refresh_trigger.value = True

    def update_choices(self, _):
        self.refresh_trigger.value = False

        for channel in self.data.value:
            self.get_channel(channel)

        keys = list(self.data_channels.keys())
        for key in keys:
            if key not in self.data.value:
                del self.data_channels[key]

        self.update_axes(refresh_plot=False)

        if self.downsampling.value != self.n_values:
            self.update_downsampling(None, refresh_plot=False)

        self.refresh_trigger.value = True

    def update_downsampling(self, _, refresh_plot=True):

        if not list(self.data_channels.values()):
            return

        self.refresh_trigger.value = False
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
        self.refresh_trigger.value = refresh_plot

    def update_objects(self, _):
        self.data_channels = {}
        self.downsampling.max = self.n_values
        self.downsampling.value = np.min([5000, self.n_values])
        self._indices = None
        self.update_downsampling(None, refresh_plot=False)

    def write_html(self):
        self.figure.write_html(
            os.path.join(
                os.path.abspath(os.path.dirname(self.h5file)), "Crossplot.html"
            )
        )
