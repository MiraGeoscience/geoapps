import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from ipywidgets import (
    Dropdown,
    Checkbox,
    IntSlider,
    FloatText,
    VBox,
    HBox,
    ToggleButton,
    interactive_output,
    Layout,
)
from geoapps.utils import symlog
from geoapps.selection import ObjectDataSelection
from geoapps.plotting import normalize, format_axis


class ScatterPlots(ObjectDataSelection):
    """
    Application for 2D and 3D crossplots of data using symlog scaling
    """

    defaults = {
        "select_multiple": True,
        "add_groups": True,
        "h5file": "../../assets/FlinFlon.geoh5",
        "objects": "geochem",
        "data": ["Al2O3", "CaO", "V", "MgO"],
        "x": "Al2O3",
        "y": "CaO",
        "y_log": True,
        "color": "V",
        "color_active": True,
        "color_log": True,
        "size": "MgO",
        "size_active": True,
        "color_maps": "inferno",
    }

    def __init__(self, **kwargs):

        self._select_multiple = True

        def channel_bounds_setter(caller):
            self.set_channel_bounds(caller["owner"].name)

        self._x = Dropdown()
        self._x.observe(channel_bounds_setter, names="value")
        self._x.name = "x"
        self._x_active = Checkbox(description="Active", value=True, indent=False)
        self._x_log = Checkbox(description="Log10", value=False, indent=False,)
        self._x_thresh = FloatText(description="Threshold", value=1e-1, indent=False,)
        self._x_min = FloatText(description="Min", indent=False,)
        self._x_max = FloatText(description="Max", indent=False,)
        self._x_panel = VBox(
            [
                self._x_active,
                HBox([self._x]),
                HBox([self._x_log, self._x_thresh]),
                HBox([self._x_min, self._x_max]),
            ]
        )

        self._y = Dropdown()
        self._y.observe(channel_bounds_setter, names="value")
        self._y.name = "y"
        self._y_active = Checkbox(description="Active", value=True, indent=False,)
        self._y_log = Checkbox(description="Log10", value=False, indent=False)
        self._y_thresh = FloatText(description="Threshold", value=1e-1, indent=False,)
        self._y_min = FloatText(description="Min", indent=False,)
        self._y_max = FloatText(description="Max", indent=False,)
        self._y_panel = VBox(
            [
                self._y_active,
                HBox([self._y]),
                HBox([self._y_log, self._y_thresh]),
                HBox([self._y_min, self._y_max]),
            ]
        )

        self._z = Dropdown()
        self._z.observe(channel_bounds_setter, names="value")
        self._z.name = "z"
        self._z_active = Checkbox(description="Active", value=False, indent=False,)
        self._z_log = Checkbox(description="Log10", value=False, indent=False,)
        self._z_thresh = FloatText(description="Threshold", value=1e-1, indent=False,)
        self._z_min = FloatText(description="Min", indent=False,)
        self._z_max = FloatText(description="Max", indent=False,)
        self._z_panel = VBox(
            [
                self._z_active,
                HBox([self._z]),
                HBox([self._z_log, self._z_thresh]),
                HBox([self._z_min, self._z_max]),
            ]
        )

        self._color = Dropdown()
        self._color.observe(channel_bounds_setter, names="value")
        self._color.name = "color"
        self._color_log = Checkbox(description="Log10", value=False, indent=False,)
        self._color_thresh = FloatText(
            description="Threshold", value=1e-1, indent=False,
        )
        self._color_active = Checkbox(description="Active", value=False, indent=False,)
        self._color_maps = Dropdown(
            description="Colormaps",
            options=px.colors.named_colorscales(),
            value="viridis",
        )
        self._color_min = FloatText(description="Min", indent=False,)
        self._color_max = FloatText(description="Max", indent=False,)
        self._color_panel = VBox(
            [
                self._color_active,
                HBox([self._color]),
                self._color_maps,
                HBox([self._color_log, self._color_thresh]),
                HBox([self._color_min, self._color_max]),
            ]
        )

        self._size = Dropdown()
        self._size.observe(channel_bounds_setter, names="value")
        self._size.name = "size"
        self._size_active = Checkbox(description="Active", value=False, indent=False,)
        self._size_log = Checkbox(description="Log10", value=False, indent=False,)
        self._size_thresh = FloatText(
            description="Threshold", value=1e-1, indent=False,
        )
        self._size_markers = IntSlider(
            min=1, max=100, value=20, description="Marker size", continuous_update=False
        )
        self._size_min = FloatText(description="Min", indent=False,)
        self._size_max = FloatText(description="Max", indent=False,)
        self._size_panel = VBox(
            [
                self._size_active,
                HBox([self._size]),
                self._size_markers,
                HBox([self._size_log, self._size_thresh]),
                HBox([self._size_min, self._size_max]),
            ]
        )

        self.refresh_trigger = ToggleButton(description="Refresh Plot", value=True)

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

        self.axes_pannels.observe(axes_pannels_trigger)
        self.axes_options = HBox([self.axes_pannels, self._x_panel])
        self.data_channels = {}
        self.crossplot_fig = go.FigureWidget()

        def update_choices(_):
            self.update_choices()

        self.data.observe(update_choices, names="value")

        def update_data_dict(_):
            self.update_data_dict()

        self.objects.observe(update_data_dict, names="value")

        super().__init__(**self.apply_defaults(**kwargs))

        def plot_selection(
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
            self.plot_selection(
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
            )

        self.crossplot = interactive_output(
            plot_selection,
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

        self._widget = VBox(
            [
                self.project_panel,
                VBox([HBox([self.objects, self.data]), self.axes_options,]),
                self.crossplot,
                self.crossplot_fig,
            ]
        )

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

    def axes_pannels_trigger(self):
        self.axes_options.children = [
            self.axes_pannels,
            self.panels[self.axes_pannels.value],
        ]

    def get_channel(self, channel):
        obj, _ = self.get_selected_entities()

        if channel is None:
            return None

        if channel not in self.data_channels.keys() and obj.get_data(channel):
            values = obj.get_data(channel)[0].values.copy()
            values[(values > 1e-38) * (values < 2e-38)] = np.nan
            self.data_channels[channel] = values
        return self.data_channels[channel].copy()

    def set_channel_bounds(self, name):
        """
        Set the min and max values for the given axis channel
        """
        self.refresh.value = False
        channel = getattr(self, "_" + name).value
        self.get_channel(channel)

        if channel in self.data_channels.keys():

            values = self.data_channels[channel]
            values = values[~np.isnan(values)]

            cmin = getattr(self, "_" + name + "_min")
            cmin.value = f"{np.min(values):.2e}"
            cmax = getattr(self, "_" + name + "_max")
            cmax.value = f"{np.max(values):.2e}"
        self.refresh.value = True

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

        if not refresh_trigger or not self.refresh.value:
            return None

        if self.get_channel(size) is not None and size_active:
            vals = self.get_channel(size)
            inbound = (vals > size_min) * (vals < size_max)
            vals[~inbound] = np.nan
            size = normalize(vals)

            if size_log:
                size = symlog(size, size_thresh)
            size *= size_markers
        else:
            size = None

        if self.get_channel(color) is not None and color_active:
            vals = self.get_channel(color)
            inbound = (vals > color_min) * (vals < color_max)
            vals[~inbound] = np.nan
            color = normalize(vals)
            if color_log:
                color = symlog(color, color_thresh)
        else:
            color = "black"

        self.crossplot_fig.data = []
        x_axis, y_axis, z_axis = None, None, None

        if np.sum([x_active, y_active, z_active]) > 1:

            if x_active:
                x_axis = self.get_channel(x)
            else:
                x_axis = self.get_channel(y)

            if y_active:
                y_axis = self.get_channel(y)
            else:
                y_axis = self.get_channel(z)

            if z_active:
                z_axis = self.get_channel(z)

            if x_axis is None or y_axis is None:
                return

            if x_axis is not None:
                inbound = (x_axis > x_min) * (x_axis < x_max)
                x_axis[~inbound] = np.nan
                x_axis, x_label, x_ticks, x_ticklabels = format_axis(
                    x, x_axis, x_log, x_thresh
                )

            if y_axis is not None:
                inbound = (y_axis > y_min) * (y_axis < y_max)
                y_axis[~inbound] = np.nan
                y_axis, y_label, y_ticks, y_ticklabels = format_axis(
                    y, y_axis, y_log, y_thresh
                )

            if z_axis is not None:
                inbound = (z_axis > z_min) * (z_axis < z_max)
                z_axis[~inbound] = np.nan
                z_axis, z_label, z_ticks, z_ticklabels = format_axis(
                    z, z_axis, z_log, z_thresh
                )

            # 3D Scatter
            if np.sum([x_active, y_active, z_active]) == 3:
                self.crossplot_fig.add_trace(
                    go.Scatter3d(
                        x=x_axis,
                        y=y_axis,
                        z=z_axis,
                        mode="markers",
                        marker={"color": color, "size": size, "colorscale": color_maps},
                    )
                )

                self.crossplot_fig.update_layout(
                    scene={
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
                    }
                )
            # 2D Scatter
            else:
                self.crossplot_fig.add_trace(
                    go.Scatter(
                        x=x_axis,
                        y=y_axis,
                        mode="markers",
                        marker={"color": color, "size": size, "colorscale": color_maps},
                    )
                )
                self.crossplot_fig.update_layout(
                    margin=dict(l=0, r=0, b=0, t=0),
                    xaxis={
                        "tickvals": x_ticks,
                        # "ticktext": [f"{label:.2e}" for label in x_ticklabels],
                        "exponentformat": "e",
                        "title": x_label,
                    },
                    yaxis={
                        "tickvals": y_ticks,
                        # "ticktext": [f"{label:.2e}" for label in y_ticklabels],
                        "exponentformat": "e",
                        "title": y_label,
                    },
                )
        else:
            self.crossplot_fig.data = []

    def update_choices(self):
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
            widget.options = self.data.value
            if val in widget.options:
                widget.value = val
            else:
                widget.value = None

        self.refresh_trigger.value = True

    def update_data_dict(self):
        self.data_channels = {}
