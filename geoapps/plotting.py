import re
import ipywidgets as widgets
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from copy import copy
from geoh5py.objects import Curve, Grid2D, Points, Surface
from geoh5py.workspace import Workspace
import plotly.graph_objects as go
import plotly.express as px
from ipywidgets import (
    Dropdown,
    Checkbox,
    SelectMultiple,
    FloatSlider,
    IntSlider,
    FloatText,
    VBox,
    HBox,
    ToggleButton,
    Text,
    interactive_output,
    Label,
    Layout,
)
from geoapps.base import BaseApplication
from geoapps.utils import (
    filter_xy,
    rotate_xy,
    format_labels,
    find_value,
    symlog,
    inv_symlog,
)
from geoapps.selection import ObjectDataSelection


def normalize(values):
    ind = ~np.isnan(values)
    values[ind] = np.abs(values[ind])
    values[ind] /= values[ind].max()
    values[ind == False] = 0
    return values


def format_axis(channel, axis, log, threshold):
    label = channel

    if log:
        axis = symlog(axis, threshold)

    values = axis[~np.isnan(axis)]
    ticks = np.linspace(values.min(), values.max(), 5)

    if log:
        label = f"Log({channel})"
        ticklabels = inv_symlog(ticks, threshold)
    else:
        ticklabels = ticks

    return axis, label, ticks, ticklabels.tolist()


class ScatterPlots(BaseApplication):
    """
    Application for 2D and 3D crossplots of data using symlog scaling
    """

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.selection = ObjectDataSelection(
            select_multiple=True, workspace=self.workspace
        )

        self._data = self.selection.data
        self._objects = self.selection.objects

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
        self.refresh = True

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

        self.widget = VBox(
            [
                self.project_panel,
                VBox(
                    [
                        HBox([self.selection.objects, self.selection.data]),
                        self.axes_options,
                    ]
                ),
                self.crossplot,
                self.crossplot_fig,
            ]
        )

        self.__populate__(**kwargs)

        def update_channels(_):
            self.update_channels()

        self.data.observe(update_channels, names="value")

        def update_data_list(_):
            self.update_data_list()

        self.objects.observe(update_data_list)

    @property
    def objects(self):
        """
        :obj:`ipywidgets.Dropdown`
        """
        return self._objects

    @property
    def data(self):
        """
        :obj:`ipywidgets.Dropdown`
        """
        return self._data

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
        obj, _ = self.selection.get_selected_entities()

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
        self.refresh = False
        channel = getattr(self, "_" + name).value
        self.get_channel(channel)

        if channel in self.data_channels.keys():

            values = self.data_channels[channel]
            values = values[~np.isnan(values)]

            cmin = getattr(self, "_" + name + "_min")
            cmin.value = f"{np.min(values):.2e}"
            cmax = getattr(self, "_" + name + "_max")
            cmax.value = f"{np.max(values):.2e}"
        self.refresh = True

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

        if not refresh_trigger or not self.refresh:
            return None

        if self.get_channel(size) is not None and size_active:
            vals = self.get_channel(size).copy()
            inbound = (vals > size_min) * (vals < size_max)
            vals[~inbound] = np.nan
            size = normalize(vals)

            if size_log:
                size = symlog(size, size_thresh)
            size *= size_markers
        else:
            size = None

        if self.get_channel(color) is not None and color_active:
            vals = self.get_channel(color).copy()
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

    def update_channels(self):
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
                self.set_channel_bounds(name)
        self.refresh_trigger.value = True

    def update_data_list(self):
        self.data_channels = {}


class PlotSelection2D(ObjectDataSelection):
    """
    Application for selecting data in 2D plan map view
    """

    defaults = {
        "h5file": "../../assets/FlinFlon.geoh5",
        "objects": "Gravity_Magnetics_drape60m",
        "data": "Airborne_TMI",
    }

    def __init__(self, **kwargs):

        self.collections = []

        self._azimuth = FloatSlider(
            min=-90,
            max=90,
            value=0,
            steps=5,
            description="Azimuth",
            continuous_update=False,
        )
        self._center_x = FloatSlider(
            min=-100, max=100, steps=10, description="Easting", continuous_update=False,
        )
        self._center_y = FloatSlider(
            min=-100,
            max=100,
            steps=10,
            description="Northing",
            continuous_update=False,
            orientation="vertical",
        )
        self._contours = widgets.Text(
            value="", description="Contours", disabled=False, continuous_update=False,
        )
        self._data_count = Label("Data Count: 0", tooltip="Keep <1500 for speed")
        self._resolution = FloatText(description="Grid Resolution (m)",)
        self._width = FloatSlider(
            min=0,
            max=100,
            steps=10,
            value=1000,
            description="Width",
            continuous_update=False,
        )
        self._height = FloatSlider(
            min=0,
            max=100,
            steps=10,
            value=1000,
            description="Height",
            continuous_update=False,
            orientation="vertical",
        )
        self._zoom_extent = ToggleButton(
            value=True,
            description="Zoom on selection",
            tooltip="Keep plot extent on selection",
            icon="check",
        )

        def set_bounding_box(_):
            self.set_bounding_box()

        self.highlight_selection = None

        def plot_selection(
            data_name,
            resolution,
            center_x,
            center_y,
            width,
            height,
            azimuth,
            zoom_extent,
            contours,
            refresh,
        ):

            self.plot_selection(
                data_name,
                resolution,
                center_x,
                center_y,
                width,
                height,
                azimuth,
                zoom_extent,
                contours,
                refresh,
            )

        super().__init__(**self.apply_defaults(**kwargs))

        self.window_plot = widgets.interactive_output(
            plot_selection,
            {
                "data_name": self.data,
                "resolution": self.resolution,
                "center_x": self.center_x,
                "center_y": self.center_y,
                "width": self.width,
                "height": self.height,
                "azimuth": self.azimuth,
                "zoom_extent": self.zoom_extent,
                "contours": self.contours,
                "refresh": self.refresh,
            },
        )

        self.plot_widget = VBox(
            [
                VBox([self.resolution, self.data_count,]),
                HBox(
                    [
                        self.center_y,
                        self.height,
                        VBox(
                            [
                                self.width,
                                self.center_x,
                                self.window_plot,
                                self.azimuth,
                                self.zoom_extent,
                            ]
                        ),
                    ],
                    layout=Layout(align_items="center"),
                ),
            ]
        )
        self._widget = VBox([self.widget, self.plot_widget])
        self.figure = None
        self.axis = None
        self.indices = None

        self.objects.observe(set_bounding_box, names="value")
        self.set_bounding_box()

    @property
    def azimuth(self):
        """
        :obj:`ipywidgets.FloatSlider`: Rotation angle of the selection box.
        """
        return self._azimuth

    @property
    def center_x(self):
        """
        :obj:`ipywidgets.FloatSlider`: Easting position of the selection box.
        """
        return self._center_x

    @property
    def center_y(self):
        """
        :obj:`ipywidgets.FloatSlider`: Northing position of the selection box.
        """
        return self._center_y

    @property
    def contours(self):
        """
        :obj:`ipywidgets.widgets.Text` String defining sets of contours.
        Contours can be defined over an interval `50:200:10` and/or at a fix value `215`.
        Any combination of the above can be used:
        50:200:10, 215 => Contours between values 50 and 200 every 10, with a contour at 215.
        """
        return self._contours

    @property
    def data_count(self):
        """
        :obj:`ipywidgets.Label`: Data counter included in the selection box.
        """
        return self._data_count

    @property
    def height(self):
        """
        :obj:`ipywidgets.FloatSlider`: Height (m) of the selection box
        """
        return self._height

    @property
    def resolution(self):
        """
        :obj:`ipywidgets.FloatText`: Minimum data separation (m)
        """
        return self._resolution

    @property
    def widget(self):
        """
        :obj:`ipywidgets.VBox`: Application layout
        """
        return self._widget

    @property
    def width(self):
        """
        :obj:`ipywidgets.FloatSlider`: Width (m) of the selection box
        """
        return self._width

    @property
    def zoom_extent(self):
        """
        :obj:`ipywidgets.ToggleButton`: Set plotting limits to the selection box
        """
        return self._zoom_extent

    def plot_selection(
        self,
        data_name,
        resolution,
        center_x,
        center_y,
        width,
        height,
        azimuth,
        zoom_extent,
        contours,
        refresh,
    ):
        if not refresh:
            return

        # Parse the contours string
        if contours != "":
            vals = re.split(",", contours)
            cntrs = []
            for val in vals:
                if ":" in val:
                    param = np.asarray(re.split(":", val), dtype="int")
                    if len(param) == 2:
                        cntrs += [np.arange(param[0], param[1])]
                    else:
                        cntrs += [np.arange(param[0], param[1], param[2])]
                else:
                    cntrs += [np.float(val)]
            contours = np.unique(np.sort(np.hstack(cntrs)))
        else:
            contours = None

        entity, _ = self.get_selected_entities()
        data_obj = None
        if entity.get_data(self.data.value):
            data_obj = entity.get_data(self.data.value)[0]

        if isinstance(entity, (Grid2D, Surface, Points, Curve)):

            self.figure = plt.figure(figsize=(10, 10))
            self.axis = plt.subplot()
            corners = np.r_[
                np.c_[-1.0, -1.0],
                np.c_[-1.0, 1.0],
                np.c_[1.0, 1.0],
                np.c_[1.0, -1.0],
                np.c_[-1.0, -1.0],
            ]
            corners[:, 0] *= width / 2
            corners[:, 1] *= height / 2
            corners = rotate_xy(corners, [0, 0], -azimuth)
            self.axis.plot(corners[:, 0] + center_x, corners[:, 1] + center_y, "k")
            self.axis, _, ind_filter, _, contour_set = plot_plan_data_selection(
                entity,
                data_obj,
                **{
                    "axis": self.axis,
                    "resolution": resolution,
                    "window": {
                        "center": [center_x, center_y],
                        "size": [width, height],
                        "azimuth": azimuth,
                    },
                    "zoom_extent": zoom_extent,
                    "resize": True,
                    "contours": contours,
                    "highlight_selection": self.highlight_selection,
                    "collections": self.collections,
                },
            )

            self.indices = ind_filter
            self.contours.contour_set = contour_set
            self.data_count.value = f"Data Count: {ind_filter.sum()}"

    def set_bounding_box(self):
        # Fetch vertices in the project
        lim_x = [1e8, -1e8]
        lim_y = [1e8, -1e8]

        obj, _ = self.get_selected_entities()
        if isinstance(obj, Grid2D):
            lim_x[0], lim_x[1] = obj.centroids[:, 0].min(), obj.centroids[:, 0].max()
            lim_y[0], lim_y[1] = obj.centroids[:, 1].min(), obj.centroids[:, 1].max()
        elif isinstance(obj, (Points, Curve, Surface)):
            lim_x[0], lim_x[1] = obj.vertices[:, 0].min(), obj.vertices[:, 0].max()
            lim_y[0], lim_y[1] = obj.vertices[:, 1].min(), obj.vertices[:, 1].max()
        else:
            return

        self.refresh.value = False
        self.center_x.min = -np.inf
        self.center_x.max = lim_x[1]
        self.center_x.value = np.mean(lim_x)
        self.center_x.min = lim_x[0]

        self.center_y.min = -np.inf
        self.center_y.max = lim_y[1]
        self.center_y.value = np.mean(lim_y)
        self.center_y.min = lim_y[0]

        self.width.max = lim_x[1] - lim_x[0]
        self.width.value = self.width.max / 2.0
        self.width.min = 0

        self.height.max = lim_y[1] - lim_y[0]
        self.height.min = 0
        self.height.value = self.height.max / 2.0
        self.refresh.value = True


def plot_plan_data_selection(entity, data, **kwargs):
    """
    Plot data values in 2D with contours

    :param entity: `geoh5py.objects`
        Input object with either `vertices` or `centroids` property.
    :param data: `geoh5py.data`
        Input data with `values` property.

    :return ax:
    :return out:
    :return indices:
    :return line_selection:
    :return contour_set:
    """
    indices = None
    line_selection = None
    contour_set = None
    values = None
    axis = None
    out = None

    if isinstance(entity, (Grid2D, Points, Curve, Surface)):
        if "axis" not in kwargs.keys():
            plt.figure(figsize=(8, 8))
            axis = plt.subplot()
        else:
            axis = kwargs["axis"]
    else:
        return axis, out, indices, line_selection, contour_set

    for collection in axis.collections:
        collection.remove()

    locations = entity.vertices
    if "resolution" not in kwargs.keys():
        resolution = 0
    else:
        resolution = kwargs["resolution"]

    if "indices" in kwargs.keys():
        indices = kwargs["indices"]
        if isinstance(indices, np.ndarray) and np.all(indices == False):
            indices = None

    if isinstance(getattr(data, "values", None), np.ndarray):
        if not isinstance(data.values[0], str):
            values = data.values.copy()
            values[(values > 1e-18) * (values < 2e-18)] = np.nan
            values[values == -99999] = np.nan

    color_norm = None
    if "color_norm" in kwargs.keys():
        color_norm = kwargs["color_norm"]

    window = None
    if "window" in kwargs.keys():
        window = kwargs["window"]

    if data is not None and data.entity_type.color_map is not None:
        new_cmap = data.entity_type.color_map.values
        map_vals = new_cmap["Value"].copy()
        cmap = colors.ListedColormap(
            np.c_[
                new_cmap["Red"] / 255, new_cmap["Green"] / 255, new_cmap["Blue"] / 255,
            ]
        )
        color_norm = colors.BoundaryNorm(map_vals, cmap.N)
    else:
        cmap = "Spectral_r"

    if isinstance(entity, Grid2D):
        x = entity.centroids[:, 0].reshape(entity.shape, order="F")
        y = entity.centroids[:, 1].reshape(entity.shape, order="F")
        indices = filter_xy(x, y, resolution, window=window)

        ind_x, ind_y = (
            np.any(indices, axis=1),
            np.any(indices, axis=0),
        )

        X = x[ind_x, :][:, ind_y]
        Y = y[ind_x, :][:, ind_y]

        if values is not None:
            values = values.reshape(entity.shape, order="F")
            values[indices == False] = np.nan
            values = values[ind_x, :][:, ind_y]

        if np.any(values):
            out = axis.pcolormesh(
                X, Y, values, cmap=cmap, norm=color_norm, shading="auto"
            )

        if (
            "contours" in kwargs.keys()
            and kwargs["contours"] is not None
            and np.any(values)
        ):
            contour_set = axis.contour(
                X, Y, values, levels=kwargs["contours"], colors="k", linewidths=1.0
            )

    else:
        x, y = entity.vertices[:, 0], entity.vertices[:, 1]
        if indices is None:
            indices = filter_xy(x, y, resolution, window=window,)
        X, Y = x[indices], y[indices]
        if values is not None:
            values = values[indices]

        if "marker_size" not in kwargs.keys():
            marker_size = 5
        else:
            marker_size = kwargs["marker_size"]

        out = axis.scatter(X, Y, marker_size, values, cmap=cmap, norm=color_norm)

        if (
            "contours" in kwargs.keys()
            and kwargs["contours"] is not None
            and np.any(values)
        ):
            ind = ~np.isnan(values)
            contour_set = axis.tricontour(
                X[ind],
                Y[ind],
                values[ind],
                levels=kwargs["contours"],
                colors="k",
                linewidths=1.0,
            )

    if "collections" in kwargs.keys():
        for collection in kwargs["collections"]:
            axis.add_collection(copy(collection))

    if "zoom_extent" in kwargs.keys() and kwargs["zoom_extent"] and np.any(values):
        ind = ~np.isnan(values.ravel())
        x = X.ravel()[ind]
        y = Y.ravel()[ind]
        if ind.sum() > 0:
            format_labels(x, y, axis)
            axis.set_xlim([x.min(), x.max()])
            axis.set_ylim([y.min(), y.max()])
    elif np.any(x) and np.any(y):
        format_labels(x, y, axis)
        axis.set_xlim([x.min(), x.max()])
        axis.set_ylim([y.min(), y.max()])

    if (
        "colorbar" in kwargs.keys()
        and values[~np.isnan(values)].min() != values[~np.isnan(values)].max()
    ):
        plt.colorbar(out, ax=axis)

    line_selection = np.zeros_like(indices, dtype=bool)
    if "highlight_selection" in kwargs.keys() and isinstance(
        kwargs["highlight_selection"], dict
    ):
        for key, values in kwargs["highlight_selection"].items():

            if not np.any(entity.get_data(key)):
                continue

            for line in values:
                ind = np.where(entity.get_data(key)[0].values == line)[0]
                x, y, values = (
                    locations[ind, 0],
                    locations[ind, 1],
                    entity.get_data(key)[0].values[ind],
                )
                ind_line = filter_xy(x, y, resolution, window=window)
                axis.scatter(x[ind_line], y[ind_line], marker_size * 2, "k", marker="+")
                line_selection[ind[ind_line]] = True

    return axis, out, indices, line_selection, contour_set


def plot_profile_data_selection(
    entity,
    field_list,
    uncertainties=None,
    selection={},
    resolution=None,
    plot_legend=False,
    ax=None,
    color=[0, 0, 0],
):

    locations = entity.vertices

    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = plt.subplot()

    pos = ax.get_position()
    xx, yy = [], []
    threshold = 1e-14
    for key, values in selection.items():

        for line in values:

            if entity.get_data(key):
                ind = np.where(entity.get_data(key)[0].values == line)[0]
            else:
                continue
            if len(ind) == 0:
                continue

            if resolution is not None:
                dwn_ind = filter_xy(locations[ind, 0], locations[ind, 1], resolution,)

                ind = ind[dwn_ind]

            xyLocs = locations[ind, :]

            if np.std(xyLocs[:, 0]) > np.std(xyLocs[:, 1]):
                dist = xyLocs[:, 0].copy()
            else:
                dist = xyLocs[:, 1].copy()

            dist -= dist.min()
            order = np.argsort(dist)
            legend = []

            c_increment = [(1 - c) / (len(field_list) + 1) for c in color]

            for ii, field in enumerate(field_list):
                if (
                    entity.get_data(field)
                    and entity.get_data(field)[0].values is not None
                ):
                    values = entity.get_data(field)[0].values[ind][order]

                    xx.append(dist[order][~np.isnan(values)])
                    yy.append(values[~np.isnan(values)])

                    if uncertainties is not None:
                        ax.errorbar(
                            xx[-1],
                            yy[-1],
                            yerr=uncertainties[ii][0] * np.abs(yy[-1])
                            + uncertainties[ii][1],
                            color=[c + ii * i for c, i in zip(color, c_increment)],
                        )
                    else:
                        ax.plot(
                            xx[-1],
                            yy[-1],
                            color=[c + ii * i for c, i in zip(color, c_increment)],
                        )
                    legend.append(field)

                    threshold = np.max([threshold, np.percentile(np.abs(yy[-1]), 2)])

            if plot_legend:
                ax.legend(legend, loc=3, bbox_to_anchor=(0, -0.25), ncol=3)

            if xx and yy:
                format_labels(
                    np.hstack(xx),
                    np.hstack(yy),
                    ax,
                    labels=["Distance (m)", "Fields"],
                    aspect="auto",
                )

    return ax, threshold


def plot_em_data_widget(h5file):
    workspace = Workspace(h5file)

    curves = [
        entity.parent.name + "." + entity.name
        for entity in workspace.all_objects()
        if isinstance(entity, Curve)
    ]
    names = [name for name in sorted(curves)]

    def get_parental_child(parental_name):

        parent, child = parental_name.split(".")

        parent_entity = workspace.get_entity(parent)[0]

        children = [entity for entity in parent_entity.children if entity.name == child]
        return children

    def plot_profiles(entity_name, groups, line_field, lines, scale, threshold):

        fig = plt.figure(figsize=(12, 12))
        entity = get_parental_child(entity_name)[0]

        ax = plt.subplot()
        colors = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]

        for group, color in zip(groups, colors):

            prop_group = entity.get_property_group(group)

            if prop_group is not None:
                fields = [
                    entity.workspace.get_entity(uid)[0].name
                    for uid in prop_group.properties
                ]

                ax, _ = plot_profile_data_selection(
                    prop_group.parent,
                    fields,
                    selection={line_field: lines},
                    ax=ax,
                    color=color,
                )

        ax.grid(True)

        plt.yscale(scale, linthreshy=10.0 ** threshold)

    def updateList(_):
        entity = get_parental_child(objects.value)[0]
        data_list = entity.get_data_list()
        obj = get_parental_child(objects.value)[0]

        options = [pg.name for pg in obj.property_groups]
        options = [option for option in sorted(options)]
        groups.options = options
        groups.value = [groups.options[0]]
        line_field.options = data_list
        line_field.value = find_value(data_list, ["line"])

        if line_field.value is None:
            line_ids = []
            value = []
        else:
            line_ids = np.unique(entity.get_data(line_field.value)[0].values)
            value = [line_ids[0]]

        lines.options = line_ids
        lines.value = value

    objects = Dropdown(options=names, value=names[0], description="Object:",)

    obj = get_parental_child(objects.value)[0]

    order = np.sort(obj.vertices[:, 0])

    entity = get_parental_child(objects.value)[0]

    data_list = entity.get_data_list()
    line_field = Dropdown(
        options=data_list,
        value=find_value(data_list, ["line"]),
        description="Lines field",
    )

    options = [pg.name for pg in obj.property_groups]
    options = [option for option in sorted(options)]
    groups = SelectMultiple(options=options, value=[options[0]], description="Data: ",)

    if line_field.value is None:
        line_list = []
        value = []
    else:

        line_list = np.unique(entity.get_data(line_field.value)[0].values)
        value = [line_list[0]]

    lines = SelectMultiple(options=line_list, value=value, description="Data: ")

    objects.observe(updateList, names="value")

    scale = Dropdown(
        options=["linear", "symlog"], value="symlog", description="Scaling",
    )

    threshold = FloatSlider(
        min=-16,
        max=-1,
        value=-12,
        steps=0.5,
        description="Log-linear threshold",
        continuous_update=False,
    )

    apps = VBox([objects, line_field, lines, groups, scale, threshold])
    layout = HBox(
        [
            apps,
            interactive_output(
                plot_profiles,
                {
                    "entity_name": objects,
                    "groups": groups,
                    "line_field": line_field,
                    "lines": lines,
                    "scale": scale,
                    "threshold": threshold,
                },
            ),
        ]
    )
    return layout
