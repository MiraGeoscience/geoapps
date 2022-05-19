#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import os
import uuid

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
    Widget,
)

from geoh5py.shared import Entity
from geoh5py.ui_json import InputFile
from geoh5py.objects import ObjectBase

from geoapps.base.selection import ObjectDataSelection
from geoapps.scatter_plot.constants import app_initializer
from geoapps.scatter_plot.driver import ScatterPlotDriver
from geoapps.scatter_plot.params import ScatterPlotParams

from geoapps.utils.utils import random_sampling, sorted_children_dict, find_value


class ScatterPlots(ObjectDataSelection):
    """
    Application for 2D and 3D crossplots of data using symlog scaling
    """

    _param_class = ScatterPlotParams
    _select_multiple = True
    _add_groups = False
    _downsampling = None
    _color = None
    _x = None
    _y = None
    _z = None
    _size = None

    def __init__(self, ui_json=None, **kwargs):
        app_initializer.update(kwargs)
        if ui_json is not None and os.path.exists(ui_json):
            self.params = self._param_class(InputFile(ui_json))
        else:
            self.params = self._param_class(**app_initializer)

        self.defaults = {}
        for key, value in self.params.to_dict().items():
            if isinstance(value, Entity):
                self.defaults[key] = value.uid
            else:
                self.defaults[key] = value

        self.custom_colormap = []
        self._indices = None



        def channel_bounds_setter(caller):
            self.set_channel_bounds(caller["owner"].name)

        self.x.observe(channel_bounds_setter, names="value")
        self.x.name = "x"
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
        self.y.observe(channel_bounds_setter, names="value")
        self.y.name = "y"
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
        self.z.observe(channel_bounds_setter, names="value")
        self.z.name = "z"
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
        self.color.observe(channel_bounds_setter, names="value")
        self.color.name = "color"
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
        self.size.observe(channel_bounds_setter, names="value")
        self.size.name = "size"
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
        self._refresh = ToggleButton(description="Refresh Plot", value=False)

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
        self.data_values = {}
        self.objects.observe(self.update_objects, names="value")
        self.objects.observe(self.update_choices, names="value")
        self.downsampling.observe(self.update_downsampling, names="value")
        self.figure = go.FigureWidget()

        self.x.observe(self.plot_selection, names="value")
        self.x_log.observe(self.plot_selection, names="value")
        self.x_active.observe(self.plot_selection, names="value")
        self.x_thresh.observe(self.plot_selection, names="value")
        self.x_min.observe(self.plot_selection, names="value")
        self.x_max.observe(self.plot_selection, names="value")
        self.y.observe(self.plot_selection, names="value")
        self.y_log.observe(self.plot_selection, names="value")
        self.y_active.observe(self.plot_selection, names="value")
        self.y_thresh.observe(self.plot_selection, names="value")
        self.y_min.observe(self.plot_selection, names="value")
        self.y_max.observe(self.plot_selection, names="value")
        self.z.observe(self.plot_selection, names="value")
        self.z_log.observe(self.plot_selection, names="value")
        self.z_active.observe(self.plot_selection, names="value")
        self.z_thresh.observe(self.plot_selection, names="value")
        self.z_min.observe(self.plot_selection, names="value")
        self.z_max.observe(self.plot_selection, names="value")
        self.color.observe(self.plot_selection, names="value")
        self.color_log.observe(self.plot_selection, names="value")
        self.color_active.observe(self.plot_selection, names="value")
        self.color_thresh.observe(self.plot_selection, names="value")
        self.color_min.observe(self.plot_selection, names="value")
        self.color_max.observe(self.plot_selection, names="value")
        self.color_maps.observe(self.plot_selection, names="value")
        self.size.observe(self.plot_selection, names="value")
        self.size_log.observe(self.plot_selection, names="value")
        self.size_active.observe(self.plot_selection, names="value")
        self.size_thresh.observe(self.plot_selection, names="value")
        self.size_min.observe(self.plot_selection, names="value")
        self.size_max.observe(self.plot_selection, names="value")
        self.size_markers.observe(self.plot_selection, names="value")
        self.refresh.observe(self.plot_selection, names="value")

        self.trigger.on_click(self.trigger_click)
        self.trigger.description = "Save HTML"

        super().__init__(**self.defaults)

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
        Widget for the selection of color values
        """
        if getattr(self, "_color", None) is None:
            self._color = Dropdown(description="Data:")

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
    def downsampling(self) -> IntSlider:
        """
        Widget controlling the size of the population.
        """
        if getattr(self, "_downsampling", None) is None:
            self._downsampling = IntSlider(
                description="Population Downsampling:",
                min=1,
                style={"description_width": "initial"},
                continuous_update=False,
            )
        return self._downsampling

    @property
    def main(self):
        if self._main is None:
            self._main = VBox(
                [
                    self.project_panel,
                    self.objects,
                    VBox([Label("Downsampling"), self.downsampling]),
                    self.axes_options,
                    self.refresh,
                    self.figure,
                    self.trigger,
                ]
            )

        return self._main

    @property
    def refresh(self):
        """"""
        return self._refresh

    @property
    def size(self):
        """
        Widget for the selection of size scaling values
        """
        if getattr(self, "_size", None) is None:
            self._size = Dropdown(description="Data:")
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
        Widget for the selection of x-axis values
        """
        if getattr(self, "_x", None) is None:
            self._x = Dropdown(description="Data:")
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
        Widget for the selection of y-axis values
        """
        if getattr(self, "_y", None) is None:
            self._y = Dropdown(description="Data:")
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
        Widget for the selection of z-axis values
        """
        if getattr(self, "_z", None) is None:
            self._z = Dropdown(description="Data:")
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

    def plot_selection(self, _):

        new_params_dict = {}
        for key, value in self.params.to_dict().items():
            param = getattr(self, key, None)
            if param is None:
                pass
            elif getattr(param, "value", None) is None:
                new_params_dict[key] = param
            else:
                new_params_dict[key] = param.value

        ifile = InputFile(
            ui_json=self.params.input_file.ui_json,
            validation_options={"disabled": True},
        )

        new_params = ScatterPlotParams(input_file=ifile, **new_params_dict)
        new_params.write_input_file()

        driver = ScatterPlotDriver(new_params)
        self.figure = driver.run()

    def update_axes(self, refresh_plot=True):
        for name in [
            "x",
            "y",
            "z",
            "color",
            "size",
        ]:
            self.refresh.value = False
            widget = getattr(self, "_" + name)
            widget.options = list(self.data_channels.keys())
            val = widget.value

            '''
            if val in list(dict(widget.options).values()):
            #if val in list(dict(widget.options)):
                widget.value = val
            else:
                widget.value = None
            '''
        if refresh_plot:
            self.refresh.value = True

    def update_choices(self, _):
        self.refresh.value = False

        obj, _ = self.get_selected_entities()
        channel_list = ObjectBase.get_data_list(obj)

        if "Visual Parameters" in channel_list:
            channel_list.remove("Visual Parameters")

        channel_list.extend(["X", "Y", "Z"])

        self.data_values = []
        for channel in channel_list:
            self.get_channel(channel)
            self.data_values.append(ObjectBase.get_data(obj, channel_list[0]))

        #channel_list.insert(0, "None")
        #self.data_values.insert(0, None)

        self.update_axes(refresh_plot=False)

        if self.downsampling.value != self.n_values:
            self.update_downsampling(None, refresh_plot=False)

        self.refresh.value = True

    def update_downsampling(self, _, refresh_plot=True):
        '''
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
        self.refresh.value = refresh_plot
        '''

    def update_objects(self, _):
        self.data_channels = {}
        self.refresh.value = False
        self.figure.data = []
        self.x_active.value = False
        self.y_active.value = False
        self.z_active.value = False
        self.color_active.value = False
        self.size_active.value = False
        if self.n_values is not None:
            self.downsampling.max = self.n_values
            self.downsampling.value = np.min([5000, self.n_values])
        self._indices = None
        self.update_downsampling(None, refresh_plot=False)
        self.refresh.value = True

    def trigger_click(self, _):
        self.figure.write_html(
            os.path.join(
                os.path.abspath(os.path.dirname(self.h5file)), "Crossplot.html"
            )
        )
