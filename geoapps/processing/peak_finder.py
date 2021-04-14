#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import re

import dask
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dask.distributed import Client, get_client
from geoh5py.data import ReferencedData
from geoh5py.objects import Curve, Points, Surface
from geoh5py.workspace import Workspace
from ipywidgets import (
    Box,
    Button,
    Checkbox,
    ColorPicker,
    Dropdown,
    FloatLogSlider,
    FloatSlider,
    FloatText,
    HBox,
    IntSlider,
    Label,
    Layout,
    SelectMultiple,
    ToggleButton,
    ToggleButtons,
    VBox,
    interactive_output,
)
from scipy.spatial import cKDTree

from geoapps.selection import LineOptions, ObjectDataSelection
from geoapps.utils.utils import (
    LineDataDerivatives,
    colors,
    find_value,
    geophysical_systems,
    hex_to_rgb,
    rotate_azimuth_dip,
    running_mean,
)


class PeakFinder(ObjectDataSelection):
    """
    Application for the picking of targets along Time-domain EM profiles
    """

    defaults = {
        "object_types": Curve,
        "h5file": "../../assets/FlinFlon.geoh5",
        "add_groups": True,
        "objects": "Data_TEM_pseudo3D",
        "data": ["Observed"],
        "model": {"objects": "Inversion_VTEM_Model", "data": "Iteration_7_model"},
        "lines": {"objects": "Data_TEM_pseudo3D", "data": "Line", "lines": 6073400.0},
        "boreholes": {"objects": "geochem", "data": "Al2O3"},
        "doi": {"data": "Z"},
        "doi_percent": 60,
        "doi_revert": True,
        "center": 4050,
        "width": 1000,
        "smoothing": 6,
        "tem_checkbox": True,
        "markers": True,
        "slice_width": 25,
        "x_label": "Distance",
        "ga_group_name": "PeakFinder",
    }

    def __init__(self, **kwargs):

        kwargs = self.apply_defaults(**kwargs)

        try:
            self.client = get_client()
        except ValueError:
            self.client = Client()
        self.decay_figure = None
        self.all_anomalies = []
        self.borehole_trace = None
        self.data_channels = {}
        self.data_channel_options = {}
        self.em_system_specs = geophysical_systems.parameters()
        self.marker = {"left": "<", "right": ">"}
        self.pause_plot_refresh = False
        self.surface_model = None
        self._survey = None
        self._time_groups = None
        self.objects.observe(self.objects_change, names="value")
        self.model_panel = VBox([self.show_model])
        self.show_model.observe(self.show_model_trigger, names="value")
        self.doi_panel = VBox([self.show_doi])
        self.show_doi.observe(self.show_doi_trigger, names="value")
        self.borehole_panel = VBox([self.show_borehole])
        self.show_borehole.observe(self.show_borehole_trigger, names="value")
        self.system.observe(self.system_observer, names="value")
        self.system_panel_option.observe(self.system_panel_trigger)
        self.system_panel = VBox([self.system_panel_option])
        self.groups_setter.observe(self.groups_trigger)
        self.groups_widget = VBox([self.groups_setter])
        self.groups_panel = VBox([self.group_list, self.channels, self.group_color])
        self.decay_panel = VBox([self.show_decay])
        self.previous_line = self.lines.lines.value
        self.objects.description = "Survey"
        self.boreholes.objects.description = "Points"
        self.boreholes.data.description = "Values"
        self.model_selection.objects.description = "Surface:"
        self.model_selection.data.description = "Model"
        self.doi_selection.data.description = "DOI Layer"
        self.scale_panel = VBox([self.scale_button, self.scale_value])
        self.scale_button.observe(self.scale_update)
        self.channel_selection.observe(self.channel_panel_update, names="value")
        self.channels.observe(self.edit_group, names="value")
        self.group_list.observe(self.highlight_selection, names="value")
        self.highlight_selection(None)
        self.run_all.on_click(self.run_all_click)
        self.flip_sign.observe(self.set_data, names="value")
        plotting = interactive_output(
            self.plot_data_selection,
            {
                "ind": self.lines.lines,
                "smoothing": self.smoothing,
                "max_migration": self.max_migration,
                "min_channels": self.min_channels,
                "min_amplitude": self.min_amplitude,
                "min_value": self.min_value,
                "min_width": self.min_width,
                "residual": self.residual,
                "markers": self.markers,
                "scale": self.scale_button,
                "scale_value": self.scale_value,
                "center": self.center,
                "width": self.width,
                "groups": self.group_list,
                "plot_trigger": self.plot_trigger,
                "x_label": self.x_label,
            },
        )
        self.decay = interactive_output(
            self.plot_decay_curve,
            {
                "ind": self.lines.lines,
                "smoothing": self.smoothing,
                "residual": self.residual,
                "center": self.center,
                "groups": self.group_list,
                "plot_trigger": self.plot_trigger,
            },
        )
        self.show_decay.observe(self.show_decay_trigger, names="value")
        self.tem_box = HBox(
            [
                self.tem_checkbox,
                self.system_panel,
                self.groups_widget,
                self.decay_panel,
            ]
        )
        self.tem_checkbox.observe(self.objects_change, names="value")
        self.model_section = interactive_output(
            self.plot_model_selection,
            {
                "ind": self.lines.lines,
                "center": self.center,
                "width": self.width,
                "model": self.model_selection.data,
                "smoothing": self.smoothing,
                "slice_width": self.slice_width,
                "x_label": self.x_label,
                "colormap": self.color_maps,
                "log": self.model_log,
                "min": self.model_min,
                "max": self.model_max,
                "reverse": self.reverse_cmap,
                "opacity": self.opacity,
                "doi_show": self.show_doi,
                "doi": self.doi_selection.data,
                "doi_percent": self.doi_percent,
                "doi_revert": self.doi_revert,
                "borehole_show": self.show_borehole,
                "borehole_object": self.boreholes.objects,
                "borehole_data": self.boreholes.data,
                "boreholes_size": self.boreholes_size,
                "max_migration": self.max_migration,
                "min_channels": self.min_channels,
                "min_amplitude": self.min_amplitude,
                "min_value": self.min_value,
                "min_width": self.min_width,
                "plot_trigger": self.plot_trigger,
            },
        )

        super().__init__(**kwargs)

        if "lines" in kwargs.keys():
            self.lines.__populate__(**kwargs["lines"])

        if "boreholes" in kwargs.keys():
            self.boreholes.__populate__(**kwargs["boreholes"])

        if "model" in kwargs.keys():
            self.model_selection.__populate__(**kwargs["model"])

        if "doi" in kwargs.keys():
            self.doi_selection.__populate__(**kwargs["doi"])

        self.channel_panel = VBox(
            [
                self.channel_selection,
            ]
        )
        self.system_options = VBox(
            [
                self.system,
                self.channel_panel,
            ]
        )

        self.trigger.on_click(self.trigger_click)
        self.trigger.description = "Export Peaks"
        self.trigger_panel = VBox(
            [
                VBox([self.trigger, self.structural_markers, self.ga_group_name]),
                self.live_link_panel,
            ]
        )
        self.ga_group_name.description = "Save As"
        self.visual_parameters = VBox(
            [
                self.center,
                self.width,
                self.x_label,
                self.scale_panel,
                self.markers,
            ]
        )
        self.detection_parameters = VBox(
            [
                self.smoothing,
                self.min_amplitude,
                self.min_value,
                self.min_width,
                self.max_migration,
                self.min_channels,
                self.residual,
            ]
        )
        self.model_parameters = VBox(
            [
                self.model_selection.objects,
                self.model_selection.data,
                self.model_log,
                self.model_min,
                self.model_max,
                self.color_maps,
                self.reverse_cmap,
                self.opacity,
            ],
        )
        self.doi_parameters = VBox(
            [
                self.doi_selection.data,
                self.doi_percent,
                self.doi_revert,
            ]
        )
        self.scatter_parameters = VBox(
            [
                self.boreholes.main,
                self.slice_width,
                self.boreholes_size,
            ]
        )
        self.output_panel = VBox(
            [
                self.run_all,
                self.trigger_panel,
            ]
        )
        self._main = VBox(
            [
                self.project_panel,
                HBox(
                    [
                        VBox(
                            [self.main, self.flip_sign],
                            layout=Layout(width="50%"),
                        ),
                        Box(
                            children=[self.lines.main],
                            layout=Layout(
                                display="flex",
                                flex_flow="row",
                                align_items="stretch",
                                width="100%",
                                justify_content="flex-start",
                            ),
                        ),
                    ],
                ),
                self.tem_box,
                plotting,
                HBox(
                    [
                        VBox(
                            [Label("Visual Parameters"), self.visual_parameters],
                            layout=Layout(width="50%"),
                        ),
                        VBox(
                            [Label("Detection Parameters"), self.detection_parameters],
                            layout=Layout(width="50%"),
                        ),
                    ]
                ),
                self.model_panel,
                self.output_panel,
            ]
        )

    @property
    def plot_trigger(self):
        """
        :obj:`ipywidgets.ToggleButton`: Trigger refresh of all plots
        """
        if getattr(self, "_plot_trigger", None) is None:
            self._plot_trigger = ToggleButton(
                description="Pick nearest target", value=False
            )

        return self._plot_trigger

    @property
    def boreholes(self):
        """
        :obj:`geoapps.selection.ObjectDataSelection`: Widget for the selection of borehole data
        """
        if getattr(self, "_boreholes", None) is None:
            self._boreholes = ObjectDataSelection(object_types=Points)

        return self._boreholes

    @property
    def boreholes_size(self):
        """
        :obj:`ipywidgets.IntSlider`: Adjust the size of borehole markers
        """
        if getattr(self, "_boreholes_size", None) is None:
            self._boreholes_size = IntSlider(
                value=3,
                min=1,
                max=20,
                description="Marker Size",
                continuous_update=False,
            )

        return self._boreholes_size

    @property
    def center(self):
        """
        :obj:`ipywidgets.FloatSlider`: Adjust the data plot center position along line
        """
        if getattr(self, "_center", None) is None:
            self._center = FloatSlider(
                min=0,
                max=5000,
                step=1.0,
                description="Window Center",
                disabled=False,
                continuous_update=False,
                orientation="horizontal",
            )

        return self._center

    @property
    def channels(self):
        """
        :obj:`ipywidgets.SelectMultiple`: Selection of data channels
        """
        if getattr(self, "_channels", None) is None:
            self._channels = SelectMultiple(description="Channels")

        return self._channels

    @property
    def channel_selection(self):
        """
        :obj:`ipywidgets.Dropdown`: Selection of data channels expected from the selected em system
        """
        if getattr(self, "_channel_selection", None) is None:
            self._channel_selection = Dropdown(
                description="Time Gate",
                options=self.em_system_specs[self.system.value]["channels"].keys(),
            )

        return self._channel_selection

    @property
    def color_maps(self):
        """
        :obj:`ipywidgets.Dropdown`: Selection of colormap used by the model plot
        """
        if getattr(self, "_color_maps", None) is None:
            self._color_maps = Dropdown(
                description="Colormaps",
                options=px.colors.named_colorscales(),
                value="edge",
            )

        return self._color_maps

    @property
    def data(self):
        """
        :obj:`ipywidgets.SelectMultiple`: Data selection used by the application
        """
        if getattr(self, "_data", None) is None:
            self.data = SelectMultiple(description="Data: ")

        return self._data

    @data.setter
    def data(self, value):
        assert isinstance(
            value, (Dropdown, SelectMultiple)
        ), f"'Objects' must be of type {Dropdown} or {SelectMultiple}"
        self._data = value
        self._data.observe(self.set_data, names="value")
        self.set_data(None)

    @property
    def doi_percent(self):
        """
        :obj:`ipywidgets.FloatSlider`: Define the DOI index used to mask the model plot
        """
        if getattr(self, "_doi_percent", None) is None:
            self._doi_percent = FloatSlider(
                value=20.0,
                min=0.0,
                max=100.0,
                step=0.1,
                continuous_update=False,
                description="DOI %",
            )

        return self._doi_percent

    @property
    def doi_revert(self):
        """
        :obj:`ipywidgets.Checkbox`: Apply the inverse of the DOI index
        """
        if getattr(self, "_doi_revert", None) is None:
            self._doi_revert = Checkbox(description="Revert", value=False)

        return self._doi_revert

    @property
    def doi_selection(self):
        """
        :obj:`geoapps.selection.ObjectDataSelection`: Widget for the selection of a DOI model
        """
        if getattr(self, "_doi_selection", None) is None:
            self._doi_selection = ObjectDataSelection(
                objects=self.model_selection.objects
            )

        return self._doi_selection

    @property
    def flip_sign(self):
        """
        :obj:`ipywidgets.ToggleButton`: Apply a sign flip to the selected data
        """
        if getattr(self, "_flip_sign", None) is None:
            self._flip_sign = ToggleButton(
                description="Flip Y (-1x)", button_style="warning"
            )

        return self._flip_sign

    @property
    def group_color(self):
        """
        :obj:`ipywidgets.ColorPicker`: Assign a color to the selected group
        """
        if getattr(self, "_group_color", None) is None:
            self._group_color = ColorPicker(
                concise=False, description="Color", value="blue", disabled=False
            )

        return self._group_color

    @property
    def group_list(self):
        """
        :obj:`ipywidgets.Dropdown`: List of default time data groups
        """
        if getattr(self, "_group_list", None) is None:
            self._group_list = Dropdown(
                description="",
                options=[
                    "early",
                    "early + middle",
                    "middle",
                    "early + middle + late",
                    "middle + late",
                    "late",
                ],
            )

        return self._group_list

    @property
    def groups_setter(self):
        """
        :obj:`ipywidgets.ToggleButton`: Display the group options panel
        """
        if getattr(self, "_groups_setter", None) is None:
            self._groups_setter = ToggleButton(
                description="Select Time Groups", value=False
            )

        return self._groups_setter

    @property
    def lines(self):
        """
        :obj:`geoapps.selection.LineOptions`: Line selection widget defining the profile used for plotting.
        """
        if getattr(self, "_lines", None) is None:
            self._lines = LineOptions(multiple_lines=False, objects=self.objects)

        return self._lines

    @property
    def markers(self):
        """
        :obj:`ipywidgets.ToggleButton`: Display markers on the data plot
        """
        if getattr(self, "_markers", None) is None:
            self._markers = ToggleButton(description="Show markers")

        return self._markers

    @property
    def max_migration(self):
        """
        :obj:`ipywidgets.FloatSlider`: Filter anomalies based on maximum horizontal migration of peaks.
        """
        if getattr(self, "_max_migration", None) is None:
            self._max_migration = FloatSlider(
                value=25,
                min=1.0,
                max=1000.0,
                step=1.0,
                continuous_update=False,
                description="Maximum peak migration (m)",
                style={"description_width": "initial"},
                disabled=True,
            )

        return self._max_migration

    @property
    def min_amplitude(self):
        """
        :obj:`ipywidgets.IntSlider`: Filter small anomalies based on amplitude ratio
        between peaks and lows.
        """
        if getattr(self, "_min_amplitude", None) is None:
            self._min_amplitude = IntSlider(
                value=1,
                min=0,
                max=100,
                continuous_update=False,
                description="Minimum amplitude (%)",
                style={"description_width": "initial"},
            )

        return self._min_amplitude

    @property
    def min_channels(self):
        """
        :obj:`ipywidgets.IntSlider`: Filter peak groups based on minimum number of data channels overlap.
        """
        if getattr(self, "_min_channels", None) is None:
            self._min_channels = IntSlider(
                value=1,
                min=1,
                max=10,
                continuous_update=False,
                description="Minimum # channels",
                style={"description_width": "initial"},
                disabled=True,
            )

        return self._min_channels

    @property
    def min_value(self):
        """
        :obj:`ipywidgets.FloatText`: Filter out small data values.
        """
        if getattr(self, "_min_value", None) is None:
            self._min_value = FloatText(
                value=0,
                continuous_update=False,
                description="Minimum data value",
                style={"description_width": "initial"},
            )

        return self._min_value

    @property
    def min_width(self):
        """
        :obj:`ipywidgets.FloatSlider`: Filter small anomalies based on width
        between lows.
        """
        if getattr(self, "_min_width", None) is None:
            self._min_width = FloatSlider(
                value=100,
                min=1.0,
                max=1000.0,
                step=1.0,
                continuous_update=False,
                description="Minimum width (m)",
                style={"description_width": "initial"},
            )

        return self._min_width

    @property
    def model_log(self):
        """
        :obj:`ipywidgets.Checkbox`: Display model values in log10 scale
        """
        if getattr(self, "_model_log", None) is None:
            self._model_log = Checkbox(description="log", value=True, indent=False)

        return self._model_log

    @property
    def model_max(self):
        """
        :obj:`ipywidgets.FloatText`: Upper bound value used to plot the model
        """
        if getattr(self, "_model_max", None) is None:
            self._model_max = FloatText(
                description="max", value=1e-1, continuous_update=False
            )

        return self._model_max

    @property
    def model_min(self):
        """
        :obj:`ipywidgets.FloatText`: Lower bound value used to plot the model
        """
        if getattr(self, "_model_min", None) is None:
            self._model_min = FloatText(
                description="min", value=1e-4, continuous_update=False
            )

        return self._model_min

    @property
    def model_selection(self):
        """
        :obj:`geoapps.selection.ObjectDataSelection`: Widget for the selection of a surface model object and values
        """
        if getattr(self, "_model_selection", None) is None:
            self._model_selection = ObjectDataSelection(object_types=Surface)

        return self._model_selection

    @property
    def opacity(self):
        """
        :obj:`ipywidgets.FloatSlider`: Adjust the transparency of the model plot
        """
        if getattr(self, "_opacity", None) is None:
            self._opacity = FloatSlider(
                value=0.9,
                min=0.0,
                max=1.0,
                step=0.05,
                continuous_update=False,
                description="Opacity",
            )

        return self._opacity

    @property
    def residual(self):
        """
        :obj:`ipywidgets.Checkbox`: Use the residual between the original and smoothed data profile
        """
        if getattr(self, "_residual", None) is None:
            self._residual = Checkbox(description="Use residual", value=False)

        return self._residual

    @property
    def reverse_cmap(self):
        """
        :obj:`ipywidgets.ToggleButton`: Reverse the colormap used by the model plot
        """
        if getattr(self, "_reverse_cmap", None) is None:
            self._reverse_cmap = ToggleButton(description="Flip colormap", value=False)

        return self._reverse_cmap

    @property
    def run_all(self):
        """
        :obj:`ipywidgets.Button`: Trigger the peak finder calculation for all lines
        """
        if getattr(self, "_run_all", None) is None:
            self._run_all = Button(
                description="Process All Lines", button_style="warning"
            )

        return self._run_all

    @property
    def scale_button(self):
        """
        :obj:`ipywidgets.ToggleButtons`: Scale the vertical axis of the data plot
        """
        if getattr(self, "_scale_button", None) is None:
            self._scale_button = ToggleButtons(
                options=[
                    "linear",
                    "symlog",
                ],
                value="symlog",
                description="Y-axis scaling",
            )

        return self._scale_button

    @property
    def scale_value(self):
        """
        :obj:`ipywidgets.FloatLogSlider`: Threshold value used by th symlog scaling
        """
        if getattr(self, "_scale_value", None) is None:
            self._scale_value = FloatLogSlider(
                min=-18,
                max=10,
                step=0.1,
                base=10,
                value=1e-2,
                description="Linear threshold",
                continuous_update=False,
                style={"description_width": "initial"},
            )

        return self._scale_value

    @property
    def show_borehole(self):
        """
        :obj:`ipywidgets.ToggleButton`: Display the borehole panel
        """
        if getattr(self, "_show_borehole", None) is None:
            self._show_borehole = ToggleButton(
                description="Show Scatter",
                value=False,
            )

        return self._show_borehole

    @property
    def show_decay(self):
        """
        :obj:`ipywidgets.ToggleButton`: Display the decay curve plot
        """
        if getattr(self, "_show_decay", None) is None:
            self._show_decay = ToggleButton(description="Show decay", value=False)

        return self._show_decay

    @property
    def show_doi(self):
        """
        :obj:`ipywidgets.ToggleButton`: Display the doi options panel
        """
        if getattr(self, "_show_doi", None) is None:
            self._show_doi = ToggleButton(
                description="Show DOI",
                value=False,
            )

        return self._show_doi

    @property
    def show_model(self):
        """
        :obj:`ipywidgets.ToggleButton`: Display the model plot options panel
        """
        if getattr(self, "_show_model", None) is None:
            self._show_model = ToggleButton(
                description="Show model",
                value=False,
            )

        return self._show_model

    @property
    def slice_width(self):
        """
        :obj:`ipywidgets.FloatSlider`: Change the search radius for plotting around the model section
        """
        if getattr(self, "_slice_width", None) is None:
            self._slice_width = FloatSlider(
                value=10.0,
                min=1.0,
                max=500.0,
                step=1.0,
                description="Slice width (m)",
                disabled=False,
                continuous_update=False,
                orientation="horizontal",
            )

        return self._slice_width

    @property
    def smoothing(self):
        """
        :obj:`ipywidgets.IntSlider`: Number of neighboring data points used for the running mean smoothing
        """
        if getattr(self, "_smoothing", None) is None:
            self._smoothing = IntSlider(
                min=0,
                max=64,
                value=0,
                description="Smoothing",
                continuous_update=False,
            )

        return self._smoothing

    @property
    def structural_markers(self):
        """
        :obj:`ipywidgets.Checkbox`: Export peaks as structural markers
        """
        if getattr(self, "_structural_markers", None) is None:
            self._structural_markers = Checkbox(description="All Markers")

        return self._structural_markers

    @property
    def survey(self):
        """
        Selected curve object
        """
        return self._survey

    @property
    def system(self):
        """
        :obj:`ipywidgets.Dropdown`: Selection of a TEM system
        """
        if getattr(self, "_system", None) is None:
            self._system = Dropdown(
                options=[
                    key
                    for key, specs in self.em_system_specs.items()
                    if specs["type"] == "time"
                ],
                description="Time-Domain System:",
                style={"description_width": "initial"},
            )
        return self._system

    @property
    def system_panel_option(self):
        """
        :obj:`ipywidgets.ToggleButton`: Display system options
        """
        if getattr(self, "_system_panel_option", None) is None:
            self._system_panel_option = ToggleButton(
                description="Select TEM System", value=False
            )

        return self._system_panel_option

    @property
    def tem_checkbox(self):
        """
        :obj:`ipywidgets.Checkbox`: Enable options specific to TEM data groups
        """
        if getattr(self, "_tem_checkbox", None) is None:
            self._tem_checkbox = Checkbox(description="TEM Data", value=True)

        return self._tem_checkbox

    @property
    def time_groups(self):
        """
        Dict of time groups used to classify peaks
        """

        if getattr(self, "_time_groups", None) is None:
            self._time_groups = {
                0: {"name": "early", "label": [0], "color": "#0000FF", "channels": []},
                1: {"name": "middle", "label": [1], "color": "#FFFF00", "channels": []},
                2: {"name": "late", "label": [2], "color": "#FF0000", "channels": []},
                3: {
                    "name": "early + middle",
                    "label": [0, 1],
                    "color": "#00FFFF",
                    "channels": [],
                },
                4: {
                    "name": "early + middle + late",
                    "label": [0, 1, 2],
                    "color": "#008000",
                    "channels": [],
                },
                5: {
                    "name": "middle + late",
                    "label": [1, 2],
                    "color": "#FFA500",
                    "channels": [],
                },
            }
        return self._time_groups

    @time_groups.setter
    def time_groups(self, groups: dict):
        self._time_groups = groups

    @property
    def width(self):
        """
        :obj:`ipywidgets.FloatSlider`: Adjust the length of data displayed on the data plot
        """
        if getattr(self, "_width", None) is None:
            self._width = FloatSlider(
                min=0.0,
                max=5000.0,
                step=1.0,
                description="Window Width",
                disabled=False,
                continuous_update=False,
                orientation="horizontal",
            )

        return self._width

    @property
    def workspace(self):
        """
        Target geoh5py workspace
        """
        if (
            getattr(self, "_workspace", None) is None
            and getattr(self, "_h5file", None) is not None
        ):
            self.workspace = Workspace(self.h5file)
        return self._workspace

    @workspace.setter
    def workspace(self, workspace):
        assert isinstance(workspace, Workspace), f"Workspace must of class {Workspace}"
        self._workspace = workspace
        self._h5file = workspace.h5file
        self.update_objects_list()
        self.lines._workspace = workspace
        self.model_selection.workspace = workspace
        self.boreholes.workspace = workspace
        self.doi_selection._workspace = workspace
        self.reset_model_figure()

    @property
    def x_label(self):
        """
        :obj:`ipywidgets.ToggleButtons`: Units of distance displayed on the data plot
        """
        if getattr(self, "_x_label", None) is None:
            self._x_label = ToggleButtons(
                options=["Distance", "Easting", "Northing"],
                value="Distance",
                description="X-axis label:",
            )

        return self._x_label

    def channel_panel_update(self, _):
        """
        Observer of :obj:`geoapps.processing.PeakFinder.channel_selection`: Change data channel panel
        """
        try:
            self.channel_panel.children = [
                self.channel_selection,
                self.data_channel_options[self.channel_selection.value],
            ]
        except KeyError:
            self.channel_panel.children = [self.channel_selection]

    def set_data(self, _):
        """
        Observer of :obj:`geoapps.processing.PeakFinder.data`: Populate the list of available channels and refresh groups
        """
        if getattr(self, "survey", None) is not None:
            self.pause_plot_refresh = True
            groups = [p_g.name for p_g in self.survey.property_groups]
            channels = []

            # Add all selected data channels | groups once
            for channel in self.data.value:
                if channel in groups:
                    for prop in self.survey.get_property_group(channel).properties:
                        name = self.workspace.get_entity(prop)[0].name
                        if prop not in channels:
                            channels.append(name)
                elif channel not in channels:
                    channels.append(channel)

            self.channels.options = channels
            for channel in channels:
                if self.survey.get_data(channel):
                    self.data_channels[channel] = self.survey.get_data(channel)[0]

            # Generate default groups
            self.reset_groups()

            if self.tem_checkbox.value:
                for key, widget in self.data_channel_options.items():
                    widget.children[0].options = channels
                    widget.children[0].value = find_value(channels, [key])
            self.pause_plot_refresh = False
            self.plot_trigger.value = False
            self.plot_trigger.value = True

    def objects_change(self, _):
        """
        Observer of :obj:`geoapps.processing.PeakFinder.objects`: Reset data and auto-detect AEM system
        """
        if self.workspace.get_entity(self.objects.value):
            self._survey = self.workspace.get_entity(self.objects.value)[0]
            self.update_data_list(None)
            not_tem = True
            self.active_channels = []
            if self.tem_checkbox.value:
                not_tem = False
                for aem_system, specs in self.em_system_specs.items():
                    if any(
                        [
                            specs["flag"] in channel
                            for channel in self._survey.get_data_list()
                        ]
                    ):
                        if aem_system in self.system.options:
                            self.system.value = aem_system
                            # not_tem = False
                            break

            if not_tem:
                self.tem_box.children = [self.tem_checkbox]
                self.min_channels.disabled = True
                self.max_migration.disabled = False
            else:
                self.tem_box.children = [
                    self.tem_checkbox,
                    self.system_panel,
                    self.groups_widget,
                    self.decay_panel,
                ]
                self.min_channels.disabled = False
                self.max_migration.disabled = False

            self.set_data(None)

    def reset_model_figure(self):
        self.model_figure = go.FigureWidget()
        self.model_figure.add_trace(go.Scatter3d())
        self.model_figure.add_trace(go.Cone())
        self.model_figure.add_trace(go.Mesh3d())
        self.model_figure.add_trace(go.Scatter3d())
        self.model_figure.update_layout(
            scene={
                "xaxis_title": "Easting (m)",
                "yaxis_title": "Northing (m)",
                "zaxis_title": "Elevation (m)",
                "yaxis": {"autorange": "reversed"},
                "xaxis": {"autorange": "reversed"},
                "aspectmode": "data",
            },
            width=700,
            height=500,
            autosize=False,
            uirevision=False,
        )
        self.show_model.value = False

    def system_panel_trigger(self, _):
        """
        Observer of :obj:`geoapps.processing.PeakFinder.`:
        """
        if self.system_panel_option.value:
            self.system_panel.children = [self.system_panel_option, self.system_options]
        else:
            self.system_panel.children = [self.system_panel_option]

    def system_observer(self, _):
        """
        Observer of :obj:`geoapps.processing.PeakFinder.`:
        """

        def channel_setter(caller):
            channel = caller["owner"]
            data_widget = self.data_channel_options[channel.header]
            data_widget.children[0].value = find_value(
                data_widget.children[0].options, [channel.header]
            )

        system_specs = {}
        for key, time_gate in self.em_system_specs[self.system.value][
            "channels"
        ].items():
            system_specs[key] = f"{time_gate:.5e}"

        self.channel_selection.options = self.em_system_specs[self.system.value][
            "channels"
        ].keys()

        self.data_channel_options = {}
        for ind, (key, value) in enumerate(system_specs.items()):
            channel_selection = Dropdown(
                description="Channel",
                style={"description_width": "initial"},
                options=self.channels.options,
                value=find_value(self.channels.options, [key]),
            )
            channel_selection.header = key
            channel_selection.observe(channel_setter, names="value")

            channel_time = FloatText(description="Time (s)", value=value)

            self.data_channel_options[key] = VBox([channel_selection, channel_time])

        self.reset_groups()

    def groups_trigger(self, _):
        """
        Observer of :obj:`geoapps.processing.PeakFinder.`:
        """
        if self.groups_setter.value:
            self.groups_widget.children = [self.groups_setter, self.groups_panel]
        else:
            self.groups_widget.children = [self.groups_setter]

    def edit_group(self, _):
        """
        Observer of :obj:`geoapps.processing.PeakFinder.`: Change channels associated with groups
        """
        gates = {}
        for key, group in self.time_groups.items():
            if group["name"] == self.group_list.value and group["channels"] != list(
                self.channels.value
            ):
                gates[key] = list(self.channels.value)

        self.reset_groups(gates=gates)
        self.plot_trigger.value = False
        self.plot_trigger.value = True

    def run_all_click(self, _):
        """
        Observer of :obj:`geoapps.processing.PeakFinder.`: to process the entire Curve object for all lines
        """
        self.run_all.description = "Computing..."
        anomalies = []
        vertices = self.client.scatter(self.survey.vertices)
        channels = self.client.scatter(self.active_channels)
        time_groups = self.client.scatter(self.time_groups)
        for line_id in list(self.lines.lines.options)[1:]:
            line_indices = self.get_line_indices(line_id)

            if line_indices is None:
                continue

            anomalies += [
                self.client.compute(
                    find_anomalies(
                        vertices,
                        line_indices,
                        channels,
                        time_groups,
                        data_normalization=self.em_system_specs[self.system.value][
                            "normalization"
                        ],
                        smoothing=self.smoothing.value,
                        use_residual=self.residual.value,
                        min_amplitude=self.min_amplitude.value,
                        min_value=self.min_value.value,
                        min_width=self.min_width.value,
                        max_migration=self.max_migration.value,
                        min_channels=self.min_channels.value,
                        minimal_output=True,
                    )
                )
            ]

        self.all_anomalies = self.client.gather(anomalies)
        self.trigger.button_style = "success"
        self.run_all.description = "Process All Lines"

    def regroup(self):
        group_id = -1
        for group in self.lines.anomalies:

            channels = group["channels"].tolist()

            labels = [self.active_channels[ii]["name"] for ii in channels]
            match = False
            for key, time_group in self.time_groups.items():
                if labels == time_group["channels"]:
                    group["time_group"] = key
                    match = True
                    print("matchfound:", key)

                    break

            if match:
                continue
            group_id += 1
            self.time_groups[group_id] = {}
            self.time_groups[group_id]["channels"] = labels
            self.time_groups[group_id]["name"] = "-".join(labels)
            self.time_groups[group_id]["color"] = colors[group_id]

            group["time_group"] = group_id

    def trigger_click(self, _):
        """
        Observer of :obj:`geoapps.processing.PeakFinder.`:
        """
        if not self.all_anomalies:
            return

        if not self.tem_checkbox:
            self.regroup()

        # Append all lines
        (
            time_group,
            tau,
            migration,
            azimuth,
            cox,
            amplitude,
            inflx_up,
            inflx_dwn,
            start,
            end,
            skew,
            peaks,
        ) = ([], [], [], [], [], [], [], [], [], [], [], [])
        for line in self.all_anomalies:
            for group in line:
                if "time_group" in list(group.keys()) and len(group["cox"]) > 0:
                    time_group += [group["time_group"]]

                    if group["linear_fit"] is None:
                        tau += [0]
                    else:
                        tau += [np.abs(group["linear_fit"][0] ** -1.0)]
                    migration += [group["migration"]]
                    amplitude += [group["amplitude"]]
                    azimuth += [group["azimuth"]]
                    cox += [group["cox"]]
                    inflx_dwn += [group["inflx_dwn"]]
                    inflx_up += [group["inflx_up"]]
                    start += [group["start"]]
                    end += [group["end"]]
                    skew += [group["skew"]]
                    peaks += [group["peaks"]]

        if cox:
            time_group = np.hstack(time_group) + 1  # Start count at 1

            # Create reference values and color_map
            group_map, color_map = {}, []
            for ind, group in self.time_groups.items():
                group_map[ind + 1] = group["name"]
                color_map += [[ind + 1] + hex_to_rgb(group["color"]) + [1]]

            color_map = np.core.records.fromarrays(
                np.vstack(color_map).T, names=["Value", "Red", "Green", "Blue", "Alpha"]
            )
            points = Points.create(
                self.workspace,
                name="PointMarkers",
                vertices=np.vstack(cox),
                parent=self.ga_group,
            )
            points.entity_type.name = self.ga_group_name.value
            migration = np.hstack(migration)
            dip = migration / migration.max()
            dip = np.rad2deg(np.arccos(dip))
            skew = np.hstack(skew)
            azimuth = np.hstack(azimuth)
            points.add_data(
                {
                    "amplitude": {"values": np.hstack(amplitude)},
                    "skew": {"values": skew},
                }
            )

            if self.tem_checkbox.value:
                points.add_data(
                    {
                        "tau": {"values": np.hstack(tau)},
                        "azimuth": {"values": azimuth},
                        "dip": {"values": dip},
                    }
                )

            time_group_data = points.add_data(
                {
                    "time_group": {
                        "type": "referenced",
                        "values": np.hstack(time_group),
                        "value_map": group_map,
                    }
                }
            )
            time_group_data.entity_type.color_map = {
                "name": "Time Groups",
                "values": color_map,
            }

            if self.tem_checkbox.value:
                group = points.find_or_create_property_group(
                    name="AzmDip", property_group_type="Dip direction & dip"
                )
                group.properties = [
                    points.get_data("azimuth")[0].uid,
                    points.get_data("dip")[0].uid,
                ]

            # Add structural markers
            if self.structural_markers.value:

                if self.tem_checkbox.value:
                    markers = []

                    def rotation_2D(angle):
                        R = np.r_[
                            np.c_[
                                np.cos(np.pi * angle / 180),
                                -np.sin(np.pi * angle / 180),
                            ],
                            np.c_[
                                np.sin(np.pi * angle / 180), np.cos(np.pi * angle / 180)
                            ],
                        ]
                        return R

                    for azm, xyz, mig in zip(
                        np.hstack(azimuth).tolist(),
                        np.vstack(cox).tolist(),
                        migration.tolist(),
                    ):
                        marker = np.r_[
                            np.c_[-0.5, 0.0] * 50,
                            np.c_[0.5, 0] * 50,
                            np.c_[0.0, 0.0],
                            np.c_[0.0, 1.0] * mig,
                        ]

                        marker = (
                            np.c_[np.dot(rotation_2D(-azm), marker.T).T, np.zeros(4)]
                            + xyz
                        )
                        markers.append(marker.squeeze())

                    curves = Curve.create(
                        self.workspace,
                        name="TickMarkers",
                        vertices=np.vstack(markers),
                        cells=np.arange(len(markers) * 4, dtype="uint32").reshape(
                            (-1, 2)
                        ),
                        parent=self.ga_group,
                    )
                    time_group_data = curves.add_data(
                        {
                            "time_group": {
                                "type": "referenced",
                                "values": np.kron(np.hstack(time_group), np.ones(4)),
                                "value_map": group_map,
                            }
                        }
                    )
                    time_group_data.entity_type.color_map = {
                        "name": "Time Groups",
                        "values": color_map,
                    }
                inflx_pts = Points.create(
                    self.workspace,
                    name="Inflections_Up",
                    vertices=np.vstack(inflx_up),
                    parent=self.ga_group,
                )
                time_group_data = inflx_pts.add_data(
                    {
                        "time_group": {
                            "type": "referenced",
                            "values": np.repeat(
                                np.hstack(time_group), [ii.shape[0] for ii in inflx_up]
                            ),
                            "value_map": group_map,
                        }
                    }
                )
                time_group_data.entity_type.color_map = {
                    "name": "Time Groups",
                    "values": color_map,
                }
                inflx_pts = Points.create(
                    self.workspace,
                    name="Inflections_Down",
                    vertices=np.vstack(inflx_dwn),
                    parent=self.ga_group,
                )
                time_group_data.copy(parent=inflx_pts)

                start_pts = Points.create(
                    self.workspace,
                    name="Starts",
                    vertices=np.vstack(start),
                    parent=self.ga_group,
                )
                time_group_data.copy(parent=start_pts)

                end_pts = Points.create(
                    self.workspace,
                    name="Ends",
                    vertices=np.vstack(end),
                    parent=self.ga_group,
                )
                time_group_data.copy(parent=end_pts)

                peak_pts = Points.create(
                    self.workspace,
                    name="Peaks",
                    vertices=np.vstack(peaks),
                    parent=self.ga_group,
                )

        if self.live_link.value:
            self.live_link_output(points)

            if self.structural_markers.value:
                self.live_link_output(curves)

        self.workspace.finalize()

    def highlight_selection(self, _):
        """
        Observer of :obj:`geoapps.processing.PeakFinder.`:: Highlight the time group data selection
        """
        for group in self.time_groups.values():
            if group["name"] == self.group_list.value:
                self.group_color.value = group["color"]
                self.channels.value = group["channels"]

    def plot_data_selection(
        self,
        ind,
        smoothing,
        max_migration,
        min_channels,
        min_amplitude,
        min_value,
        min_width,
        residual,
        markers,
        scale,
        scale_value,
        center,
        width,
        groups,
        plot_trigger,
        x_label,
    ):
        """
        Observer of :obj:`geoapps.processing.PeakFinder.`:
        """
        if self.pause_plot_refresh:
            return

        self.line_update()

        if (
            getattr(self, "survey", None) is None
            or getattr(self.lines, "profile", None) is None
            or self.plot_trigger.value is False
            or len(self.active_channels) == 0
        ):
            return

        center = self.center.value
        width = self.width.value

        axs = None
        if (
            residual != self.lines.profile.residual
            or smoothing != self.lines.profile.smoothing
        ):
            self.lines.profile.residual = residual
            self.lines.profile.smoothing = smoothing
            self.line_update()

        if not self.tem_checkbox:
            self.regroup()

        lims = np.searchsorted(
            self.lines.profile.locations_resampled,
            [
                (center - width / 2.0),
                (center + width / 2.0),
            ],
        )
        sub_ind = np.arange(lims[0], lims[1])
        y_min, y_max = np.inf, -np.inf

        if getattr(self.survey, "line_indices", None) is None:
            return

        locs = self.lines.profile.locations_resampled
        for cc, channel in enumerate(self.active_channels):
            if axs is None:
                self.figure = plt.figure(figsize=(12, 6))
                axs = plt.subplot()

            if len(self.survey.line_indices) < 2:
                return

            self.lines.profile.values = channel["values"][self.survey.line_indices]
            values = self.lines.profile.values_resampled
            y_min = np.min([values[sub_ind].min(), y_min])
            y_max = np.max([values[sub_ind].max(), y_max])

            axs.plot(locs, values, color=[0.5, 0.5, 0.5, 1])

            # Plot the anomalies by time group color
            for group in self.lines.anomalies:
                query = np.where(group["channels"] == cc)[0]

                if (
                    len(query) == 0
                    or group["peak"][query[0]] < lims[0]
                    or group["peak"][query[0]] > lims[1]
                ):
                    continue

                ii = query[0]
                start = group["start"][ii]
                end = group["end"][ii]
                axs.plot(
                    locs[start:end],
                    values[start:end],
                    color=self.time_groups[group["time_group"]]["color"],
                )

                if group["azimuth"] < 180:
                    ori = "right"
                else:
                    ori = "left"

                if markers:
                    axs.scatter(
                        locs[group["peak"][ii]],
                        values[group["peak"][ii]],
                        s=200,
                        c=self.time_groups[group["time_group"]]["color"],
                        marker=self.marker[ori],
                    )

                    axs.scatter(
                        locs[group["start"][ii]],
                        values[group["start"][ii]],
                        s=100,
                        color=np.c_[0, 0, 0, 0],
                        edgecolors="b",
                        marker="o",
                    )
                    axs.scatter(
                        locs[group["end"][ii]],
                        values[group["end"][ii]],
                        s=100,
                        color=np.c_[0, 0, 0, 0],
                        edgecolors="c",
                        marker="o",
                    )
                    axs.scatter(
                        locs[group["inflx_up"][ii]],
                        values[group["inflx_up"][ii]],
                        color="k",
                        marker="1",
                        s=100,
                    )
                    axs.scatter(
                        locs[group["inflx_dwn"][ii]],
                        values[group["inflx_dwn"][ii]],
                        color="k",
                        marker="2",
                        s=100,
                    )

            if not residual:
                raw = self.lines.profile._values_resampled_raw
                axs.fill_between(
                    locs, values, raw, where=raw > values, color=[1, 0, 0, 0.5]
                )
                axs.fill_between(
                    locs, values, raw, where=raw < values, color=[0, 0, 1, 0.5]
                )

        if scale == "symlog":
            plt.yscale("symlog", linthreshy=scale_value)

        x_lims = [
            center - width / 2.0,
            center + width / 2.0,
        ]
        axs.set_xlim(x_lims)
        axs.set_ylim([np.max([y_min, min_value]), y_max])
        axs.set_ylabel("Data")

        axs.scatter(center, (y_min + y_max) / 2, s=100, c="k", marker="d")
        if x_label == "Easting":
            axs.text(
                center,
                y_min,
                f"{self.lines.profile.interp_x(center):.0f} m E",
                va="top",
                ha="center",
                bbox={"edgecolor": "r"},
            )
            xlbl = [
                f"{self.lines.profile.interp_x(label):.0f}"
                for label in axs.get_xticks()
            ]
            axs.set_xticklabels(xlbl)
            axs.set_xlabel("Easting (m)")

        elif x_label == "Northing":
            axs.text(
                center,
                y_min,
                f"{self.lines.profile.interp_y(center):.0f} m N",
                va="top",
                ha="center",
                bbox={"edgecolor": "r"},
            )
            xlbl = [
                f"{self.lines.profile.interp_y(label):.0f}"
                for label in axs.get_xticks()
            ]
            axs.set_xticklabels(xlbl)
            axs.set_xlabel("Northing (m)")

        else:
            axs.text(
                center,
                y_min,
                f"{center:.0f} m",
                va="top",
                ha="center",
                bbox={"edgecolor": "r"},
            )
            axs.set_xlabel("Distance (m)")

        axs.grid(True)

    def plot_decay_curve(self, ind, smoothing, residual, center, groups, plot_trigger):
        """
        Observer of :obj:`geoapps.processing.PeakFinder.`:
        """
        if self.pause_plot_refresh:
            return

        if (
            self.plot_trigger.value
            and hasattr(self.lines, "profile")
            and self.tem_checkbox.value
        ):

            center = self.center.value
            # Loop through groups and find nearest to cursor
            dist = np.inf
            group = None
            for anomaly in self.lines.anomalies:
                delta_x = np.abs(
                    center - self.lines.profile.locations_resampled[anomaly["peak"][0]]
                )
                if delta_x < dist and anomaly["linear_fit"] is not None:
                    dist = delta_x
                    group = anomaly

            # Get the times of the group and plot the linear regression
            times = []
            if group is not None and group["linear_fit"] is not None:
                times = [
                    channel["time"]
                    for ii, channel in enumerate(self.active_channels)
                    if ii in list(group["channels"])
                ]
            if any(times):
                times = np.hstack(times)

                if self.decay_figure is None:
                    self.decay_figure = plt.figure(figsize=(8, 8))

                else:
                    plt.figure(self.decay_figure.number)

                axs = plt.subplot()
                y = np.exp(times * group["linear_fit"][1] + group["linear_fit"][0])
                axs.plot(
                    times,
                    y,
                    "--",
                    linewidth=2,
                    color="k",
                )
                axs.text(
                    np.mean(times),
                    np.mean(y),
                    f"Tau: {np.abs(group['linear_fit'][1] ** -1.)*1e+3:.2e} msec",
                    color="k",
                )
                axs.scatter(
                    times,
                    group["peak_values"],
                    s=100,
                    color=self.time_groups[group["time_group"]]["color"],
                    marker="^",
                    edgecolors="k",
                )
                axs.grid(True)

                plt.yscale("log")
                axs.yaxis.set_label_position("right")
                axs.yaxis.tick_right()
                axs.set_ylabel("log(V)")
                axs.set_xlabel("Time (sec)")
                axs.set_title("Decay - MADTau")

    def plot_model_selection(
        self,
        ind,
        center,
        width,
        x_label,
        colormap,
        log,
        min,
        max,
        reverse,
        opacity,
        model,
        smoothing,
        slice_width,
        doi_show,
        doi,
        doi_percent,
        doi_revert,
        borehole_show,
        borehole_object,
        borehole_data,
        boreholes_size,
        max_migration,
        min_channels,
        min_amplitude,
        min_value,
        min_width,
        plot_trigger,
    ):
        """
        Observer of :obj:`geoapps.processing.PeakFinder.`:
        """
        if (
            getattr(self, "survey", None) is None
            or getattr(self.lines, "profile", None) is None
            or self.show_model.value is False
            or getattr(self.lines, "model_vertices", None) is None
            or plot_trigger is False
            or self.pause_plot_refresh
        ):
            return

        self.update_line_model()
        self.update_line_boreholes()

        if reverse:
            colormap += "_r"

        if (
            getattr(self.lines, "model_vertices", None) is not None
            and getattr(self.lines, "model_values", None) is not None
        ):
            tree = cKDTree(self.lines.model_vertices)

            # Create dip marker
            center_x = float(self.lines.profile.interp_x(center))
            center_y = float(self.lines.profile.interp_y(center))
            center_z = float(self.lines.profile.interp_z(center))

            _, ind = tree.query(np.c_[center_x, center_y, center_z])

            self.model_figure.data[0].x = self.lines.model_vertices[ind, 0]
            self.model_figure.data[0].y = self.lines.model_vertices[ind, 1]
            self.model_figure.data[0].z = self.lines.model_vertices[ind, 2]
            self.model_figure.data[0].mode = "markers"
            self.model_figure.data[0].marker = {
                "symbol": "diamond",
                "color": "red",
                "size": 10,
            }

            cox, azimuth, dip = [], [], []
            locs = self.lines.profile.locations_resampled
            for group in self.lines.anomalies:
                _, ind = tree.query(
                    np.c_[
                        self.lines.profile.interp_x(locs[group["peak"][0]]),
                        self.lines.profile.interp_y(locs[group["peak"][0]]),
                        self.lines.profile.interp_z(locs[group["peak"][0]]),
                    ]
                )

                cox += [
                    np.c_[
                        self.lines.model_vertices[ind, 0],
                        self.lines.model_vertices[ind, 1],
                        self.lines.model_vertices[ind, 2],
                    ]
                ]
                azimuth += [group["azimuth"]]
                dip += [group["migration"]]

            self.model_figure.data[1].x = []
            self.model_figure.data[1].y = []
            self.model_figure.data[1].z = []

            if len(cox) > 0:
                dip = np.hstack(dip)
                dip /= dip.max() + 1e-4
                dip = np.rad2deg(np.arcsin(dip))

                vec = rotate_azimuth_dip(np.hstack(azimuth), dip)
                cox = np.vstack(cox)
                scaler = 100
                self.model_figure.data[1].x = cox[:, 0]
                self.model_figure.data[1].y = cox[:, 1]
                self.model_figure.data[1].z = cox[:, 2]
                self.model_figure.data[1].u = vec[:, 0] * scaler
                self.model_figure.data[1].v = vec[:, 1] * scaler
                self.model_figure.data[1].w = vec[:, 2] * scaler
                self.model_figure.data[1].colorscale = [
                    [0, "rgb(0,0,0)"],
                    [1, "rgb(0,0,0)"],
                ]
                self.model_figure.data[1].showscale = False

            simplices = self.lines.model_cells.reshape((-1, 3))

            if log:
                model_values = np.log10(self.lines.model_values)
                min = np.log10(min)
                max = np.log10(max)
            else:
                model_values = self.lines.model_values

            if self.show_doi.value:
                model_values[self.lines.doi_values > doi_percent] = np.nan

            self.model_figure.data[2].x = self.lines.model_vertices[:, 0]
            self.model_figure.data[2].y = self.lines.model_vertices[:, 1]
            self.model_figure.data[2].z = self.lines.model_vertices[:, 2]
            self.model_figure.data[2].intensity = model_values
            self.model_figure.data[2].opacity = opacity
            self.model_figure.data[2].i = simplices[:, 0]
            self.model_figure.data[2].j = simplices[:, 1]
            self.model_figure.data[2].k = simplices[:, 2]
            self.model_figure.data[2].colorscale = colormap
            self.model_figure.data[2].cmin = min
            self.model_figure.data[2].cmax = max

            if (
                getattr(self.lines, "borehole_vertices", None) is not None
                and self.show_borehole.value
            ):
                self.model_figure.data[3].visible = True
                self.model_figure.data[3].x = self.lines.borehole_vertices[:, 0]
                self.model_figure.data[3].y = self.lines.borehole_vertices[:, 1]
                self.model_figure.data[3].z = self.lines.borehole_vertices[:, 2]
                self.model_figure.data[3].mode = "markers"
                self.model_figure.data[3].marker.size = boreholes_size

                if getattr(self.lines, "borehole_values", None) is not None:
                    self.model_figure.data[3].marker.color = self.lines.borehole_values
            else:
                self.model_figure.data[3].visible = False

    def scale_update(self, _):
        """
        Observer of :obj:`geoapps.processing.PeakFinder.`:
        """
        if self.scale_button.value == "symlog":
            self.scale_panel.children = [
                self.scale_button,
                self.scale_value,
            ]
        else:
            self.scale_panel.children = [self.scale_button]

    def line_update(self):
        """
        Re-compute derivatives
        """

        if getattr(self, "survey", None) is None:
            return

        if (
            len(self.survey.get_data(self.lines.data.value)) == 0
            or self.lines.lines.value == ""
        ):
            return

        line_indices = self.get_line_indices(self.lines.lines.value)

        if line_indices is None:
            return
        self.survey.line_indices = line_indices
        result = self.client.compute(
            find_anomalies(
                self.survey.vertices,
                line_indices,
                self.active_channels,
                self.time_groups,
                data_normalization=self.em_system_specs[self.system.value][
                    "normalization"
                ],
                smoothing=self.smoothing.value,
                use_residual=self.residual.value,
                min_amplitude=self.min_amplitude.value,
                min_value=self.min_value.value,
                min_width=self.min_width.value,
                max_migration=self.max_migration.value,
                min_channels=self.min_channels.value,
                return_profile=True,
            )
        ).result()

        if len(result) > 0:
            self.lines.anomalies, self.lines.profile = result
        else:
            return

        self.pause_plot_refresh = True

        if self.previous_line != self.lines.lines.value:
            end = self.lines.profile.locations_resampled[-1]
            mid = self.lines.profile.locations_resampled[-1] * 0.5

            if self.center.value >= end:
                self.center.value = 0
                self.center.max = end
                self.center.value = mid
            else:
                self.center.max = end

        if self.previous_line != self.lines.lines.value:
            end = self.lines.profile.locations_resampled[-1]
            mid = self.lines.profile.locations_resampled[-1] * 0.5
            if self.width.value >= end:
                self.width.value = 0
                self.width.max = end
                self.width.value = mid
            else:
                self.width.max = end

        self.previous_line = self.lines.lines.value
        if self.show_model.value:
            self.update_line_model()

        self.pause_plot_refresh = False

    def get_line_indices(self, line_id):
        """
        Find the vertices for a given line ID
        """
        line_data = self.survey.get_data(self.lines.data.value)[0]

        if isinstance(line_data, ReferencedData):
            line_id = [
                key
                for key, value in line_data.value_map.map.items()
                if value == line_id
            ]

            if line_id:
                line_id = line_id[0]

        indices = np.where(np.asarray(line_data.values) == line_id)[0]

        if len(indices) == 0:
            return

        return indices

    def update_line_model(self):
        if getattr(self.lines, "profile", None) is None:
            return

        entity_name = self.model_selection.objects.value
        if self.workspace.get_entity(entity_name) and (
            getattr(self, "surface_model", None) is None
            or self.surface_model.name != entity_name
        ):

            self.show_model.description = "Processing line ..."
            self.surface_model = self.workspace.get_entity(entity_name)[0]
            self.surface_model.tree = cKDTree(self.surface_model.vertices[:, :2])

        if getattr(self, "surface_model", None) is None:
            return

        if (
            getattr(self.lines.profile, "line_id", None) is None
            or self.lines.profile.line_id != self.lines.lines.value
        ):

            lims = [
                (self.center.value - self.width.value / 2.0),
                (self.center.value + self.width.value / 2.0),
            ]
            x_locs = self.lines.profile.x_locs
            y_locs = self.lines.profile.y_locs
            z_locs = self.lines.profile.z_locs
            xyz = np.c_[x_locs, y_locs, z_locs]

            ind = (
                (xyz[:, 0] > self.lines.profile.interp_x(lims[0]))
                * (xyz[:, 0] < self.lines.profile.interp_x(lims[1]))
                * (xyz[:, 1] > self.lines.profile.interp_y(lims[0]))
                * (xyz[:, 1] < self.lines.profile.interp_y(lims[1]))
            )

            tree = cKDTree(xyz[ind, :2])
            ind = tree.query_ball_tree(self.surface_model.tree, self.slice_width.value)

            indices = np.zeros(self.surface_model.n_vertices, dtype="bool")

            indices[np.r_[np.hstack(ind)].astype("int")] = True

            cells_in = indices[self.surface_model.cells].reshape((-1, 3))
            cells = self.surface_model.cells[np.any(cells_in, axis=1), :]
            vert_ind, cell_ind = np.unique(np.asarray(cells), return_inverse=True)

            # Give new indices to subset
            self.lines.model_vertices = self.surface_model.vertices[vert_ind, :]
            self.lines.model_cells = cell_ind
            self.lines.vertices_map = vert_ind
            # Save the current line id to skip next time
            self.lines.profile.line_id = self.lines.lines.value

        if self.surface_model.get_data(self.model_selection.data.value):
            self.lines.model_values = self.surface_model.get_data(
                self.model_selection.data.value
            )[0].values[self.lines.vertices_map]

        doi_values = np.zeros_like(self.lines.vertices_map)
        if self.surface_model.get_data(self.doi_selection.data.value):

            doi_values = self.surface_model.get_data(self.doi_selection.data.value)[
                0
            ].values[self.lines.vertices_map]
        elif self.doi_selection.data.value == "Z":
            doi_values = self.surface_model.vertices[:, 2][self.lines.vertices_map]

        if np.any(doi_values != 0):
            doi_values -= doi_values.min()
            doi_values /= doi_values.max()
            doi_values *= 100.0

            if self.doi_revert.value:
                doi_values = np.abs(100 - doi_values)

        self.lines.doi_values = doi_values

        if self.show_model.value:
            self.show_model.description = "Hide model"
        else:
            self.show_model.description = "Show model"

    def update_line_boreholes(self):
        if getattr(self.lines, "profile", None) is None:
            return

        entity_name = self.boreholes.objects.value
        if self.workspace.get_entity(entity_name) and (
            getattr(self, "borehole_trace", None) is None
            or self.borehole_trace.name != entity_name
        ):
            self.borehole_trace = self.workspace.get_entity(entity_name)[0]

        if getattr(self, "borehole_trace", None) is None:
            return

        if getattr(self.lines, "model_vertices", None) is not None:

            tree = cKDTree(self.lines.model_vertices)
            rad, ind = tree.query(self.borehole_trace.vertices)

            # Give new indices to subset
            if np.any(rad < self.slice_width.value):
                in_slice = rad < self.slice_width.value
                self.lines.borehole_vertices = self.borehole_trace.vertices[in_slice, :]

                if self.borehole_trace.get_data(self.boreholes.data.value):
                    self.lines.borehole_values = self.borehole_trace.get_data(
                        self.boreholes.data.value
                    )[0].values[in_slice]
                else:
                    self.lines.borehole_values = "black"
            else:
                self.lines.borehole_vertices = None

    def reset_groups(self, gates={}):
        if gates:
            for key, values in gates.items():
                if key in list(self.time_groups.keys()):
                    self.time_groups[key]["channels"] = values

        else:
            self._time_groups = None

            if self.tem_checkbox.value:
                start = self.em_system_specs[self.system.value]["channel_start_index"]
                end = (
                    len(self.em_system_specs[self.system.value]["channels"].keys()) + 1
                )

                # Divide channels in three equal blocks
                block = int((end - start) / 3)
                early = np.arange(start, start + block).tolist()
                mid = np.arange(start + block, start + 2 * block).tolist()
                late = np.arange(start + 2 * block, end).tolist()

                gates_list = [[], [], []]
                try:
                    for channel in self.channels.options:
                        [
                            gates_list[ii].append(channel)
                            for ii, block in enumerate([early, mid, late])
                            if int(re.findall(r"\d+", channel)[-1]) in block
                        ]

                    for key, gates in enumerate(gates_list):
                        self.time_groups[key]["channels"] = gates

                except IndexError:
                    print(
                        "Could not find a time channel for the given list of time channels. "
                        "Switching to non-tem mode."
                    )
                    self.tem_checkbox.value = False

            else:  # One group per selected channel
                self._time_groups = {}
                for ii, channel in enumerate(self.channels.options):
                    self._time_groups[ii] = {
                        "name": channel,
                        "label": [ii],
                        "color": colors[ii],
                        "channels": [channel],
                    }

        channels = []
        for group in self.time_groups.values():
            [
                channels.append(channel)
                for channel in group["channels"]
                if channel not in channels
            ]

        self.active_channels = [{"name": c} for c in channels]
        d_min, d_max = np.inf, -np.inf
        thresh_value = np.inf
        for channel in self.active_channels:

            try:
                if self.tem_checkbox.value:
                    gate = int(re.findall(r"\d+", channel["name"])[-1])
                    channel["time"] = (
                        self.data_channel_options[f"[{gate}]"].children[1].value
                    )
                channel["values"] = (-1.0) ** self.flip_sign.value * self.data_channels[
                    channel["name"]
                ].values.copy()

            except KeyError:
                continue
            thresh_value = np.min(
                [thresh_value, np.percentile(np.abs(channel["values"]), 95)]
            )
            d_min = np.min([d_min, channel["values"].min()])
            d_max = np.max([d_max, channel["values"].max()])

        if d_max > -np.inf:
            self.plot_trigger.value = False
            self.min_value.value = d_min
            self.scale_value.value = thresh_value

    def show_model_trigger(self, _):
        """
        Observer of :obj:`geoapps.processing.PeakFinder.`: Add the model widget
        """
        if self.show_model.value:
            self.model_panel.children = [
                self.show_model,
                self.model_figure,
                HBox([self.model_parameters, self.doi_panel, self.borehole_panel]),
            ]
            self.show_model.description = "Hide model"
            self.plot_trigger.value = False
            self.plot_trigger.value = True
        else:
            self.model_panel.children = [self.show_model]
            self.show_model.description = "Show model"

    def show_decay_trigger(self, _):
        """
        Observer of :obj:`geoapps.processing.PeakFinder.`: Add the decay curve plot
        """
        if self.show_decay.value:
            self.decay_panel.children = [self.show_decay, self.decay]
            self.show_decay.description = "Hide decay curve"
        else:
            self.decay_panel.children = [self.show_decay]
            self.show_decay.description = "Show decay curve"

    def show_doi_trigger(self, _):
        """
        Observer of :obj:`geoapps.processing.PeakFinder.`: Add the DOI options
        """
        if self.show_doi.value:
            self.doi_panel.children = [self.show_doi, self.doi_parameters]
            self.show_doi.description = "Hide DOI"
        else:
            self.doi_panel.children = [self.show_doi]
            self.show_doi.description = "Show DOI"

    def show_borehole_trigger(self, _):
        """
        Observer of :obj:`geoapps.processing.PeakFinder.`: Add the DOI options
        """
        if self.show_borehole.value:
            self.borehole_panel.children = [self.show_borehole, self.scatter_parameters]
            self.show_borehole.description = "Hide Scatter"
        else:
            self.borehole_panel.children = [self.show_borehole]
            self.show_borehole.description = "Show Scatter"
        return


@dask.delayed
def find_anomalies(
    locations,
    line_indices,
    channels,
    time_groups,
    smoothing=1,
    use_residual=False,
    data_normalization=[1],
    min_amplitude=25,
    min_value=-np.inf,
    min_width=200,
    max_migration=50,
    min_channels=3,
    minimal_output=False,
    return_profile=False,
):
    """
    Find all anomalies along a line profile of data.
    Anomalies are detected based on the lows, inflection points and a peaks.
    Neighbouring anomalies are then grouped and assigned a time_group label.

    :param: :obj:`geoh5py.objects.Curve`
        Curve object containing data.
    :param: list
        List of Data channels
    :param: array of int or bool
        Array defining a line of data from the input Curve object


    :return: list of dict
    """
    profile = LineDataDerivatives(
        locations=locations[line_indices], smoothing=smoothing, residual=use_residual
    )
    locs = profile.locations_resampled

    if data_normalization == "ppm":
        data_normalization = [1e-6]

    if locs is None:
        return {}

    xy = np.c_[profile.interp_x(locs), profile.interp_y(locs)]
    angles = np.arctan2(xy[1:, 1] - xy[:-1, 1], xy[1:, 0] - xy[:-1, 0])
    angles = np.r_[angles[0], angles].tolist()
    azimuth = (450.0 - np.rad2deg(running_mean(angles, width=5))) % 360.0
    anomalies = {
        "channel": [],
        "start": [],
        "inflx_up": [],
        "peak": [],
        "peak_values": [],
        "inflx_dwn": [],
        "end": [],
        "amplitude": [],
        "group": [],
        "time_group": [],
    }
    for cc, channel in enumerate(channels):
        if "values" not in list(channel.keys()):
            continue
        values = channel["values"][line_indices].copy()
        profile.values = values
        values = profile.values_resampled

        dx = profile.derivative(order=1)
        ddx = profile.derivative(order=2)
        peaks = np.where(
            (np.diff(np.sign(dx)) != 0) & (ddx[1:] < 0) & (values[:-1] > min_value)
        )[0]
        lows = np.where(
            (np.diff(np.sign(dx)) != 0) & (ddx[1:] > 0) & (values[:-1] > min_value)
        )[0]

        # Add end of line as possible bump limits
        lows = np.r_[0, lows, locs.shape[0] - 1]

        up_inflx = np.where(
            (np.diff(np.sign(ddx)) != 0) & (dx[1:] > 0) & (values[:-1] > min_value)
        )[0]
        dwn_inflx = np.where(
            (np.diff(np.sign(ddx)) != 0) & (dx[1:] < 0) & (values[:-1] > min_value)
        )[0]

        if len(peaks) == 0 or len(lows) < 2 or len(up_inflx) < 2 or len(dwn_inflx) < 2:
            continue

        for peak in peaks:
            ind = np.median(
                [0, lows.shape[0] - 1, np.searchsorted(locs[lows], locs[peak]) - 1]
            ).astype(int)
            start = lows[ind]
            ind = np.median(
                [0, lows.shape[0] - 1, np.searchsorted(locs[lows], locs[peak])]
            ).astype(int)
            end = np.min([locs.shape[0] - 1, lows[ind]])
            ind = np.median(
                [
                    0,
                    up_inflx.shape[0] - 1,
                    np.searchsorted(locs[up_inflx], locs[peak]) - 1,
                ]
            ).astype(int)
            inflx_up = up_inflx[ind]
            ind = np.median(
                [
                    0,
                    dwn_inflx.shape[0] - 1,
                    np.searchsorted(locs[dwn_inflx], locs[peak]),
                ]
            ).astype(int)
            inflx_dwn = np.min([locs.shape[0] - 1, dwn_inflx[ind] + 1])

            # Check amplitude and width thresholds
            delta_amp = (
                np.abs(values[peak] - np.min([values[start], values[end]]))
                / (np.abs(np.min([values[start], values[end]])) + 2e-32)
            ) * 100.0
            delta_x = locs[end] - locs[start]
            amplitude = np.sum(np.abs(values[start:end])) * profile.sampling
            if (delta_amp > min_amplitude) & (delta_x > min_width):
                anomalies["channel"] += [cc]
                anomalies["start"] += [start]
                anomalies["inflx_up"] += [inflx_up]
                anomalies["peak"] += [peak]
                anomalies["peak_values"] += [values[peak]]
                anomalies["inflx_dwn"] += [inflx_dwn]
                anomalies["amplitude"] += [amplitude]
                anomalies["end"] += [end]
                anomalies["group"] += [-1]
                anomalies["time_group"] += [
                    key
                    for key, time_group in time_groups.items()
                    if channel["name"] in time_group["channels"]
                ]

    if len(anomalies["peak"]) == 0:
        if return_profile:
            return {}, profile
        else:
            return {}

    groups = []

    # Re-cast as numpy arrays
    for key, values in anomalies.items():
        anomalies[key] = np.hstack(values)

    group_id = -1
    peaks_position = locs[anomalies["peak"]]
    for ii in range(peaks_position.shape[0]):
        # Skip if already labeled
        if anomalies["group"][ii] != -1:
            continue

        group_id += 1  # Increment group id

        dist = np.abs(peaks_position[ii] - peaks_position)

        # Find anomalies across channels within horizontal range
        near = np.where((dist < max_migration) & (anomalies["group"] == -1))[0]

        # Reject from group if channel gap >1
        u_gates, u_count = np.unique(anomalies["channel"][near], return_counts=True)
        if len(u_gates) > 1 and np.any((u_gates[1:] - u_gates[:-1]) > 2):

            cutoff = u_gates[np.where((u_gates[1:] - u_gates[:-1]) > 2)[0][0]]
            near = near[anomalies["channel"][near] > cutoff, ...]  # Remove after cutoff

        # Check for multiple nearest peaks on single channel
        # and keep the nearest
        u_gates, u_count = np.unique(anomalies["channel"][near], return_counts=True)
        for gate in u_gates[np.where(u_count > 1)]:
            mask = np.ones_like(near, dtype="bool")
            sub_ind = anomalies["channel"][near] == gate
            sub_ind[np.where(sub_ind)[0][np.argmin(dist[near][sub_ind])]] = False
            mask[sub_ind] = False
            near = near[mask, ...]

        # Keep largest overlapping time group
        in_gate, count = np.unique(anomalies["time_group"][near], return_counts=True)
        in_gate = in_gate[(count >= min_channels) & (in_gate != -1)].tolist()

        time_group = [
            ii
            for ii, group in enumerate(time_groups.values())
            if all([(gate in in_gate) for gate in group["label"]])
        ]

        if len(time_group) > 0:
            time_group = np.max(time_group)
        else:
            continue

        # Remove anomalies not in group
        channel_list = [
            time_groups[ii]["channels"] for ii in time_groups[time_group]["label"]
        ]
        mask = [
            ii
            for ii, id in enumerate(near)
            if channels[anomalies["channel"][id]] not in channel_list
        ]
        near = near[mask, ...]

        anomalies["group"][near] = group_id

        gates = anomalies["channel"][near]
        cox = anomalies["peak"][near]
        inflx_dwn = anomalies["inflx_dwn"][near]
        inflx_up = anomalies["inflx_up"][near]
        cox_sort = np.argsort(locs[cox])
        azimuth_near = azimuth[cox]
        dip_direction = azimuth[cox[0]]

        if cox_sort[-1] < cox_sort[0]:
            dip_direction = (dip_direction + 180) % 360.0

        migration = np.abs(locs[cox[cox_sort[-1]]] - locs[cox[cox_sort[0]]])
        skew = (locs[cox][cox_sort[0]] - locs[inflx_up][cox_sort]) / (
            locs[inflx_dwn][cox_sort] - locs[cox][cox_sort[0]]
        )
        skew[azimuth_near[cox_sort] > 180] = 1.0 / (
            skew[azimuth_near[cox_sort] > 180] + 1e-2
        )

        # Change skew factor from [-100, 1]
        flip_skew = skew < 1
        skew[flip_skew] = 1.0 / (skew[flip_skew] + 1e-2)
        skew = 1.0 - skew
        skew[flip_skew] *= -1

        values = anomalies["peak_values"][near] * np.prod(data_normalization)
        amplitude = np.sum(anomalies["amplitude"][near])
        times = [
            channel["time"]
            for ii, channel in enumerate(channels)
            if (ii in list(gates) and "time" in channel.keys())
        ]
        linear_fit = None

        if len(times) > 2 and len(cox) > 0:
            times = np.hstack(times)[values > 0]
            if len(times) > 2:
                # Compute linear trend
                A = np.c_[np.ones_like(times), times]
                y0, slope = np.linalg.solve(
                    np.dot(A.T, A), np.dot(A.T, np.log(values[values > 0]))
                )
                linear_fit = [y0, slope]

        group = {
            "channels": gates,
            "start": anomalies["start"][near],
            "inflx_up": anomalies["inflx_up"][near],
            "peak": cox,
            "cox": np.mean(
                np.c_[
                    profile.interp_x(locs[cox[cox_sort[0]]]),
                    profile.interp_y(locs[cox[cox_sort[0]]]),
                    profile.interp_z(locs[cox[cox_sort[0]]]),
                ],
                axis=0,
            ),
            "inflx_dwn": anomalies["inflx_dwn"][near],
            "end": anomalies["end"][near],
            "azimuth": dip_direction,
            "migration": migration,
            "amplitude": amplitude,
            "time_group": time_group,
            "linear_fit": linear_fit,
        }
        if minimal_output:

            group["skew"] = np.mean(skew)
            group["inflx_dwn"] = np.c_[
                profile.interp_x(locs[inflx_dwn]),
                profile.interp_y(locs[inflx_dwn]),
                profile.interp_z(locs[inflx_dwn]),
            ]
            group["inflx_up"] = np.c_[
                profile.interp_x(locs[inflx_up]),
                profile.interp_y(locs[inflx_up]),
                profile.interp_z(locs[inflx_up]),
            ]
            start = anomalies["start"][near]
            group["start"] = np.c_[
                profile.interp_x(locs[start]),
                profile.interp_y(locs[start]),
                profile.interp_z(locs[start]),
            ]

            end = anomalies["end"][near]
            group["peaks"] = np.c_[
                profile.interp_x(locs[cox]),
                profile.interp_y(locs[cox]),
                profile.interp_z(locs[cox]),
            ]

            group["end"] = np.c_[
                profile.interp_x(locs[end]),
                profile.interp_y(locs[end]),
                profile.interp_z(locs[end]),
            ]

        else:
            group["peak_values"] = values

        groups += [group]

    if return_profile:
        return groups, profile
    else:
        return groups
