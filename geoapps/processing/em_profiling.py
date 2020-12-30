import os
import re
import numpy as np
from scipy.spatial import cKDTree
import plotly.graph_objects as go
import plotly.express as px
import time
from dask.distributed import get_client
import dask
import matplotlib.pyplot as plt
from geoh5py.workspace import Workspace
from geoh5py.objects import Points, Curve, Surface
from geoh5py.groups import ContainerGroup
from ipywidgets import (
    Button,
    Dropdown,
    ColorPicker,
    SelectMultiple,
    Text,
    IntSlider,
    Checkbox,
    FloatSlider,
    FloatText,
    VBox,
    HBox,
    Box,
    ToggleButton,
    ToggleButtons,
    interactive_output,
    FloatLogSlider,
    Label,
    Layout,
    RadioButtons,
)
from geoapps.base import BaseApplication
from geoapps.utils import (
    find_value,
    geophysical_systems,
    signal_processing_1d,
    rotate_azimuth_dip,
    running_mean,
    hex_to_rgb,
)
from geoapps.selection import ObjectDataSelection, LineOptions


class EMLineProfiler(ObjectDataSelection):
    """
    Application for the picking of targets along Time-domain EM profiles
    """

    defaults = {
        "object_types": Curve,
        "h5file": "../../assets/FlinFlon.geoh5",
        "add_groups": True,
        "time_groups": {
            0: {"name": "early", "label": [0], "color": "#0000FF"},
            1: {"name": "middle", "label": [1], "color": "#FFFF00"},
            2: {"name": "late", "label": [2], "color": "#FF0000"},
            3: {"name": "early + middle", "label": [0, 1], "color": "#00FFFF"},
            4: {
                "name": "early + middle + late",
                "label": [0, 1, 2],
                "color": "#008000",
            },
            5: {"name": "middle + late", "label": [1, 2], "color": "#FFA500"},
        },
        "objects": "Data_TEM_pseudo3D",
        "data": ["Observed"],
        "model": {"objects": "Inversion_VTEM_Model", "data": "Iteration_7_model"},
        "lines": {"data": "Line", "lines": 6073400.0},
        "boreholes": {"objects": "geochem", "data": "Al2O3"},
        "doi": {"data": "Z"},
        "doi_percent": 60,
        "doi_revert": True,
        "center": 0.35,
        "width": 0.28,
        "smoothing": 6,
        # "show_model": False,
        # "show_borehole": False,
        "markers": True,
        # "show_doi": False,
        "slice_width": 25,
        "x_label": "Distance",
        "ga_group_name": "PeakFinder",
    }

    def __init__(self, **kwargs):
        self.client = get_client()
        kwargs = self.apply_defaults(**kwargs)
        self.all_anomalies = []
        self.borehole_trace = None
        self.data_channels = {}
        self.data_channel_options = {}
        self.em_system_specs = geophysical_systems.parameters()
        self.marker = {"left": "<", "right": ">"}
        self.pause_plot_refresh = False
        self.surface_model = None
        self._survey = None
        self._time_groups = {}
        self.objects.observe(self.objects_change, names="value")
        self.model_panel = VBox([self.show_model])
        self.show_model.observe(self.show_model_trigger, names="value")
        self.doi_panel = VBox([self.show_doi])
        self.show_doi.observe(self.show_doi_trigger, names="value")
        self.borehole_panel = VBox([self.show_borehole])
        self.show_borehole.observe(self.show_borehole_trigger, names="value")

        super().__init__(**kwargs)

        self.objects.description = "Survey"

        if "lines" in kwargs.keys():
            self.lines.__populate__(**kwargs["lines"])

        if "boreholes" in kwargs.keys():
            self.boreholes.__populate__(**kwargs["boreholes"])
        self.boreholes.objects.description = "Borehole"
        self.boreholes.data.description = "Log"

        if "model" in kwargs.keys():
            self.model_selection.__populate__(**kwargs["model"])
        self.model_selection.objects.description = "Surface:"
        self.model_selection.data.description = "Model"

        self.doi_selection.update_data_list(None)
        if "doi" in kwargs.keys():
            self.doi_selection.__populate__(**kwargs["doi"])
        self.doi_selection.data.description = "DOI Layer"
        self.scale_panel = VBox([self.scale_button])
        self.scale_button.observe(self.scale_update)
        self.system.observe(self.system_observer, names="value")
        self.system_observer(None)
        self.channel_selection.observe(self.channel_panel_update, names="value")
        self.channel_panel = VBox(
            [
                self.channel_selection,
                self.data_channel_options[self.channel_selection.value],
            ]
        )
        self.system_box_option = ToggleButton(
            description="Select TEM System", value=False
        )
        self.system_box_option.observe(self.system_box_trigger)
        self.system_box = VBox([self.system_box_option])
        self.channels.observe(self.edit_group, names="value")
        self.group_list.observe(self.highlight_selection, names="value")
        self.highlight_selection(None)
        self.groups_setter.observe(self.groups_trigger)
        self.groups_widget = VBox([self.groups_setter])
        self.run_all.on_click(self.run_all_click)
        self.trigger.on_click(self.trigger_click)
        self.trigger.description = "Export Peaks"

        self.trigger_panel = VBox(
            [
                VBox([self.trigger, self.structural_markers, self.ga_group_name]),
                self.live_link_panel,
            ]
        )
        self.ga_group_name.description = "Save As"
        plotting = interactive_output(
            self.plot_data_selection,
            {
                "data": self.data,
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
                "threshold": self.threshold,
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
                "threshold": self.threshold,
            },
        )
        self.decay_panel = VBox([self.show_decay])
        self.show_decay.observe(self.show_decay_trigger, names="value")
        self.model_section = interactive_output(
            self.plot_model_selection,
            {
                "ind": self.lines.lines,
                "center": self.center,
                "width": self.width,
                "objects": self.model_selection.objects,
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
        self._widget = VBox(
            [
                self.project_panel,
                HBox(
                    [
                        VBox([self.widget,], layout=Layout(width="50%"),),
                        VBox(
                            [self.system_box, self.groups_widget, self.decay_panel,],
                            layout=Layout(width="50%"),
                        ),
                    ],
                ),
                Box(
                    children=[self.lines.widget],
                    layout=Layout(
                        display="flex",
                        flex_flow="row",
                        align_items="stretch",
                        width="100%",
                        justify_content="center",
                    ),
                ),
                HBox(
                    [
                        VBox(
                            [
                                Label("Detection Parameters"),
                                self.smoothing,
                                self.max_migration,
                                self.min_channels,
                                self.min_amplitude,
                                self.min_value,
                                self.min_width,
                                self.residual,
                                Label("Visual Parameters"),
                                self.center,
                                self.width,
                                self.x_label,
                                self.scale_panel,
                                self.markers,
                            ],
                            layout=Layout(width="50%"),
                        ),
                        VBox([plotting]),
                    ]
                ),
                self.model_panel,
                self.run_all,
                self.trigger_panel,
            ]
        )
        self.plot_trigger.value = True

    @property
    def plot_trigger(self):
        if getattr(self, "_plot_trigger", None) is None:
            self._plot_trigger = ToggleButton(
                description="Pick nearest target", value=False
            )

        return self._plot_trigger

    # @property
    # def azimuth_rotation(self):
    #     if getattr(self, "_azimuth_rotation", None) is None:
    #         self._azimuth_rotation = FloatSlider(
    #             value=0,
    #             min=0,
    #             max=360,
    #             step=1.0,
    #             description="Rotate azimuth (dd)",
    #             disabled=False,
    #             continuous_update=False,
    #             orientation="horizontal",
    #             style={"description_width": "initial"},
    #         )
    #
    #     return self._azimuth_rotation

    @property
    def boreholes(self):
        if getattr(self, "_boreholes", None) is None:
            self._boreholes = ObjectDataSelection(object_types=Points)

        return self._boreholes

    @property
    def boreholes_size(self):
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
        if getattr(self, "_center", None) is None:
            self._center = FloatSlider(
                value=0.5,
                min=0,
                max=1.0,
                step=0.001,
                description="Window Center",
                disabled=False,
                continuous_update=False,
                orientation="horizontal",
            )

        return self._center

    @property
    def channels(self):
        if getattr(self, "_channels", None) is None:
            self._channels = SelectMultiple(description="Channels")

        return self._channels

    @property
    def channel_selection(self):
        if getattr(self, "_channel_selection", None) is None:
            self._channel_selection = Dropdown(
                description="Time Gate",
                options=self.em_system_specs[self.system.value]["channels"].keys(),
            )

        return self._channel_selection

    @property
    def color_maps(self):
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
        Data selector
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
    def dip_rotation(self):
        if getattr(self, "_dip_rotation", None) is None:
            self._dip_rotation = FloatSlider(
                value=0,
                min=0,
                max=180,
                step=1.0,
                description="Rotate dip (dd)",
                disabled=False,
                continuous_update=False,
                orientation="horizontal",
                style={"description_width": "initial"},
            )

        return self._dip_rotation

    @property
    def doi_percent(self):
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
        if getattr(self, "_doi_revert", None) is None:
            self._doi_revert = Checkbox(description="Revert", value=False)

        return self._doi_revert

    @property
    def doi_selection(self):
        if getattr(self, "_doi_selection", None) is None:
            self._doi_selection = ObjectDataSelection()

        return self._doi_selection

    @property
    def width(self):
        if getattr(self, "_width", None) is None:
            self._width = FloatSlider(
                value=1.0,
                min=0.025,
                max=1.0,
                step=0.005,
                description="Window Width",
                disabled=False,
                continuous_update=False,
                orientation="horizontal",
            )

        return self._width

    @property
    def group_add(self):
        if getattr(self, "_group_add", None) is None:
            self._group_add = ToggleButton(description="^ Add New Group ^")

        return self._group_add

    @property
    def group_color(self):
        if getattr(self, "_group_color", None) is None:
            self._group_color = ColorPicker(
                concise=False, description="Color", value="blue", disabled=False
            )

        return self._group_color

    @property
    def group_list(self):
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
    def group_name(self):
        if getattr(self, "_group_name", None) is None:
            self._group_name = Text(description="Name")

        return self._group_name

    @property
    def groups_setter(self):
        if getattr(self, "_groups_setter", None) is None:
            self._groups_setter = ToggleButton(
                description="Select Time Groups", value=False
            )

        return self._groups_setter

    @property
    def lines(self):
        """
        Line selection widget. Stores the profile for plotting.
        """
        if getattr(self, "_lines", None) is None:
            self._lines = LineOptions(multiple_lines=False)

        return self._lines

    @property
    def markers(self):
        if getattr(self, "_markers", None) is None:
            self._markers = ToggleButton(description="Show markers")

        return self._markers

    @property
    def max_migration(self):
        """
        Filter anomalies based on maximum horizontal migration of peaks.
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
            )

        return self._max_migration

    @property
    def min_amplitude(self):
        """
        Filter small anomalies based on amplitude ratio
        between peaks and lows.
        """
        if getattr(self, "_min_amplitude", None) is None:
            self._min_amplitude = IntSlider(
                value=25,
                min=0,
                max=1000,
                continuous_update=False,
                description="Minimum amplitude (%)",
                style={"description_width": "initial"},
            )

        return self._min_amplitude

    @property
    def min_channels(self):
        """
        Filter peak groups based on minimum number of data channels overlap.
        """
        if getattr(self, "_min_channels", None) is None:
            self._min_channels = IntSlider(
                value=2,
                min=1,
                max=10,
                continuous_update=False,
                description="Minimum # channels",
                style={"description_width": "initial"},
            )

        return self._min_channels

    @property
    def min_value(self):
        """
        Filter out small data values.
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
        Filter small anomalies based on width
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
        if getattr(self, "_model_log", None) is None:
            self._model_log = Checkbox(description="log", value=True, indent=False)

        return self._model_log

    @property
    def model_max(self):
        if getattr(self, "_model_max", None) is None:
            self._model_max = FloatText(
                description="max", value=1e-1, continuous_update=False
            )

        return self._model_max

    @property
    def model_min(self):
        if getattr(self, "_model_min", None) is None:
            self._model_min = FloatText(
                description="min", value=1e-4, continuous_update=False
            )

        return self._model_min

    @property
    def model_selection(self):
        if getattr(self, "_model_selection", None) is None:
            self._model_selection = ObjectDataSelection(object_types=Surface)

        return self._model_selection

    @property
    def opacity(self):
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
        if getattr(self, "_residual", None) is None:
            self._residual = Checkbox(description="Use residual", value=False)

        return self._residual

    @property
    def reverse_cmap(self):
        if getattr(self, "_reverse_cmap", None) is None:
            self._reverse_cmap = ToggleButton(description="Flip colormap", value=False)

        return self._reverse_cmap

    @property
    def run_all(self):
        if getattr(self, "_run_all", None) is None:
            self._run_all = Button(
                description="Process All Lines", button_style="warning"
            )

        return self._run_all

    @property
    def scale_button(self):
        if getattr(self, "_scale_button", None) is None:
            self._scale_button = ToggleButtons(
                options=["linear", "symlog",],
                description="Y-axis scaling",
                orientation="vertical",
            )

        return self._scale_button

    @property
    def scale_value(self):
        if getattr(self, "_scale_value", None) is None:
            self._scale_value = FloatLogSlider(
                min=-18,
                max=10,
                step=0.5,
                base=10,
                value=1e-2,
                description="Linear threshold",
                continuous_update=False,
            )

        return self._scale_value

    @property
    def shift_cox_z(self):
        if getattr(self, "_shift_cox_z", None) is None:
            self._shift_cox_z = FloatSlider(
                value=0,
                min=0,
                max=200,
                step=1.0,
                description="Z shift (m)",
                disabled=False,
                continuous_update=False,
                orientation="horizontal",
            )

        return self._shift_cox_z

    @property
    def show_borehole(self):
        if getattr(self, "_show_borehole", None) is None:
            self._show_borehole = ToggleButton(
                description="Show Boreholes", value=False,
            )

        return self._show_borehole

    @property
    def show_decay(self):
        if getattr(self, "_show_decay", None) is None:
            self._show_decay = ToggleButton(description="Show decay", value=False)

        return self._show_decay

    @property
    def show_doi(self):
        if getattr(self, "_show_doi", None) is None:
            self._show_doi = ToggleButton(description="Show DOI", value=False,)

        return self._show_doi

    @property
    def show_model(self):
        if getattr(self, "_show_model", None) is None:
            self._show_model = ToggleButton(description="Show model", value=False,)

        return self._show_model

    @property
    def slice_width(self):
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
        if getattr(self, "_smoothing", None) is None:
            self._smoothing = IntSlider(
                min=0,
                max=64,
                value=0,
                description="Smoothing",
                continuous_update=False,
                tooltip="Running mean width",
            )

        return self._smoothing

    @property
    def structural_markers(self):
        if getattr(self, "_structural_markers", None) is None:
            self._structural_markers = Checkbox(description="Structural Markers")

        return self._structural_markers

    @property
    def survey(self):
        """

        """

        return self._survey

    @property
    def system(self):
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
    def time_groups(self):

        return self._time_groups

    @time_groups.setter
    def time_groups(self, groups: dict):

        # # Append keys for profiling
        # for group in groups.values():
        #     group["channels"] = []

        self._time_groups = groups

    @property
    def threshold(self):
        if getattr(self, "_threshold", None) is None:
            self._threshold = FloatSlider(
                value=50,
                min=10,
                max=90,
                step=5,
                continuous_update=False,
                description="Decay threshold (%)",
            )

        return self._threshold

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

        # Refresh the list of objects
        self.update_objects_list()

        self.lines.objects = self.objects
        self.lines.workspace = workspace

        self.model_selection.workspace = workspace
        self.boreholes.workspace = workspace
        self.doi_selection.objects = self.model_selection.objects
        self.doi_selection.workspace = workspace

        self.reset_model_figure()

    @property
    def x_label(self):
        if getattr(self, "_x_label", None) is None:
            self._x_label = ToggleButtons(
                options=["Distance", "Easting", "Northing"],
                value="Distance",
                description="X-axis label:",
                orientation="horizontal",
            )

        return self._x_label

    def channel_panel_update(self, _):
        self.channel_panel.children = [
            self.channel_selection,
            self.data_channel_options[self.channel_selection.value],
        ]

    def set_data(self, _):
        if getattr(self, "survey", None) is not None:
            groups = [p_g.name for p_g in self.survey.property_groups]
            channels = []  # list(self.data.value)

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

            for key, widget in self.data_channel_options.items():
                widget.children[0].options = channels
                widget.children[0].value = find_value(channels, [key])

            self.plot_trigger.value = False
            self.plot_trigger.value = True

    def objects_change(self, _):
        if self.workspace.get_entity(self.objects.value):
            self._survey = self.workspace.get_entity(self.objects.value)[0]

            for aem_system, specs in self.em_system_specs.items():
                if any(
                    [
                        specs["flag"] in channel
                        for channel in self._survey.get_data_list()
                    ]
                ):
                    self.system.value = aem_system

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

    def system_box_trigger(self, _):
        if self.system_box_option.value:
            self.system_box.children = [
                self.system_box_option,
                self.system,
                self.channel_panel,
            ]
        else:
            self.system_box.children = [self.system_box_option]

    def system_observer(self, _):
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
        if self.groups_setter.value:
            self.groups_widget.children = [
                self.groups_setter,
                self.group_list,
                self.channels,
                self.group_color,
            ]
        else:
            self.groups_widget.children = [self.groups_setter]

    def set_default_groups(self, channels):
        """
        Assign TEM channel for given gate #
        """

    def edit_group(self, _):
        """
        Change channels associated with groups
        """
        gates = []
        all_gates = []  # Track unique gates
        for key in range(3):
            if self.time_groups[key]["name"] == self.group_list.value:
                group_gates = []
                for channel in list(self.channels.value):
                    if re.findall(r"\d+", channel):
                        value = int(re.findall(r"\d+", channel)[-1])
                        if value not in all_gates:
                            group_gates += [value]

                gates += [group_gates]
            else:
                gates += [self.time_groups[key]["gates"]]
            all_gates.extend(gates[-1])

        # Refresh only if list of gates has changed
        if not all_gates == [channel["gate"] for channel in self.active_channels]:
            self.reset_groups(gates=gates)
            self.plot_trigger.value = False
            self.plot_trigger.value = True

    def run_all_click(self, _):
        """
        Process the entire Curve object for all lines
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

    def trigger_click(self, _):

        # for group in self.group_list.value:
        if not self.all_anomalies:
            return

        # Append all lines
        time_group, tau, migration, azimuth, cox, amplitude, inflx_up, inflx_dwn = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for line in self.all_anomalies:
            if "time_group" in list(line.keys()) and len(line["cox"]) > 0:
                time_group += [line["time_group"]]
                tau += [line["tau"]]
                migration += [line["migration"]]
                amplitude += [line["amplitude"]]
                azimuth += [line["azimuth"]]
                cox += [line["cox"]]
                inflx_dwn += [np.vstack(line["inflx_dwn"])]
                inflx_up += [np.vstack(line["inflx_up"])]

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
            points.add_data(
                {
                    "amplitude": {"values": np.hstack(amplitude)},
                    "azimuth": {"values": np.hstack(azimuth)},
                    "dip": {"values": dip},
                    "tau": {"values": np.hstack(tau)},
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
            group = points.find_or_create_property_group(
                name="AzmDip", property_group_type="Dip direction & dip"
            )
            group.properties = [
                points.get_data("azimuth")[0].uid,
                points.get_data("dip")[0].uid,
            ]

            # Add structural markers
            if self.structural_markers.value:
                markers = []

                def rotation_2D(angle):
                    R = np.r_[
                        np.c_[
                            np.cos(np.pi * angle / 180), -np.sin(np.pi * angle / 180)
                        ],
                        np.c_[np.sin(np.pi * angle / 180), np.cos(np.pi * angle / 180)],
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
                        np.c_[np.dot(rotation_2D(-azm), marker.T).T, np.zeros(4)] + xyz
                    )
                    markers.append(marker.squeeze())

                curves = Curve.create(
                    self.workspace,
                    name="TickMarkers",
                    vertices=np.vstack(markers),
                    cells=np.arange(len(markers) * 4, dtype="uint32").reshape((-1, 2)),
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
                Points.create(
                    self.workspace,
                    name="Inflexion_Up",
                    vertices=np.vstack(inflx_up),
                    parent=self.ga_group,
                )
                Points.create(
                    self.workspace,
                    name="Inflexion_Down",
                    vertices=np.vstack(inflx_dwn),
                    parent=self.ga_group,
                )
        if self.live_link.value:
            self.live_link_output(points)

            if self.structural_markers.value:
                self.live_link_output(curves)

        self.workspace.finalize()

    def highlight_selection(self, _):
        """
        Highlight the time group data selection
        """
        for group in self.time_groups.values():
            if group["name"] == self.group_list.value:
                self.group_color.value = group["color"]
                self.channels.value = group["channels"]

    def plot_data_selection(
        self,
        data,
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
        threshold,
    ):

        self.line_update()

        if (
            getattr(self, "survey", None) is None
            or getattr(self.lines, "profile", None) is None
            or self.plot_trigger.value is False
            or len(self.active_channels) == 0
        ):
            return

        axs = None
        center_x = center * self.lines.profile.locations_resampled[-1]

        if (
            residual != self.lines.profile.residual
            or smoothing != self.lines.profile.smoothing
        ):
            self.lines.profile.residual = residual
            self.lines.profile.smoothing = smoothing
            self.line_update()

        lims = np.searchsorted(
            self.lines.profile.locations_resampled,
            [
                (center - width / 2.0) * self.lines.profile.locations_resampled[-1],
                (center + width / 2.0) * self.lines.profile.locations_resampled[-1],
            ],
        )
        sub_ind = np.arange(lims[0], lims[1])
        y_min, y_max = np.inf, -np.inf

        if getattr(self.survey, "line_indices", None) is None:
            return

        locs = self.lines.profile.locations_resampled
        for channel in self.active_channels:
            if axs is None:
                plt.figure(figsize=(12, 6))
                axs = plt.subplot()

            self.lines.profile.values = channel["values"][self.survey.line_indices]
            values = self.lines.profile.values_resampled
            y_min = np.min([values[sub_ind].min(), y_min])
            y_max = np.max([values[sub_ind].max(), y_max])

            axs.plot(locs, values, color=[0.5, 0.5, 0.5, 1])

            # Plot the anomalies by time group color
            for group in self.lines.anomalies:
                query = np.where(group["gates"] == channel["gate"])[0]

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

                axs.scatter(
                    locs[group["peak"][ii]],
                    values[group["peak"][ii]],
                    s=200,
                    c=self.time_groups[group["time_group"]]["color"],
                    marker=self.marker[ori],
                )

                if markers:
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
            center_x - width / 2.0 * self.lines.profile.locations_resampled[-1],
            center_x + width / 2.0 * self.lines.profile.locations_resampled[-1],
        ]
        axs.set_xlim(x_lims)
        axs.set_ylim([np.max([y_min, min_value]), y_max])
        # axs.set_title(f"Line: {ind}")
        axs.set_ylabel("dBdT")

        axs.scatter(center_x, (y_min + y_max) / 2, s=100, c="k", marker="d")
        if x_label == "Easting":
            axs.text(
                center_x,
                0,
                f"{self.lines.profile.interp_x(center_x):.0f} m E",
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
                center_x,
                0,
                f"{self.lines.profile.interp_y(center_x):.0f} m N",
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
                center_x,
                0,
                f"{center_x:.0f} m",
                va="top",
                ha="center",
                bbox={"edgecolor": "r"},
            )
            axs.set_xlabel("Distance (m)")

        axs.grid(True)

    def plot_decay_curve(
        self, ind, smoothing, residual, center, groups, plot_trigger, threshold
    ):
        axs = None
        if self.plot_trigger.value and hasattr(self.lines, "profile"):

            center_x = self.center.value * self.lines.profile.locations_resampled[-1]
            # Loop through groups and find nearest to cursor
            dist = np.inf
            group = None
            for anomaly in self.lines.anomalies:
                delta_x = np.abs(
                    center_x
                    - self.lines.profile.locations_resampled[anomaly["peak"][0]]
                )
                if delta_x < dist:
                    dist = delta_x
                    group = anomaly

            # Get the times of the group and plot the linear regression
            times = []
            if group is not None:
                times = [
                    channel["time"]
                    for channel in self.active_channels
                    if channel["gate"] in list(group["gates"])
                ]

            if any(times):
                times = np.hstack(times)

                if axs is None:
                    plt.figure(figsize=(8, 8))
                    axs = plt.subplot()

                y = np.exp(times * group["linear_fit"][1] + group["linear_fit"][0])
                axs.plot(
                    times, y, "--", linewidth=2, color="k",
                )
                axs.text(
                    np.mean(times),
                    np.mean(y),
                    f"Tau: {np.abs(group['linear_fit'][0] ** -1.):.2e}",
                    color="k",
                )
                #                 plt.yscale('symlog', linthreshy=scale_value)
                #                 axs.set_aspect('equal')
                axs.scatter(
                    times,
                    group["peak_values"],
                    color=self.time_groups[group["time_group"]]["color"],
                    marker="^",
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
        objects,
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
        self.update_line_model()
        self.update_line_boreholes()

        if (
            getattr(self, "survey", None) is None
            or getattr(self.lines, "profile", None) is None
            or self.show_model.value is False
            or getattr(self.lines, "model_vertices", None) is None
            or plot_trigger is False
        ):
            return

        if reverse:
            colormap += "_r"

        if (
            getattr(self.lines, "model_vertices", None) is not None
            and getattr(self.lines, "model_values", None) is not None
        ):
            tree = cKDTree(self.lines.model_vertices)

            # Create dip marker
            center_l = center * self.lines.profile.locations_resampled[-1]
            center_x = float(self.lines.profile.interp_x(center_l))
            center_y = float(self.lines.profile.interp_y(center_l))
            center_z = float(self.lines.profile.interp_z(center_l))

            _, ind = tree.query(np.c_[center_x, center_y, center_z])

            self.model_figure.data[0].x = self.lines.model_vertices[ind, 0]
            self.model_figure.data[0].y = self.lines.model_vertices[ind, 1]
            self.model_figure.data[0].z = self.lines.model_vertices[ind, 2]
            self.model_figure.data[0].mode = "markers"
            self.model_figure.data[0].marker = {
                "symbol": "diamond",
                "color": "black",
                "size": 5,
            }

            cox, azimuth, dip = [], [], []
            locs = self.lines.profile.locations_resampled
            for group in self.lines.anomalies:
                cox += [
                    [
                        self.lines.profile.interp_x(locs[group["peak"][0]]),
                        self.lines.profile.interp_y(locs[group["peak"][0]]),
                        self.lines.profile.interp_z(locs[group["peak"][0]]),
                    ]
                ]
                azimuth += [group["azimuth"]]
                dip += [group["migration"]]

            self.model_figure.data[1].x = []
            self.model_figure.data[1].y = []
            self.model_figure.data[1].z = []

            if len(cox) > 0:
                dip = np.hstack(dip)
                dip /= dip.max()
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
            # else:

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

            self.model_figure.show()

    def scale_update(self, _):
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
        self.lines.anomalies, self.lines.profile = self.client.compute(
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

        if self.show_model.value:
            self.update_line_model()

    def get_line_indices(self, line_id):
        """
        Find the vertices for a given line ID
        """
        indices = np.where(
            np.asarray(self.survey.get_data(self.lines.data.value)[0].values) == line_id
        )[0]

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
                (self.center.value - self.width.value / 2.0)
                * self.lines.profile.locations_resampled[-1],
                (self.center.value + self.width.value / 2.0)
                * self.lines.profile.locations_resampled[-1],
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

    def reset_groups(self, gates=None):

        if gates is None:
            start = self.em_system_specs[self.system.value]["channel_start_index"]
            end = len(self.em_system_specs[self.system.value]["channels"].keys()) + 1

            # Divide channels in three equal blocks
            block = int((end - start) / 3)
            early = np.arange(start, start + block).tolist()
            mid = np.arange(start + block, start + 2 * block).tolist()
            late = np.arange(start + 2 * block, end).tolist()

            gates = [early, mid, late]
        for group in self.time_groups.values():
            group["gates"] = []
            group["channels"] = []
            for ind in group["label"]:
                group["gates"] += gates[ind]

        for channel in self.channels.options:
            if re.findall(r"\d+", channel):
                value = int(re.findall(r"\d+", channel)[-1])
                for group in self.time_groups.values():
                    if value in group["gates"]:
                        group["channels"].append(channel)

        self.highlight_selection(None)
        self.active_channels = [
            {"name": c, "gate": g}
            for c, g in zip(
                (
                    self.time_groups[0]["channels"]
                    + self.time_groups[1]["channels"]
                    + self.time_groups[2]["channels"]
                ),
                (
                    self.time_groups[0]["gates"]
                    + self.time_groups[1]["gates"]
                    + self.time_groups[2]["gates"]
                ),
            )
        ]
        d_min, d_max = np.inf, -np.inf
        for channel in self.active_channels:
            channel["time"] = (
                self.data_channel_options[f"[{channel['gate']}]"].children[1].value
            )
            channel["values"] = self.data_channels[channel["name"]].values.copy()
            d_min = np.min([d_min, channel["values"].min()])
            d_max = np.max([d_max, channel["values"].max()])

        if d_max > -np.inf:
            self.plot_trigger.value = False
            self.min_value.value = d_min
        # self.set_default_groups(self.channels.options)

    def show_model_trigger(self, _):
        """
        Add the model widget
        """
        if self.show_model.value:
            self.model_panel.children = [
                self.show_model,
                HBox(
                    [
                        VBox(
                            [
                                self.model_selection.objects,
                                self.model_selection.data,
                                self.model_log,
                                self.model_min,
                                self.model_max,
                                self.color_maps,
                                self.reverse_cmap,
                                self.opacity,
                                self.doi_panel,
                                self.borehole_panel,
                                # Label("Adjust Dip Marker"),
                                # self.shift_cox_z,
                                # self.dip_rotation,
                                # self.azimuth_rotation,
                            ],
                            layout=Layout(width="50%"),
                        ),
                        self.model_figure,
                    ]
                ),
            ]
            self.show_model.description = "Hide model"
            self.plot_trigger.value = False
            self.plot_trigger.value = True
        else:
            self.model_panel.children = [self.show_model]
            self.show_model.description = "Show model"

    def show_decay_trigger(self, _):
        """
        Add the decay curve plot
        """
        if self.show_decay.value:
            self.decay_panel.children = [self.show_decay, self.decay, self.threshold]
            self.show_decay.description = "Hide decay curve"
        else:
            self.decay_panel.children = [self.show_decay]
            self.show_decay.description = "Show decay curve"

    def show_doi_trigger(self, _):
        """
        Add the DOI options
        """
        if self.show_doi.value:
            self.doi_panel.children = [
                self.show_doi,
                self.doi_selection.data,
                self.doi_percent,
                self.doi_revert,
            ]
            self.show_doi.description = "Hide DOI"
        else:
            self.doi_panel.children = [self.show_doi]
            self.show_doi.description = "Show DOI"

    def show_borehole_trigger(self, _):
        """
        Add the DOI options
        """
        if self.show_borehole.value:
            self.borehole_panel.children = [
                self.show_borehole,
                self.boreholes.widget,
                self.slice_width,
                self.boreholes_size,
            ]
            self.show_borehole.description = "Hide Boreholes"
        else:
            self.borehole_panel.children = [self.show_borehole]
            self.show_borehole.description = "Show Boreholes"

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
    min_amplitude=0.25,
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
    smoothing=self.smoothing.value, use_residual=self.residual.value, min_amplitude=self.min_amplitude.value
    min_width=self.min_width.value, max_migration=self.max_migration.value, min_channels=self.min_channels.value,
    """
    profile = signal_processing_1d(
        locations[line_indices], None, smoothing=smoothing, residual=use_residual
    )
    locs = profile.locations_resampled
    # normalization = self.em_system_specs[self.system.value]["normalization"]
    xy = np.c_[profile.interp_x(locs), profile.interp_y(locs)]
    angles = np.arctan2(xy[1:, 1] - xy[:-1, 1], xy[1:, 0] - xy[:-1, 0])
    angles = np.r_[angles[0], angles].tolist()
    azimuth = (450.0 - np.rad2deg(running_mean(angles, width=5))) % 360.0
    anomalies = {
        "gates": [],
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
    for channel in channels:
        # if channel not in list(self.channel_to_gate.keys()):
        #     continue

        profile.values = channel["values"][line_indices].copy()
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

        if len(peaks) == 0 or len(lows) < 2:
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
            #         inflx_up = np.max([0, inflx_up])
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
                / (np.min([values[start], values[end]]) + 2e-32)
            ) * 100.0
            delta_x = locs[end] - locs[start]

            amplitude = np.sum(np.abs(values[start:end])) * profile.hx
            if (delta_amp > min_amplitude) & (delta_x > min_width):
                anomalies["gates"] += [channel["gate"]]
                anomalies["start"] += [start]
                anomalies["inflx_up"] += [inflx_up]
                anomalies["peak"] += [peak]
                anomalies["peak_values"] += [values[peak]]
                anomalies["inflx_dwn"] += [inflx_dwn]
                anomalies["amplitude"] += [amplitude]
                anomalies["end"] += [end]
                anomalies["group"] += [-1]
                anomalies["time_group"] += [
                    ind
                    for ind, time_group in enumerate(
                        [
                            time_groups[0]["channels"],
                            time_groups[1]["channels"],
                            time_groups[2]["channels"],
                        ]
                    )
                    if channel["name"] in time_group
                ]

    if len(anomalies["peak"]) == 0:
        if return_profile:
            return {}, profile
        else:
            return {}

    if minimal_output:
        groups = {
            "amplitude": [],
            "azimuth": [],
            "cox": [],
            "migration": [],
            "time_group": [],
            "tau": [],
            "inflx_dwn": [],
            "inflx_up": [],
        }
    else:
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
        u_gates, u_count = np.unique(anomalies["gates"][near], return_counts=True)
        if len(u_gates) > 1 and np.any((u_gates[1:] - u_gates[:-1]) > 2):
            cutoff = u_gates[np.where((u_gates[1:] - u_gates[:-1]) > 2)[0][0]]
            near = near[anomalies["gates"][near] > cutoff, ...]

        # Check for multiple nearest peaks on single channel
        # and keep the nearest
        u_gates, u_count = np.unique(anomalies["gates"][near], return_counts=True)
        for gate in u_gates[np.where(u_count > 1)]:
            mask = np.ones_like(near, dtype="bool")
            sub_ind = anomalies["gates"][near] == gate
            sub_ind[np.where(sub_ind)[0][np.argmin(dist[near][sub_ind])]] = False
            mask[sub_ind] = False
            near = near[mask, ...]

        anomalies["group"][near] = group_id

        # Keep largest overlapping time group
        in_gate, count = np.unique(anomalies["time_group"][near], return_counts=True)
        in_gate = in_gate[(count > min_channels) & (in_gate != -1)].tolist()
        time_group = [
            ii
            for ii, time_group in enumerate(time_groups.values())
            if in_gate == time_group["label"]
        ]
        if len(in_gate) > 0 and len(time_group) > 0:

            time_group = np.max(time_group)
            gates = anomalies["gates"][near]
            cox = anomalies["peak"][near]
            cox_sort = np.argsort(locs[cox])
            azm = azimuth[cox[0]]
            if cox_sort[-1] < cox_sort[0]:
                azm = (azm + 180) % 360.0

            migration = np.abs(locs[cox[cox_sort[-1]]] - locs[cox[cox_sort[0]]])

            # Compute tau
            times = np.hstack(
                [
                    channel["time"]
                    for channel in channels
                    if channel["gate"] in list(gates)
                ]
            )
            values = anomalies["peak_values"][near] * np.prod(data_normalization)
            amplitude = np.sum(anomalies["amplitude"][near])
            if times.shape[0] < 2 or len(cox) == 0:
                continue

            # Compute linear trend
            A = np.c_[np.ones_like(times), times]
            y0, slope = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, np.log(values)))
            linear_fit = [y0, slope]

            if minimal_output:
                groups["cox"] += [
                    np.mean(
                        np.c_[
                            profile.interp_x(locs[cox]),
                            profile.interp_y(locs[cox]),
                            profile.interp_z(locs[cox]),
                        ],
                        axis=0,
                    )
                ]
                groups["azimuth"] += [azm]
                groups["migration"] += [migration]
                groups["amplitude"] += [amplitude]
                groups["time_group"] += [time_group]
                inflx_dwn = anomalies["inflx_dwn"][near]
                groups["inflx_dwn"] += [
                    np.c_[
                        profile.interp_x(locs[inflx_dwn]),
                        profile.interp_y(locs[inflx_dwn]),
                        profile.interp_z(locs[inflx_dwn]),
                    ]
                ]
                inflx_up = anomalies["inflx_up"][near]
                groups["inflx_up"] += [
                    np.c_[
                        profile.interp_x(locs[inflx_up]),
                        profile.interp_y(locs[inflx_up]),
                        profile.interp_z(locs[inflx_up]),
                    ]
                ]
                groups["tau"] += [np.abs(linear_fit[0] ** -1.0)]
            else:
                groups += [
                    {
                        "gates": gates,
                        "start": anomalies["start"][near],
                        "inflx_up": anomalies["inflx_up"][near],
                        "peak": cox,
                        "peak_values": values,
                        "inflx_dwn": anomalies["inflx_dwn"][near],
                        "end": anomalies["end"][near],
                        "azimuth": azm,
                        "migration": migration,
                        "amplitude": amplitude,
                        "time_group": time_group,
                        "linear_fit": linear_fit,
                    }
                ]

    # if minimal_output and len(groups["cox"])>0:
    #     groups["cox"] = np.vstack(groups["cox"]).reshape((-1, 3))
    if return_profile:
        return groups, profile
    else:
        return groups
