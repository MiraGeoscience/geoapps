import os
import re
import numpy as np
from scipy.spatial import cKDTree
import plotly.graph_objects as go
import plotly.express as px
import time
import matplotlib.pyplot as plt
from geoh5py.workspace import Workspace
from geoh5py.objects import Points, Curve, Surface
from geoh5py.groups import ContainerGroup
from ipywidgets import (
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
)
from geoapps.selection import ObjectDataSelection, LineOptions


class EMLineProfiler(ObjectDataSelection):
    """
    Application for the picking of targets along Time-domain EM profiles
    """

    defaults = {
        "object_types": (Curve,),
        "h5file": "../../assets/FlinFlon.geoh5",
        "add_groups": True,
        "time_groups": {
            "early": {"range": "9-16", "color": "blue"},
            "early + middle": {"range": "9-27", "color": "green"},
            "middle": {"range": "17-27", "color": "green"},
            "middle + late": {"range": "17-40", "color": "yellow"},
            "early + middle + late": {"range": "9-40", "color": "orange",},
            "late": {"range": "28-40", "color": "red"},
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
        "show_model": True,
        "show_borehole": True,
        "show_doi": True,
        "slice_width": 150,
        "x_label": "Easting",
        "ga_group_name": "EMProfiler",
    }

    def __init__(self, **kwargs):
        kwargs = self.apply_defaults(**kwargs)

        self.borehole_trace = None
        self.data_channels = {}
        self.data_channel_options = {}
        self.em_system_specs = geophysical_systems.parameters()
        self.marker = {"left": "<", "right": ">"}
        self.pause_plot_refresh = False
        self.surface_model = None
        self._survey = None
        self._time_groups = {}

        def objects_change(_):
            self.objects_change()

        self.objects.observe(objects_change, names="value")

        def show_model_trigger(_):
            self.show_model_trigger()

        self.model_panel = VBox([self.show_model])
        self.show_model.observe(show_model_trigger, names="value")

        def show_doi_trigger(_):
            self.show_doi_trigger()

        self.doi_panel = VBox([self.show_doi])
        self.show_doi.observe(show_doi_trigger, names="value")

        def show_borehole_trigger(_):
            self.show_borehole_trigger()

        self.borehole_panel = VBox([self.show_borehole])
        self.show_borehole.observe(show_borehole_trigger, names="value")

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

        self.doi_selection.update_data_list()
        if "doi" in kwargs.keys():
            self.doi_selection.__populate__(**kwargs["doi"])
        self.doi_selection.data.description = "DOI Layer"

        scale_panel = VBox([self.scale_button])

        def scale_update(_):
            if self.scale_button.value == "symlog":
                scale_panel.children = [
                    self.scale_button,
                    self.scale_value,
                ]
            else:
                scale_panel.children = [self.scale_button]

        self.scale_button.observe(scale_update)

        def system_observer(_):
            self.system_observer()

        self.system.observe(system_observer)
        self.system_observer()

        def channel_panel_update(_):
            self.channel_panel.children = [
                self.channel_selection,
                self.data_channel_options[self.channel_selection.value],
            ]

        self.channel_selection.observe(channel_panel_update, names="value")
        self.channel_panel = VBox(
            [
                self.channel_selection,
                self.data_channel_options[self.channel_selection.value],
            ]
        )

        def system_box_trigger(_):
            self.system_box_trigger()

        self.system_box_option = ToggleButton(
            description="Select TEM System", value=False
        )
        self.system_box_option.observe(system_box_trigger)
        self.system_box = VBox([self.system_box_option])

        def add_group(_):
            self.add_group()

        self.group_add.observe(add_group)

        def highlight_selection(_):
            self.highlight_selection()

        self.group_list.observe(highlight_selection, names="value")
        self.highlight_selection()

        def groups_trigger(_):
            self.groups_trigger()

        self.groups_setter.observe(groups_trigger)
        self.groups_widget = VBox([self.groups_setter])

        def trigger_click(_):
            self.trigger_click()

        self.trigger.on_click(trigger_click)
        self.trigger.description = "Export Marker"

        def plot_data_selection(
            data,
            ind,
            smoothing,
            residual,
            markers,
            scale,
            scale_value,
            center,
            width,
            groups,
            pick_trigger,
            x_label,
            threshold,
        ):
            self.plot_data_selection(
                data,
                ind,
                smoothing,
                residual,
                markers,
                scale,
                scale_value,
                center,
                width,
                groups,
                pick_trigger,
                x_label,
                threshold,
            )

        plotting = interactive_output(
            plot_data_selection,
            {
                "data": self.data,
                "ind": self.lines.lines,
                "smoothing": self.smoothing,
                "residual": self.residual,
                "markers": self.markers,
                "scale": self.scale_button,
                "scale_value": self.scale_value,
                "center": self.center,
                "width": self.width,
                "groups": self.group_list,
                "pick_trigger": self.auto_picker,
                "x_label": self.x_label,
                "threshold": self.threshold,
            },
        )

        def plot_decay_curve(
            ind, smoothing, residual, center, groups, pick_trigger, threshold,
        ):
            self.plot_decay_curve(
                ind, smoothing, residual, center, groups, pick_trigger, threshold,
            )

        self.decay = interactive_output(
            plot_decay_curve,
            {
                "ind": self.lines.lines,
                "smoothing": self.smoothing,
                "residual": self.residual,
                "center": self.center,
                "groups": self.group_list,
                "pick_trigger": self.auto_picker,
                "threshold": self.threshold,
            },
        )
        self.decay_panel = VBox([self.show_decay])

        def show_decay_trigger(_):
            self.show_decay_trigger()

        self.show_decay.observe(show_decay_trigger, names="value")

        def plot_model_selection(
            ind,
            center,
            width,
            x_label,
            colormap,
            reverse,
            z_shift,
            dip_rotation,
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
        ):
            self.update_line_model()
            self.update_line_boreholes()
            self.plot_model_selection(
                ind,
                center,
                width,
                x_label,
                colormap,
                reverse,
                z_shift,
                dip_rotation,
                opacity,
                doi_percent,
                boreholes_size,
            )

        self.model_section = interactive_output(
            plot_model_selection,
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
                "z_shift": self.shift_cox_z,
                "dip_rotation": self.dip_rotation,
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
                HBox(
                    [
                        VBox(
                            [
                                self.center,
                                self.width,
                                self.smoothing,
                                self.residual,
                                scale_panel,
                                self.markers,
                            ],
                            layout=Layout(width="50%"),
                        ),
                        VBox([plotting, self.x_label,]),
                    ]
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
                self.model_panel,
                self.trigger_widget,
            ]
        )

        self.auto_picker.value = True

    @property
    def auto_picker(self):
        if getattr(self, "_auto_picker", None) is None:
            self._auto_picker = ToggleButton(
                description="Pick nearest target", value=False
            )

        return self._auto_picker

    @property
    def boreholes(self):
        if getattr(self, "_boreholes", None) is None:
            self._boreholes = ObjectDataSelection(object_types=(Points,))

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
                description="Position (%)",
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
        def set_data(_):
            self.set_data()

        assert isinstance(
            value, (Dropdown, SelectMultiple)
        ), f"'Objects' must be of type {Dropdown} or {SelectMultiple}"
        self._data = value
        self._data.observe(set_data, names="value")
        self.set_data()

    @property
    def dip_rotation(self):
        if getattr(self, "_dip_rotation", None) is None:
            self._dip_rotation = FloatSlider(
                value=0,
                min=0,
                max=90,
                step=1.0,
                description="Rotate dip (dd)",
                disabled=False,
                continuous_update=False,
                orientation="horizontal",
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
                description="Width (%)",
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
            self._group_list = Dropdown(description="")

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
        if getattr(self, "_lines", None) is None:
            self._lines = LineOptions(multiple_lines=False)

        return self._lines

    @property
    def markers(self):
        if getattr(self, "_markers", None) is None:
            self._markers = ToggleButton(description="Show markers")

        return self._markers

    @property
    def model_selection(self):
        if getattr(self, "_model_selection", None) is None:
            self._model_selection = ObjectDataSelection(object_types=(Surface,))

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
                description="Show Boreholes", value=False, button_style="success"
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
            self._show_doi = ToggleButton(
                description="Show DOI", value=False, button_style="success"
            )

        return self._show_doi

    @property
    def show_model(self):
        if getattr(self, "_show_model", None) is None:
            self._show_model = ToggleButton(
                description="Show model", value=False, button_style="success"
            )

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
    def survey(self):
        """

        """

        return self._survey

    @property
    def system(self):
        if getattr(self, "__system", None) is None:
            self._system = Dropdown(
                options=[
                    key
                    for key, specs in self.em_system_specs.items()
                    if specs["type"] == "time"
                ],
                description="Time-Domain System:",
            )

        return self._system

    @property
    def time_groups(self):

        return self._time_groups

    @time_groups.setter
    def time_groups(self, groups: dict):

        # Append keys for profiling
        for group in groups.values():
            group["channels"] = []
            group["gates"] = []
            group["inflx_up"] = []
            group["inflx_dwn"] = []
            group["peaks"] = []
            group["times"] = []
            group["values"] = []
            group["locations"] = []

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

    def set_data(self):
        if getattr(self, "survey", None) is not None:
            groups = [p_g.name for p_g in self.survey.property_groups]
            channels = []  # list(self.data.value)

            for channel in self.data.value:
                if channel in groups:
                    for prop in self.survey.get_property_group(channel).properties:
                        name = self.workspace.get_entity(prop)[0].name
                        if prop not in channels:
                            channels.append(name)
                else:
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

            self.auto_picker.value = False
            self.auto_picker.value = True

    def objects_change(self):
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

    def system_box_trigger(self):
        if self.system_box_option.value:
            self.system_box.children = [
                self.system_box_option,
                self.system,
                self.channel_panel,
            ]
        else:
            self.system_box.children = [self.system_box_option]

    def system_observer(self):
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

    def groups_trigger(self):
        if self.groups_setter.value:
            self.groups_widget.children = [
                self.groups_setter,
                self.group_list,
                self.channels,
                self.group_name,
                self.group_color,
                self.group_add,
            ]
        else:
            self.groups_widget.children = [self.groups_setter]

    def set_default_groups(self, channels):
        """
        Assign TEM channel for given gate #
        """

    def add_group(self):
        """
        Add a group to the list of groups
        """
        if self.group_add.value:

            if self.group_name.value not in self.group_list.options:
                self.group_list.options = list(self.group_list.options) + [
                    self.group_name.value
                ]

            self.time_groups[self.group_name.value] = {
                "color": self.group_color.value,
                "channels": list(self.channels.value),
                "inflx_up": [],
                "inflx_dwn": [],
                "peaks": [],
                "mad_tau": [],
                "times": [],
                "values": [],
                "locations": [],
                "cox": [],
                "azimuth": [],
                "dip": [],
            }
            self.group_add.value = False

    def trigger_click(self):

        # for group in self.group_list.value:
        group = self.group_list.value
        tau = self.time_groups[group]["mad_tau"]
        dip = self.time_groups[group]["dip"]
        azimuth = self.time_groups[group]["azimuth"]
        cox = self.time_groups[group]["cox"]
        cox[2] -= self.shift_cox_z.value

        points = [child for child in self.ga_group.children if child.name == group]
        if any(points):
            points = points[0]
            azm_data = points.get_data("azimuth")[0]
            azm_vals = azm_data.values.copy()
            dip_data = points.get_data("dip")[0]
            dip_vals = dip_data.values.copy()

            tau_data = points.get_data("tau")[0]
            tau_vals = tau_data.values.copy()

            points.vertices = np.vstack([points.vertices, cox.reshape((1, 3))])
            azm_data.values = np.hstack([azm_vals, azimuth])
            dip_data.values = np.hstack([dip_vals, dip])
            tau_data.values = np.hstack([tau_vals, tau])

        else:
            # if self.workspace.get_entity(group)
            # parent =
            points = Points.create(
                self.workspace,
                name=group,
                vertices=cox.reshape((1, 3)),
                parent=self.ga_group,
            )
            points.entity_type.name = group
            points.add_data(
                {
                    "azimuth": {"values": np.asarray(azimuth)},
                    "dip": {"values": np.asarray(dip)},
                    "tau": {"values": np.asarray(tau)},
                }
            )
            group = points.find_or_create_property_group(
                name="AzmDip", property_group_type="Dip direction & dip"
            )
            group.properties = [
                points.get_data("azimuth")[0].uid,
                points.get_data("dip")[0].uid,
            ]

        if self.live_link.value:
            self.live_link_output(points)

        self.workspace.finalize()

    def highlight_selection(self):
        """
        Highlight the group choice
        """
        # highlights = []
        # for group in self.group_list.value:
        #     highlights +=

        if self.group_list.value in list(self.time_groups.keys()):
            self.group_color.value = self.time_groups[self.group_list.value]["color"]
            self.channels.value = self.time_groups[self.group_list.value]["channels"]

    def plot_data_selection(
        self,
        data,
        ind,
        smoothing,
        residual,
        markers,
        scale,
        scale_value,
        center,
        width,
        groups,
        pick_trigger,
        x_label,
        threshold,
    ):

        self.line_update()

        if (
            getattr(self, "survey", None) is None
            or getattr(self.lines, "profile", None) is None
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

        try:
            time_group = self.time_groups[self.group_list.value]
            channels = time_group["channels"]
        except KeyError:
            return

        if len(channels) == 0:
            channels = self.channels.options

        times = {}
        for channel in self.data_channel_options.values():
            times[channel.children[0].value] = channel.children[1].value

        y_min, y_max = np.inf, -np.inf
        time_group["inflx_up"] = []
        time_group["inflx_dwn"] = []
        time_group["peaks"] = []
        time_group["mad_tau"] = []
        time_group["times"] = []
        time_group["values"] = []
        time_group["locations"] = []

        if getattr(self.survey, "line_indices", None) is None:
            return

        for channel, d in self.data_channels.items():

            if channel not in times.keys():
                continue

            if channel not in channels:
                continue

            if axs is None:
                fig = plt.figure(figsize=(12, 6))
                axs = plt.subplot()

            self.lines.profile.values = d.values[self.survey.line_indices].copy()
            locs, values = (
                self.lines.profile.locations_resampled[sub_ind],
                self.lines.profile.values_resampled[sub_ind],
            )

            y_min = np.min([values.min(), y_min])
            y_max = np.max([values.max(), y_max])

            axs.plot(locs, values, color=[0.5, 0.5, 0.5, 1])

            if not residual:
                raw = self.lines.profile._values_resampled_raw[sub_ind]
                axs.fill_between(
                    locs, values, raw, where=raw > values, color=[1, 0, 0, 0.5]
                )
                axs.fill_between(
                    locs, values, raw, where=raw < values, color=[0, 0, 1, 0.5]
                )

            dx = self.lines.profile.derivative(order=1)[sub_ind]
            ddx = self.lines.profile.derivative(order=2)[sub_ind]

            peaks = np.where((np.diff(np.sign(dx)) != 0) * (ddx[1:] < 0))[0]
            lows = np.where((np.diff(np.sign(dx)) != 0) * (ddx[1:] > 0))[0]

            up_inflx = np.where((np.diff(np.sign(ddx)) != 0) * (dx[1:] > 0))[0]
            dwn_inflx = np.where((np.diff(np.sign(ddx)) != 0) * (dx[1:] < 0))[0]

            if markers:
                axs.scatter(locs[peaks], values[peaks], s=100, color="r", marker="v")
                axs.scatter(locs[lows], values[lows], s=100, color="b", marker="^")
                axs.scatter(locs[up_inflx], values[up_inflx], color="g")
                axs.scatter(locs[dwn_inflx], values[dwn_inflx], color="m")

            if len(peaks) == 0 or len(lows) < 2:
                continue

            if self.auto_picker.value:

                # Check dipping direction
                x_ind = np.min([values.shape[0] - 2, np.searchsorted(locs, center_x)])
                if values[x_ind] > values[x_ind + 1]:
                    cox = peaks[np.searchsorted(locs[peaks], center_x) - 1]
                else:
                    x_ind = np.min(
                        [peaks.shape[0] - 1, np.searchsorted(locs[peaks], center_x)]
                    )
                    cox = peaks[x_ind]

                start = lows[np.searchsorted(locs[lows], locs[cox]) - 1]
                end = lows[
                    np.min([np.searchsorted(locs[lows], locs[cox]), len(lows) - 1])
                ]

                bump_x = locs[start:end]
                bump_v = values[start:end]

                if len(bump_x) == 0:
                    continue

                inflx_up = np.searchsorted(
                    bump_x,
                    locs[up_inflx[np.searchsorted(locs[up_inflx], locs[cox]) - 1]],
                )
                inflx_up = np.max([0, inflx_up])

                inflx_dwn = np.searchsorted(
                    bump_x,
                    locs[dwn_inflx[np.searchsorted(locs[dwn_inflx], locs[cox])]],
                )
                inflx_dwn = np.min([bump_x.shape[0] - 1, inflx_dwn])

                peak = np.min([bump_x.shape[0] - 1, np.searchsorted(bump_x, locs[cox])])

                if channel in time_group["channels"]:

                    time_group["inflx_up"].append(
                        np.r_[bump_x[inflx_up], bump_v[inflx_up]]
                    )
                    time_group["peaks"].append(np.r_[bump_x[peak], bump_v[peak]])
                    time_group["times"].append(times[channel])
                    time_group["inflx_dwn"].append(
                        np.r_[bump_x[inflx_dwn], bump_v[inflx_dwn]]
                    )
                    time_group["locations"].append(bump_x)
                    time_group["values"].append(bump_v)

                    # Compute average dip
                    left_ratio = (bump_v[peak] - bump_v[inflx_up]) / (
                        bump_x[peak] - bump_x[inflx_up]
                    )
                    right_ratio = (bump_v[peak] - bump_v[inflx_dwn]) / (
                        bump_x[inflx_dwn] - bump_x[peak]
                    )

                    if left_ratio > right_ratio:
                        ratio = right_ratio / left_ratio
                        ori = "left"
                    else:
                        ratio = left_ratio / right_ratio
                        ori = "right"

                    dip = np.rad2deg(np.arcsin(ratio))

                    # Left
                    axs.plot(
                        bump_x[:peak], bump_v[:peak], "--", color=time_group["color"],
                    )
                    # Right
                    axs.plot(
                        bump_x[peak:], bump_v[peak:], color=time_group["color"],
                    )
                    axs.scatter(
                        time_group["peaks"][-1][0],
                        time_group["peaks"][-1][1],
                        s=100,
                        c=time_group["color"],
                        marker=self.marker[ori],
                    )
                    if ~np.isnan(dip):
                        axs.text(
                            time_group["peaks"][-1][0],
                            time_group["peaks"][-1][1],
                            f"{dip:.0f}",
                            va="bottom",
                            ha="center",
                        )
                    axs.scatter(
                        time_group["inflx_dwn"][-1][0],
                        time_group["inflx_dwn"][-1][1],
                        s=100,
                        c=time_group["color"],
                        marker="1",
                    )
                    axs.scatter(
                        time_group["inflx_up"][-1][0],
                        time_group["inflx_up"][-1][1],
                        s=100,
                        c=time_group["color"],
                        marker="2",
                    )

        if np.any(time_group["peaks"]):
            peaks = np.vstack(time_group["peaks"])
            inflx_dwn = np.vstack(time_group["inflx_dwn"])
            inflx_up = np.vstack(time_group["inflx_up"])
            ratio = peaks[:, 1] / peaks[0, 1]
            ind = np.where(ratio >= (1 - threshold / 100))[0][-1]
            peaks = np.mean(peaks[: ind + 1, :], axis=0)
            inflx_dwn = np.mean(inflx_dwn[: ind + 1, :], axis=0)
            inflx_up = np.mean(inflx_up[: ind + 1, :], axis=0)
            cox_x = self.lines.profile.interp_x(peaks[0])
            cox_y = self.lines.profile.interp_y(peaks[0])
            cox_z = self.lines.profile.interp_z(peaks[0])
            time_group["cox"] = np.r_[cox_x, cox_y, cox_z]

            # Compute average dip
            left_ratio = np.abs((peaks[1] - inflx_up[1]) / (peaks[0] - inflx_up[0]))
            right_ratio = np.abs((peaks[1] - inflx_dwn[1]) / (peaks[0] - inflx_dwn[0]))

            if left_ratio > right_ratio:
                ratio = right_ratio / left_ratio
                azm = (
                    450.0
                    - np.rad2deg(
                        np.arctan2(
                            (self.lines.profile.interp_y(inflx_up[0]) - cox_y),
                            (self.lines.profile.interp_x(inflx_up[0]) - cox_x),
                        )
                    )
                ) % 360.0
            else:
                ratio = left_ratio / right_ratio
                azm = (
                    450.0
                    - np.rad2deg(
                        np.arctan2(
                            (self.lines.profile.interp_y(inflx_dwn[0]) - cox_y),
                            (self.lines.profile.interp_x(inflx_dwn[0]) - cox_x),
                        )
                    )
                ) % 360.0

            dip = np.rad2deg(np.arcsin(ratio))

            time_group["azimuth"] = azm
            time_group["dip"] = dip
            self.pause_plot_refresh = True
            self.dip_rotation.value = dip
            self.pause_plot_refresh = False

        if axs is not None:
            axs.plot(
                [
                    self.lines.profile.locations_resampled[0],
                    self.lines.profile.locations_resampled[-1],
                ],
                [0, 0],
                "r",
            )
            axs.plot([center_x, center_x], [0, y_max], "g--")

            # for group in self.time_groups.values():
            if time_group["peaks"]:
                peaks = np.vstack(time_group["peaks"])
                ratio = peaks[:, 1] / peaks[0, 1]
                ind = np.where(ratio >= (1 - threshold / 100))[0][-1]
                #                 print(ind)
                axs.plot(
                    peaks[: ind + 1, 0],
                    peaks[: ind + 1, 1],
                    "--",
                    color=time_group["color"],
                )

            if scale == "symlog":
                plt.yscale("symlog", linthreshy=scale_value)

            x_lims = [
                center_x - width / 2.0 * self.lines.profile.locations_resampled[-1],
                center_x + width / 2.0 * self.lines.profile.locations_resampled[-1],
            ]
            axs.set_xlim(x_lims)
            axs.set_title(f"Line: {ind}")
            axs.set_ylabel("dBdT")

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
            pos = axs.get_position()

    def plot_decay_curve(
        self, ind, smoothing, residual, center, groups, pick_trigger, threshold
    ):
        axs = None
        if self.auto_picker.value:

            group = self.group_list.value
            # for group in self.group_list.value:

            if len(self.time_groups[group]["peaks"]) == 0:
                return

            peaks = (
                np.vstack(self.time_groups[group]["peaks"])
                * self.em_system_specs[self.system.value]["normalization"]
            )

            ratio = peaks[:, 1] / peaks[0, 1]
            ind = np.where(ratio >= (1 - self.threshold.value / 100))[0][-1]
            tc = np.hstack(self.time_groups[group]["times"][: ind + 1])
            vals = np.log(peaks[: ind + 1, 1])

            if tc.shape[0] < 2:
                return

            # Compute linear trend
            A = np.c_[np.ones_like(tc), tc]
            a, c = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, vals))
            d = np.r_[tc.min(), tc.max()]
            vv = d * c + a
            ratio = np.abs((vv[0] - vv[1]) / (d[0] - d[1]))

            self.time_groups[group]["mad_tau"] = ratio ** -1.0

            if axs is None:
                plt.figure(figsize=(8, 8))
                axs = plt.subplot()

            axs.plot(
                d,
                np.exp(d * c + a),
                "--",
                linewidth=2,
                color=self.time_groups[group]["color"],
            )
            axs.text(
                np.mean(d),
                np.exp(np.mean(vv)),
                f"{ratio ** -1.:.2e}",
                color=self.time_groups[group]["color"],
            )
            #                 plt.yscale('symlog', linthreshy=scale_value)
            #                 axs.set_aspect('equal')
            axs.scatter(
                np.hstack(self.time_groups[group]["times"]),
                peaks[:, 1],
                color=self.time_groups[group]["color"],
                marker="^",
            )
            axs.grid(True)

            plt.yscale("symlog")
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
        reverse,
        z_shift,
        dip_rotation,
        opacity,
        doi_percent,
        boreholes_size,
    ):
        if (
            getattr(self, "survey", None) is None
            or getattr(self.lines, "profile", None) is None
            or self.show_model.value is False
            or getattr(self.lines, "model_vertices", None) is None
            or self.pause_plot_refresh
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

            # for group in self.group_list.value:
            group = self.group_list.value
            if not np.any(self.time_groups[group]["peaks"]):
                return

            _, ind = tree.query(self.time_groups[group]["cox"].reshape((-1, 3)))
            dip = dip_rotation
            azimuth = self.time_groups[group]["azimuth"]

            if dip > 90:
                dip = 180 - dip
                azimuth += 180
            self.time_groups[group]["dip"] = dip
            self.time_groups[group]["azimuth"] = azimuth

            vec = rotate_azimuth_dip(azimuth, dip,)
            scaler = 100

            self.model_figure.data[1].x = self.lines.model_vertices[ind, 0]
            self.model_figure.data[1].y = self.lines.model_vertices[ind, 1]
            self.model_figure.data[1].z = self.lines.model_vertices[ind, 2] - z_shift
            self.model_figure.data[1].u = vec[:, 0] * scaler
            self.model_figure.data[1].v = vec[:, 1] * scaler
            self.model_figure.data[1].w = vec[:, 2] * scaler
            self.model_figure.data[1].showscale = False

            self.time_groups[group]["cox"][2] = self.lines.model_vertices[ind, 2]

            simplices = self.lines.model_cells.reshape((-1, 3))

            model_values = np.log10(self.lines.model_values)

            if self.show_doi.value:
                model_values[self.lines.doi_values > doi_percent] = np.nan
            # print(model_values.min())
            self.lines.model_values * self.lines.doi_values
            self.model_figure.data[2].x = self.lines.model_vertices[:, 0]
            self.model_figure.data[2].y = self.lines.model_vertices[:, 1]
            self.model_figure.data[2].z = self.lines.model_vertices[:, 2]
            self.model_figure.data[2].intensity = model_values
            self.model_figure.data[2].opacity = opacity
            self.model_figure.data[2].i = simplices[:, 0]
            self.model_figure.data[2].j = simplices[:, 1]
            self.model_figure.data[2].k = simplices[:, 2]
            self.model_figure.data[2].colorscale = colormap

            if (
                getattr(self.lines, "borehole_vertices", None) is not None
                and self.show_borehole.value
            ):
                self.model_figure.data[3].x = self.lines.borehole_vertices[:, 0]
                self.model_figure.data[3].y = self.lines.borehole_vertices[:, 1]
                self.model_figure.data[3].z = self.lines.borehole_vertices[:, 2]
                self.model_figure.data[3].mode = "markers"
                self.model_figure.data[3].marker.size = boreholes_size

                if getattr(self.lines, "borehole_values", None) is not None:
                    self.model_figure.data[3].marker.color = self.lines.borehole_values

            self.model_figure.show()

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

        line_ind = np.where(
            np.asarray(self.survey.get_data(self.lines.data.value)[0].values)
            == self.lines.lines.value
        )[0]

        if len(line_ind) == 0:
            return

        self.survey.line_indices = line_ind
        xyz = self.survey.vertices[line_ind, :]

        self.lines.profile = signal_processing_1d(
            xyz, None, smoothing=self.smoothing.value, residual=self.residual.value
        )

        if self.show_model.value:
            self.update_line_model()

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

    def reset_groups(self):

        for group in self.time_groups.values():
            try:
                first, last = np.asarray(group["range"].split("-"), dtype="int")
                group["gates"] = np.arange(first, last + 1).tolist()
                group["channels"] = []
                group["inflx_up"] = []
                group["inflx_dwn"] = []
                group["peaks"] = []
                group["mad_tau"] = []
                group["times"] = []
                group["values"] = []
                group["locations"] = []
                group["cox"] = []
                group["azimuth"] = []
                group["dip"] = []

            except ValueError:
                pass

        for channel in self.channels.options:
            if re.findall(r"\d+", channel):
                value = int(re.findall(r"\d+", channel)[-1])
                for group in self.time_groups.values():
                    if value in group["gates"]:
                        group["channels"].append(channel)
        self.group_list.options = self.time_groups.keys()
        self.group_list.value = self.group_list.options[0]
        self.highlight_selection()

        # self.set_default_groups(self.channels.options)

    def show_model_trigger(self):
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
                                self.color_maps,
                                self.reverse_cmap,
                                self.opacity,
                                self.doi_panel,
                                self.borehole_panel,
                                Label("Adjust Dip Marker"),
                                self.shift_cox_z,
                                self.dip_rotation,
                            ],
                            layout=Layout(width="50%"),
                        ),
                        self.model_figure,
                    ]
                ),
            ]
            self.show_model.description = "Hide model"
        else:
            self.model_panel.children = [self.show_model]
            self.show_model.description = "Show model"

    def show_decay_trigger(self):
        """
        Add the decay curve plot
        """
        if self.show_decay.value:
            self.decay_panel.children = [self.show_decay, self.decay, self.threshold]
            self.show_decay.description = "Hide decay curve"
        else:
            self.decay_panel.children = [self.show_decay]
            self.show_decay.description = "Show decay curve"

    def show_doi_trigger(self):
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

    def show_borehole_trigger(self):
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
