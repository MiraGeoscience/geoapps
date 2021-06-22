#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import re
import sys
from os import path

import dask
import matplotlib.pyplot as plt
import numpy as np
from dask.distributed import Client, get_client
from geoh5py.data import ReferencedData
from geoh5py.objects import Curve, Points
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

from geoapps.selection import LineOptions, ObjectDataSelection
from geoapps.utils import geophysical_systems
from geoapps.utils.utils import (
    LineDataDerivatives,
    colors,
    find_value,
    hex_to_rgb,
    running_mean,
)

from ..io.PeakFinder import PeakFinderParams


class PeakFinder(ObjectDataSelection):
    """
    Application for the picking of targets along Time-domain EM profiles
    """

    _param_class = PeakFinderParams
    _add_groups = "only"
    _object_types = (Curve,)
    _group_auto = None
    decay_figure = None
    marker = {"left": "<", "right": ">"}

    def __init__(self, ui_json=None, **kwargs):

        if ui_json is not None and path.exists(ui_json):
            self.params = self._param_class.from_path(ui_json)
        else:

            default_dict = self._param_class._default_ui_json
            for key, arg in kwargs.items():
                if key == "h5file":
                    key = "geoh5"
                try:
                    default_dict[key] = arg
                except KeyError:
                    continue

            self.params = self._param_class.from_dict(default_dict)

        self.defaults = self.update_defaults(**self.params.__dict__)
        self.all_anomalies = []
        self.data_channels = {}
        self.data_channel_options = {}
        self.pause_plot_refresh = False
        self._survey = None
        self._time_groups = None

        super().__init__(**kwargs)

        self.system_panel = VBox([self.system_panel_option])
        self.groups_widget = VBox([self.groups_setter])
        self.groups_panel = VBox(
            [self.group_auto, self.group_list, self.channels, self.group_color]
        )
        self.decay_panel = VBox([self.show_decay])
        self.scale_panel = VBox([self.scale_button, self.scale_value])
        self.plotting = interactive_output(
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

        # Set up interactions
        self.system_panel_option.observe(self.system_panel_trigger)
        self.objects.observe(self.objects_change, names="value")
        self.data.observe(self.set_data, names="value")
        self.system.observe(self.system_observer, names="value")
        self.tem_checkbox.observe(self.objects_change, names="value")
        self.groups_setter.observe(self.groups_trigger)
        self.previous_line = self.lines.lines.value
        self.scale_button.observe(self.scale_update)
        self.channel_selection.observe(self.channel_panel_update, names="value")
        self.channels.observe(self.edit_group, names="value")
        self.group_list.observe(self.highlight_selection, names="value")
        self.flip_sign.observe(self.set_data, names="value")
        self.highlight_selection(None)
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
        self.trigger.description = "Process All Lines"
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
        self.output_panel = VBox(
            [
                self.trigger_panel,
            ]
        )
        self.objects_change(None)

    @property
    def main(self):
        if getattr(self, "_main", None) is None:
            self._main = VBox(
                [
                    self.project_panel,
                    HBox(
                        [
                            VBox(
                                [self.data_panel, self.flip_sign],
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
                    self.plotting,
                    HBox(
                        [
                            VBox(
                                [Label("Visual Parameters"), self.visual_parameters],
                                layout=Layout(width="50%"),
                            ),
                            VBox(
                                [
                                    Label("Detection Parameters"),
                                    self.detection_parameters,
                                ],
                                layout=Layout(width="50%"),
                            ),
                        ]
                    ),
                    self.output_panel,
                ]
            )

        return self._main

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
    def data(self):
        """
        :obj:`ipywidgets.SelectMultiple`: Data selection used by the application
        """
        if getattr(self, "_data", None) is None:
            self.data = Dropdown(description="Data: ")

        return self._data

    @data.setter
    def data(self, value):
        assert isinstance(
            value, (Dropdown, SelectMultiple)
        ), f"'Objects' must be of type {Dropdown} or {SelectMultiple}"
        self._data = value

    @property
    def em_system_specs(self):
        return geophysical_systems.parameters()

    @property
    def flip_sign(self) -> ToggleButton:
        """
        Apply a sign flip to the selected data
        """
        if getattr(self, "_flip_sign", None) is None:
            self._flip_sign = ToggleButton(
                description="Flip Y (-1x)", button_style="warning"
            )

        return self._flip_sign

    @property
    def group_auto(self) -> Button:
        """
        Auto-create groups (3) from selected data channels.
        """
        if getattr(self, "_flip_sign", None) is None:
            self._group_auto = Button(description="Create groups [E | M | L]")

        return self._group_auto

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
                    "middle",
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
    def line_field(self):
        """
        Alias of lines.data widget
        """
        return self.lines.data

    @property
    def line_id(self):
        """
        Alias of lines.lines widget
        """
        return self.lines.lines

    @property
    def lines(self):
        """
        :obj:`geoapps.selection.LineOptions`: Line selection widget defining the profile used for plotting.
        """
        if getattr(self, "_lines", None) is None:
            self._lines = LineOptions(
                workspace=self.workspace, multiple_lines=False, objects=self.objects
            )

        return self._lines

    @property
    def markers(self):
        """
        :obj:`ipywidgets.ToggleButton`: Display markers on the data plot
        """
        if getattr(self, "_markers", None) is None:
            self._markers = ToggleButton(description="Show markers", value=True)

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
    def residual(self):
        """
        :obj:`ipywidgets.Checkbox`: Use the residual between the original and smoothed data profile
        """
        if getattr(self, "_residual", None) is None:
            self._residual = Checkbox(description="Show residual", value=False)

        return self._residual

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
    def show_decay(self):
        """
        :obj:`ipywidgets.ToggleButton`: Display the decay curve plot
        """
        if getattr(self, "_show_decay", None) is None:
            self._show_decay = ToggleButton(description="Show decay", value=False)

        return self._show_decay

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
        self.params.input_file.filepath = path.join(
            path.dirname(self._h5file), self.ga_group_name.value + ".ui.json"
        )
        self.update_objects_list()
        self.lines._workspace = workspace

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
        Observer of :obj:`geoapps.processing.PeakFinder.data`
        Populate the list of available channels and refresh groups
        """
        if getattr(self, "survey", None) is not None and self.data.value is not None:
            self.pause_plot_refresh = True
            groups = [[p_g.name, p_g.uid] for p_g in self.survey.property_groups]
            # channels = []

            # # Add all selected data channels | groups once
            # if self.data.value in groups:
            #     for prop in self.survey.find_or_create_property_group(
            #         name=self.data.uid_name_map[self.data.value]
            #     ).properties:
            #         name = self.workspace.get_entity(prop)[0].name
            #         if prop not in channels:
            #             channels.append(name)
            # else:
            #     channels.append(self.data.value)

            self.channels.options = groups
            for group in groups:
                # if self.survey.get_data(channel):
                for channel in group.properties:
                    obj = self.workspace.get_entity(channel)[0]
                    self.data_channels[obj.name] = obj

            # Generate default groups
            self.reset_groups()

            if self.tem_checkbox.value:
                for key, widget in self.data_channel_options.items():
                    widget.children[0].options = groups
                    widget.children[0].value = find_value(groups, [key])
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
                self.tem_box.children = [self.tem_checkbox, self.groups_widget]
                self.min_channels.disabled = True
            else:
                self.tem_box.children = [
                    self.tem_checkbox,
                    self.system_panel,
                    self.groups_widget,
                    self.decay_panel,
                ]
                self.min_channels.disabled = False

            self.set_data(None)

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

        self.reset_groups()
        self.plot_trigger.value = False
        self.plot_trigger.value = True

    # def run_all_click(self, _):
    #     """
    #     Observer of :obj:`geoapps.processing.PeakFinder.`: to process the entire Curve object for all lines
    #     """
    #     self.run_all.description = "Computing..."
    #     anomalies = []
    #     vertices = self.client.scatter(self.survey.vertices)
    #     channels = self.client.scatter(self.active_channels)
    #     time_groups = self.client.scatter(self.time_groups)
    #     for line_id in list(self.lines.lines.options)[1:]:
    #         line_indices = self.get_line_indices(line_id)
    #
    #         if line_indices is None:
    #             continue
    #
    #         anomalies += [
    #             self.client.compute(
    #                 find_anomalies(
    #                     vertices,
    #                     line_indices,
    #                     channels,
    #                     time_groups,
    #                     data_normalization=self.em_system_specs[self.system.value][
    #                         "normalization"
    #                     ],
    #                     smoothing=self.smoothing.value,
    #                     # use_residual=self.residual.value,
    #                     min_amplitude=self.min_amplitude.value,
    #                     min_value=self.min_value.value,
    #                     min_width=self.min_width.value,
    #                     max_migration=self.max_migration.value,
    #                     min_channels=self.min_channels.value,
    #                     minimal_output=True,
    #                 )
    #             )
    #         ]
    #
    #     self.all_anomalies = self.client.gather(anomalies)
    #     self.trigger.button_style = "success"
    #     self.run_all.description = "Process All Lines"

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

    # def trigger_click(self, _):
    #     """
    #     Observer of :obj:`geoapps.processing.PeakFinder.`:
    #     """
    #     if not self.all_anomalies:
    #         return
    #
    #     if not self.tem_checkbox:
    #         self.regroup()
    #
    #     # Append all lines
    #

    def highlight_selection(self, _):
        """
        Observer of :obj:`geoapps.processing.PeakFinder.`:: Highlight the time group data selection
        """
        for group in self.time_groups.values():
            if group["name"] == self.group_list.value:
                self.group_color.value = group["color"]
                self.channels.value = group["channels"]
                if "+" in group["name"]:
                    self.channels.disabled = True
                else:
                    self.channels.disabled = False

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

            if residual:
                raw = self.lines.profile._values_resampled_raw
                axs.fill_between(
                    locs, values, raw, where=raw > values, color=[1, 0, 0, 0.5]
                )
                axs.fill_between(
                    locs, values, raw, where=raw < values, color=[0, 0, 1, 0.5]
                )

        if scale == "symlog":
            plt.yscale("symlog", linthresh=scale_value)

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
            len(self.workspace.get_entity(self.lines.data.value)) == 0
            or self.lines.lines.value == ""
        ):
            return

        line_indices = self.get_line_indices(self.lines.lines.value)

        if line_indices is None:
            return
        self.survey.line_indices = line_indices
        result = find_anomalies(
            self.survey.vertices,
            line_indices,
            self.active_channels,
            self.time_groups,
            data_normalization=self.em_system_specs[self.system.value]["normalization"],
            smoothing=self.smoothing.value,
            min_amplitude=self.min_amplitude.value,
            min_value=self.min_value.value,
            min_width=self.min_width.value,
            max_migration=self.max_migration.value,
            min_channels=self.min_channels.value,
            return_profile=True,
        )

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
        self.pause_plot_refresh = False

    def get_line_indices(self, line_id):
        """
        Find the vertices for a given line ID
        """
        line_data = self.workspace.get_entity(self.lines.data.value)[0]

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
                    for channel in self.data.value:
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

    @classmethod
    def run(cls, params):
        """
        Create an octree mesh from input values
        """

        try:
            client = get_client()
        except:
            client = Client()

        workspace = params.workspace

        survey = workspace.get_entity(params.objects)[0]
        line_field = workspace.get_entity(params.line_field)[0]
        lines = np.unique(line_field.values)
        prop_group = [p_g for p_g in survey.property_groups if p_g.uid == params.data][
            0
        ]
        active_channels = []
        for prop in prop_group.properties:
            channel = workspace.get_entity(prop)[0]
            active_channels += [{"name": channel.name, "values": channel.values}]

        vertices = client.scatter(survey.vertices)
        channels = client.scatter(active_channels)
        time_groups = client.scatter(time_groups)
        anomalies = []
        for line_id in list(lines):
            line_indices = line_field.values == line_id

            anomalies += [
                client.compute(
                    find_anomalies(
                        vertices,
                        line_indices,
                        channels,
                        time_groups,
                        data_normalization=geophysical_systems.parameters[system.value][
                            "normalization"
                        ],
                        smoothing=smoothing.value,
                        # use_residual=residual.value,
                        min_amplitude=min_amplitude.value,
                        min_value=min_value.value,
                        min_width=min_width.value,
                        max_migration=max_migration.value,
                        min_channels=min_channels.value,
                        minimal_output=True,
                    )
                )
            ]

        all_anomalies = client.gather(anomalies)

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

        # for line in self.all_anomalies:
        #     for group in line:
        #         if "time_group" in list(group.keys()) and len(group["cox"]) > 0:
        #             time_group += [group["time_group"]]
        #
        #             if group["linear_fit"] is None:
        #                 tau += [0]
        #             else:
        #                 tau += [np.abs(group["linear_fit"][0] ** -1.0)]
        #             migration += [group["migration"]]
        #             amplitude += [group["amplitude"]]
        #             azimuth += [group["azimuth"]]
        #             cox += [group["cox"]]
        #             inflx_dwn += [group["inflx_dwn"]]
        #             inflx_up += [group["inflx_up"]]
        #             start += [group["start"]]
        #             end += [group["end"]]
        #             skew += [group["skew"]]
        #             peaks += [group["peaks"]]
        #
        # if cox:
        #     time_group = np.hstack(time_group) + 1  # Start count at 1
        #
        #     # Create reference values and color_map
        #     group_map, color_map = {}, []
        #     for ind, group in self.time_groups.items():
        #         group_map[ind + 1] = group["name"]
        #         color_map += [[ind + 1] + hex_to_rgb(group["color"]) + [1]]
        #
        #     color_map = np.core.records.fromarrays(
        #         np.vstack(color_map).T, names=["Value", "Red", "Green", "Blue", "Alpha"]
        #     )
        #     points = Points.create(
        #         self.workspace,
        #         name="PointMarkers",
        #         vertices=np.vstack(cox),
        #         parent=self.ga_group,
        #     )
        #     points.entity_type.name = self.ga_group_name.value
        #     migration = np.hstack(migration)
        #     dip = migration / migration.max()
        #     dip = np.rad2deg(np.arccos(dip))
        #     skew = np.hstack(skew)
        #     azimuth = np.hstack(azimuth)
        #     points.add_data(
        #         {
        #             "amplitude": {"values": np.hstack(amplitude)},
        #             "skew": {"values": skew},
        #         }
        #     )
        #
        #     if self.tem_checkbox.value:
        #         points.add_data(
        #             {
        #                 "tau": {"values": np.hstack(tau)},
        #                 "azimuth": {"values": azimuth},
        #                 "dip": {"values": dip},
        #             }
        #         )
        #
        #     time_group_data = points.add_data(
        #         {
        #             "time_group": {
        #                 "type": "referenced",
        #                 "values": np.hstack(time_group),
        #                 "value_map": group_map,
        #             }
        #         }
        #     )
        #     time_group_data.entity_type.color_map = {
        #         "name": "Time Groups",
        #         "values": color_map,
        #     }
        #
        #     if self.tem_checkbox.value:
        #         group = points.find_or_create_property_group(
        #             name="AzmDip", property_group_type="Dip direction & dip"
        #         )
        #         group.properties = [
        #             points.get_data("azimuth")[0].uid,
        #             points.get_data("dip")[0].uid,
        #         ]
        #
        #     # Add structural markers
        #     if self.structural_markers.value:
        #
        #         if self.tem_checkbox.value:
        #             markers = []
        #
        #             def rotation_2D(angle):
        #                 R = np.r_[
        #                     np.c_[
        #                         np.cos(np.pi * angle / 180),
        #                         -np.sin(np.pi * angle / 180),
        #                     ],
        #                     np.c_[
        #                         np.sin(np.pi * angle / 180), np.cos(np.pi * angle / 180)
        #                     ],
        #                 ]
        #                 return R
        #
        #             for azm, xyz, mig in zip(
        #                     np.hstack(azimuth).tolist(),
        #                     np.vstack(cox).tolist(),
        #                     migration.tolist(),
        #             ):
        #                 marker = np.r_[
        #                     np.c_[-0.5, 0.0] * 50,
        #                     np.c_[0.5, 0] * 50,
        #                     np.c_[0.0, 0.0],
        #                     np.c_[0.0, 1.0] * mig,
        #                 ]
        #
        #                 marker = (
        #                         np.c_[np.dot(rotation_2D(-azm), marker.T).T, np.zeros(4)]
        #                         + xyz
        #                 )
        #                 markers.append(marker.squeeze())
        #
        #             curves = Curve.create(
        #                 self.workspace,
        #                 name="TickMarkers",
        #                 vertices=np.vstack(markers),
        #                 cells=np.arange(len(markers) * 4, dtype="uint32").reshape(
        #                     (-1, 2)
        #                 ),
        #                 parent=self.ga_group,
        #             )
        #             time_group_data = curves.add_data(
        #                 {
        #                     "time_group": {
        #                         "type": "referenced",
        #                         "values": np.kron(np.hstack(time_group), np.ones(4)),
        #                         "value_map": group_map,
        #                     }
        #                 }
        #             )
        #             time_group_data.entity_type.color_map = {
        #                 "name": "Time Groups",
        #                 "values": color_map,
        #             }
        #         inflx_pts = Points.create(
        #             self.workspace,
        #             name="Inflections_Up",
        #             vertices=np.vstack(inflx_up),
        #             parent=self.ga_group,
        #         )
        #         time_group_data = inflx_pts.add_data(
        #             {
        #                 "time_group": {
        #                     "type": "referenced",
        #                     "values": np.repeat(
        #                         np.hstack(time_group), [ii.shape[0] for ii in inflx_up]
        #                     ),
        #                     "value_map": group_map,
        #                 }
        #             }
        #         )
        #         time_group_data.entity_type.color_map = {
        #             "name": "Time Groups",
        #             "values": color_map,
        #         }
        #         inflx_pts = Points.create(
        #             self.workspace,
        #             name="Inflections_Down",
        #             vertices=np.vstack(inflx_dwn),
        #             parent=self.ga_group,
        #         )
        #         time_group_data.copy(parent=inflx_pts)
        #
        #         start_pts = Points.create(
        #             self.workspace,
        #             name="Starts",
        #             vertices=np.vstack(start),
        #             parent=self.ga_group,
        #         )
        #         time_group_data.copy(parent=start_pts)
        #
        #         end_pts = Points.create(
        #             self.workspace,
        #             name="Ends",
        #             vertices=np.vstack(end),
        #             parent=self.ga_group,
        #         )
        #         time_group_data.copy(parent=end_pts)
        #
        #         peak_pts = Points.create(
        #             self.workspace,
        #             name="Peaks",
        #             vertices=np.vstack(peaks),
        #             parent=self.ga_group,
        #         )
        #
        # if self.live_link.value:
        #     self.live_link_output(self.export_directory.selected_path, points)
        #
        #     if self.structural_markers.value:
        #         self.live_link_output(self.export_directory.selected_path, curves)
        #
        # self.workspace.finalize()

        print(f"Selected object {survey}")

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
                np.abs(
                    np.min([values[peak] - values[start], values[peak] - values[end]])
                )
                / (np.std(values) + 2e-32)
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


if __name__ == "__main__":
    params = PeakFinderParams.from_path(sys.argv[1])
    print(params.geoh5)
    PeakFinder.run(params)
