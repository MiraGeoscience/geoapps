#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import sys
import uuid
import warnings
from copy import deepcopy
from os import path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from geoh5py.data import ReferencedData
from geoh5py.objects import Curve, ObjectBase
from geoh5py.shared import Entity
from geoh5py.ui_json import InputFile
from geoh5py.workspace import Workspace
from ipywidgets import (
    Box,
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
    ToggleButton,
    ToggleButtons,
    VBox,
    Widget,
    interactive_output,
)
from ipywidgets.widgets.widget_selection import TraitError

from geoapps.base.selection import LineOptions, ObjectDataSelection
from geoapps.peak_finder.constants import (
    app_initializer,
    default_ui_json,
    template_dict,
)
from geoapps.utils import geophysical_systems

from . import PeakFinderParams
from .driver import PeakFinderDriver
from .utils import default_groups_from_property_group, find_anomalies


class PeakFinder(ObjectDataSelection):
    """
    Application for the picking of targets along Time-domain EM profiles
    """

    _param_class = PeakFinderParams
    _add_groups = "only"
    _center = None
    _flip_sign = None
    _group_auto = None
    _group_list = None
    _group_display = None
    _groups_setter = None
    _lines = None
    _markers = None
    _max_migration = None
    _min_amplitude = None
    _min_channels = None
    _min_value = None
    _min_width = None
    _plot_trigger = None
    _residual = None
    _scale_button = None
    _scale_value = None
    _show_decay = None
    _smoothing = None
    _structural_markers = None
    _system = None
    _tem_checkbox = None
    _width = None
    _object_types = (Curve,)
    all_anomalies = []
    active_channels = {}
    _survey = None
    _channel_groups = {}
    pause_refresh = False
    decay_figure = None
    marker = {"left": "<", "right": ">"}
    plot_result = True

    def __init__(self, ui_json=None, plot_result=True, **kwargs):
        self.plot_result = plot_result
        app_initializer.update(kwargs)
        if ui_json is not None and path.exists(ui_json):
            self.params = self._param_class(InputFile(ui_json))
        else:
            self.params = self._param_class(**app_initializer)

        self.defaults = {}
        for key, value in self.params.to_dict().items():
            if isinstance(value, Entity):
                self.defaults[key] = value.uid
            else:
                self.defaults[key] = value

        self.groups_panel = VBox([])
        self.group_auto.observe(self.create_default_groups, names="value")
        self.objects.observe(self.objects_change, names="value")
        self.groups_widget = HBox([self.group_auto, self.groups_setter])
        self.decay_panel = VBox([self.show_decay])
        self.tem_box = HBox(
            [
                self.tem_checkbox,
                self.system,
                self.decay_panel,
            ]
        )
        self.data.observe(self.set_data, names="value")
        self.system.observe(self.set_data, names="value")
        self.previous_line = None
        super().__init__(**self.defaults)
        self.pause_refresh = False
        self.refresh.value = True
        self.previous_line = self.lines.lines.value
        self.smoothing.observe(self.line_update, names="value")
        self.max_migration.observe(self.line_update, names="value")
        self.min_channels.observe(self.line_update, names="value")
        self.min_amplitude.observe(self.line_update, names="value")
        self.min_value.observe(self.line_update, names="value")
        self.min_width.observe(self.line_update, names="value")
        self.lines.lines.observe(self.line_update, names="value")
        self.scale_panel = VBox([self.scale_button, self.scale_value])
        self.plotting = interactive_output(
            self.plot_data_selection,
            {
                "residual": self.residual,
                "markers": self.markers,
                "scale": self.scale_button,
                "scale_value": self.scale_value,
                "center": self.center,
                "width": self.width,
                "plot_trigger": self.plot_trigger,
                "refresh": self.refresh,
                "x_label": self.x_label,
            },
        )
        self.decay = interactive_output(
            self.plot_decay_curve,
            {
                "center": self.center,
                "plot_trigger": self.plot_trigger,
            },
        )
        self.group_display.observe(self.update_center, names="value")
        self.show_decay.observe(self.show_decay_trigger, names="value")
        self.tem_checkbox.observe(self.tem_change, names="value")
        self.groups_setter.observe(self.groups_trigger)
        self.scale_button.observe(self.scale_update)
        self.flip_sign.observe(self.set_data, names="value")
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
                self.group_display,
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
        self.line_update(None)

    def __populate__(self, **kwargs):
        super().__populate__(**kwargs)

        obj_list = self.workspace.get_entity(self.objects.value)

        if any(obj_list) and any(self.params.free_parameter_dict):
            self._channel_groups = self.params.groups_from_free_params()

            group_list = []
            for pg, params in self._channel_groups.items():
                group_list += [self.add_group_widget(pg, params)]
            self.groups_panel.children = group_list

        else:
            if not self.group_auto.value:
                self.group_auto.value = True
            else:
                self.create_default_groups(None)

    @property
    def main(self) -> VBox:
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
                    Label("Groups"),
                    self.groups_widget,
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
    def center(self) -> FloatSlider:
        """
        Adjust the data plot center position along line
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
    def em_system_specs(self) -> dict:
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
    def group_auto(self) -> ToggleButton:
        """
        Auto-create groups (3) from selected data channels.
        """
        if getattr(self, "_group_auto", None) is None:
            self._group_auto = ToggleButton(
                description="Use/Create Default", value=False
            )
        return self._group_auto

    @property
    def group_list(self) -> Dropdown:
        """
        List of default time data groups
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
    def group_display(self) -> Dropdown:
        """
        List of groups to chose from for display
        """
        if getattr(self, "_group_display", None) is None:
            self._group_display = Dropdown(description="Select Peak")
        return self._group_display

    @property
    def groups_setter(self) -> ToggleButton:
        """
        Display the group options panel
        """
        if getattr(self, "_groups_setter", None) is None:
            self._groups_setter = ToggleButton(
                description="Group Settings", value=False
            )

        return self._groups_setter

    @property
    def line_field(self) -> Dropdown:
        """
        Alias of lines.data widget
        """
        return self.lines.data

    @property
    def line_id(self) -> Dropdown:
        """
        Alias of lines.lines widget
        """
        return self.lines.lines

    @property
    def lines(self) -> LineOptions:
        """
        Line selection defining the profile used for plotting.
        """
        if getattr(self, "_lines", None) is None:
            self._lines = LineOptions(
                workspace=self.workspace, multiple_lines=False, objects=self.objects
            )

        return self._lines

    @property
    def markers(self) -> ToggleButton:
        """
        Display markers on the data plot
        """
        if getattr(self, "_markers", None) is None:
            self._markers = ToggleButton(description="Show markers", value=True)

        return self._markers

    @property
    def max_migration(self) -> FloatSlider:
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
                description="Max Peak Migration",
                style={"description_width": "initial"},
                disabled=False,
            )

        return self._max_migration

    @property
    def min_amplitude(self) -> IntSlider:
        """
        Filter small anomalies based on amplitude ratio
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
    def min_channels(self) -> IntSlider:
        """
        Filter peak groups based on minimum number of data channels overlap.
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
    def min_value(self) -> FloatText:
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
    def min_width(self) -> FloatSlider:
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
    def plot_trigger(self) -> ToggleButton:
        """
        Trigger refresh of all plots
        """
        if getattr(self, "_plot_trigger", None) is None:
            self._plot_trigger = ToggleButton(
                description="Pick nearest target", value=False
            )
        return self._plot_trigger

    @property
    def residual(self) -> Checkbox:
        """
        Use the residual between the original and smoothed data profile
        """
        if getattr(self, "_residual", None) is None:
            self._residual = Checkbox(description="Show residual", value=False)

        return self._residual

    @property
    def scale_button(self) -> ToggleButtons:
        """
        Scale the vertical axis of the data plot
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
    def scale_value(self) -> FloatLogSlider:
        """
        Threshold value used by th symlog scaling
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
    def show_decay(self) -> ToggleButton:
        """
        Display the decay curve plot
        """
        if getattr(self, "_show_decay", None) is None:
            self._show_decay = ToggleButton(description="Show decay", value=False)

        return self._show_decay

    @property
    def smoothing(self) -> IntSlider:
        """
        Number of neighboring data points used for the running mean smoothing
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
    def structural_markers(self) -> Checkbox:
        """
        Export peaks as structural markers
        """
        if getattr(self, "_structural_markers", None) is None:
            self._structural_markers = Checkbox(description="All Markers")

        return self._structural_markers

    @property
    def survey(self) -> Optional[Entity]:
        """
        Selected curve object
        """
        return self._survey

    @property
    def system(self) -> Dropdown:
        """
        Selection of a TEM system
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
    def tem_checkbox(self) -> Checkbox:
        """
        :obj:`ipywidgets.Checkbox`: Enable options specific to TEM data groups
        """
        if getattr(self, "_tem_checkbox", None) is None:
            self._tem_checkbox = Checkbox(description="TEM Data", value=True)

        return self._tem_checkbox

    @property
    def channel_groups(self) -> dict:
        """
        Dict of time groups used to classify peaks
        """
        return self._channel_groups

    @channel_groups.setter
    def channel_groups(self, groups: dict):
        self._channel_groups = groups

    @property
    def width(self) -> FloatSlider:
        """
        Adjust the length of data displayed on the data plot
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
    def workspace(self) -> Workspace:
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
        self.base_workspace_changes(workspace)
        self.update_objects_list()
        self.lines.workspace = workspace

    @property
    def x_label(self) -> ToggleButtons:
        """
        Units of distance displayed on the data plot
        """
        if getattr(self, "_x_label", None) is None:
            self._x_label = ToggleButtons(
                options=["Distance", "Easting", "Northing"],
                value="Distance",
                description="X-axis label:",
            )

        return self._x_label

    def add_group_widget(self, property_group, params: dict):
        """
        Add a group from dictionary
        """
        if getattr(self, f"Group {property_group} Data", None) is None:
            setattr(
                self,
                f"Group {property_group} Data",
                Dropdown(
                    description="Group Name:",
                ),
            )
        widget = getattr(self, f"Group {property_group} Data")
        widget.name = property_group
        widget.value = None
        widget.options = self.data.options

        try:
            widget.value = params["data"]
        except TraitError:
            pass
        if getattr(self, f"Group {property_group} Color", None) is None:
            setattr(
                self,
                f"Group {property_group} Color",
                ColorPicker(description="Color"),
            )
        getattr(self, f"Group {property_group} Color").name = property_group
        try:
            getattr(self, f"Group {property_group} Color").value = str(params["color"])
        except TraitError:
            pass

        getattr(self, f"Group {property_group} Data").observe(
            self.edit_group, names="value"
        )
        getattr(self, f"Group {property_group} Color").observe(
            self.edit_group, names="value"
        )
        return VBox(
            [
                getattr(self, f"Group {property_group} Data"),
                getattr(self, f"Group {property_group} Color"),
            ],
            layout=Layout(border="solid"),
        )

    def create_default_groups(self, _):
        if self.group_auto.value:
            obj = self.workspace.get_entity(self.objects.value)[0]
            group = [pg for pg in obj.property_groups if pg.uid == self.data.value]
            if any(group):
                channel_groups = default_groups_from_property_group(group[0])
                self._channel_groups = channel_groups
                self.pause_refresh = True

                group_list = []
                self.update_data_list(None)
                self.pause_refresh = True
                for pg, params in self._channel_groups.items():
                    group_list += [self.add_group_widget(pg, params)]

                self.pause_refresh = False
                self.groups_panel.children = group_list

                self.set_data(None)

        self.group_auto.value = False
        self._group_auto.button_style = "success"

    def edit_group(self, caller):
        """
        Observer of :obj:`geoapps.processing.peak_finder.`: Change channels associated with groups
        """
        widget = caller["owner"]
        if not self.pause_refresh:
            if isinstance(widget, Dropdown):
                obj, _ = self.get_selected_entities()
                group = {"color": getattr(self, f"Group {widget.name} Color").value}
                if widget.value in [pg.uid for pg in obj.property_groups]:
                    prop_group = [
                        pg for pg in obj.property_groups if pg.uid == widget.value
                    ]
                    group["data"] = prop_group[0].uid
                    group["properties"] = prop_group[0].properties
                else:
                    group["data"] = None
                    group["properties"] = []
                self._channel_groups[widget.name] = group
            else:
                self._channel_groups[widget.name]["color"] = widget.value
            self.set_data(None)

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

    def groups_trigger(self, _):
        """
        Observer of :obj:`geoapps.processing.peak_finder.`:
        """
        if self.groups_setter.value:
            self.groups_widget.children = [
                self.group_auto,
                self.groups_setter,
                self.groups_panel,
            ]
        else:
            self.groups_widget.children = [self.group_auto, self.groups_setter]

    def line_update(self, _):
        """
        Re-compute derivatives
        """
        if (
            getattr(self, "survey", None) is None
            or len(self.workspace.get_entity(self.lines.data.value)) == 0
            or self.lines.lines.value == ""
            or len(self.channel_groups) == 0
        ):
            return

        line_indices = self.get_line_indices(self.lines.lines.value)

        if line_indices is None:
            return

        self.plot_trigger.value = False
        self.survey.line_indices = line_indices
        result = find_anomalies(
            self.survey.vertices,
            line_indices,
            self.active_channels,
            self.channel_groups,
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
            self.group_display.disabled = True
            return

        self.pause_refresh = True
        if self.previous_line != self.lines.lines.value:
            end = self.lines.profile.locations_resampled[-1]
            mid = self.lines.profile.locations_resampled[-1] * 0.5

            if self.center.value >= end:
                self.center.value = 0
                self.center.max = end
                self.width.value = 0
                self.width.max = end
                self.width.value = mid
            else:
                self.center.max = end
                self.width.max = end

        if len(self.lines.anomalies) > 0:
            peaks = np.sort(
                self.lines.profile.locations_resampled[
                    [group["peak"][0] for group in self.lines.anomalies]
                ]
            )
            current = self.center.value
            self.group_display.options = np.round(peaks, decimals=1)
            self.group_display.value = self.group_display.options[
                np.argmin(np.abs(peaks - current))
            ]
        self.previous_line = self.lines.lines.value
        self.pause_refresh = False
        self.plot_trigger.value = True

    def objects_change(self, _):
        """
        Observer of :obj:`geoapps.processing.peak_finder.objects`: Reset data and auto-detect AEM system
        """
        if self.workspace.get_entity(self.objects.value):
            self._survey = self.workspace.get_entity(self.objects.value)[0]
            self.update_data_list(None)
            is_tem = False
            self.active_channels = {}
            self.channel_groups = {}
            for child in self.groups_panel.children:
                child.children[0].options = self.data.options

            for aem_system, specs in self.em_system_specs.items():
                if any(
                    [
                        specs["flag"] in channel
                        for channel in self._survey.get_data_list()
                    ]
                ):
                    if aem_system in self.system.options:
                        self.system.value = aem_system
                        is_tem = True
                        break

            self.tem_checkbox.value = is_tem

            if self.group_auto:
                self.create_default_groups(None)

            self.set_data(None)

    def tem_change(self, _):
        self.min_channels.disabled = not self.tem_checkbox.value
        self.show_decay.value = False
        self.system.disabled = not self.tem_checkbox.value

    def plot_data_selection(
        self,
        residual,
        markers,
        scale,
        scale_value,
        center,
        width,
        plot_trigger,
        refresh,
        x_label,
    ):
        """
        Observer of :obj:`geoapps.processing.peak_finder.`:
        """

        if (
            self.pause_refresh
            or not self.refresh.value
            or self.plot_trigger.value is False
            or not self.plot_result
        ):
            return

        self.figure = plt.figure(figsize=(12, 6))
        axs = plt.subplot()

        if (
            getattr(self, "survey", None) is None
            or getattr(self.survey, "line_indices", None) is None
            or len(self.survey.line_indices) < 2
            or len(self.active_channels) == 0
        ):
            return

        lims = np.searchsorted(
            self.lines.profile.locations_resampled,
            [
                (center - width / 2.0),
                (center + width / 2.0),
            ],
        )
        sub_ind = np.arange(lims[0], lims[1])
        if len(sub_ind) == 0:
            return

        y_min, y_max = np.inf, -np.inf
        locs = self.lines.profile.locations_resampled
        peak_markers_x, peak_markers_y, peak_markers_c = [], [], []
        end_markers_x, end_markers_y = [], []
        start_markers_x, start_markers_y = [], []
        up_markers_x, up_markers_y = [], []
        dwn_markers_x, dwn_markers_y = [], []

        for cc, (uid, channel) in enumerate(self.active_channels.items()):

            if "values" not in channel.keys():
                continue

            self.lines.profile.values = channel["values"][self.survey.line_indices]
            values = self.lines.profile.values_resampled
            y_min = np.min([values[sub_ind].min(), y_min])
            y_max = np.max([values[sub_ind].max(), y_max])
            axs.plot(locs, values, color=[0.5, 0.5, 0.5, 1])
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
                    color=group["channel_group"]["color"],
                )

                if group["azimuth"] < 180:
                    ori = "right"
                else:
                    ori = "left"

                if markers:
                    if ii == 0:
                        axs.scatter(
                            locs[group["peak"][ii]],
                            values[group["peak"][ii]],
                            s=200,
                            c="k",
                            marker=self.marker[ori],
                            zorder=10,
                        )
                    peak_markers_x += [locs[group["peak"][ii]]]
                    peak_markers_y += [values[group["peak"][ii]]]
                    peak_markers_c += [group["channel_group"]["color"]]
                    start_markers_x += [locs[group["start"][ii]]]
                    start_markers_y += [values[group["start"][ii]]]
                    end_markers_x += [locs[group["end"][ii]]]
                    end_markers_y += [values[group["end"][ii]]]
                    up_markers_x += [locs[group["inflx_up"][ii]]]
                    up_markers_y += [values[group["inflx_up"][ii]]]
                    dwn_markers_x += [locs[group["inflx_dwn"][ii]]]
                    dwn_markers_y += [values[group["inflx_dwn"][ii]]]

            if residual:
                raw = self.lines.profile._values_resampled_raw
                axs.fill_between(
                    locs, values, raw, where=raw > values, color=[1, 0, 0, 0.5]
                )
                axs.fill_between(
                    locs, values, raw, where=raw < values, color=[0, 0, 1, 0.5]
                )

        if np.isinf(y_min):
            return

        if scale == "symlog":
            plt.yscale("symlog", linthresh=scale_value)

        x_lims = [
            center - width / 2.0,
            center + width / 2.0,
        ]
        y_lims = [np.max([y_min, self.min_value.value]), y_max]
        axs.set_xlim(x_lims)
        axs.set_ylim(y_lims)
        axs.set_ylabel("Data")
        axs.plot([center, center], [y_min, y_max], "k--")

        if markers:
            axs.scatter(
                peak_markers_x,
                peak_markers_y,
                s=50,
                c=peak_markers_c,
                marker="o",
            )
            axs.scatter(
                start_markers_x,
                start_markers_y,
                s=100,
                color="k",
                marker="4",
            )
            axs.scatter(
                end_markers_x,
                end_markers_y,
                s=100,
                color="k",
                marker="3",
            )
            axs.scatter(
                up_markers_x,
                up_markers_y,
                color="k",
                marker="1",
                s=100,
            )
            axs.scatter(
                dwn_markers_x,
                dwn_markers_y,
                color="k",
                marker="2",
                s=100,
            )

        ticks_loc = axs.get_xticks().tolist()
        axs.set_xticks(ticks_loc)

        if x_label == "Easting":
            axs.text(
                center,
                y_lims[0],
                f"{self.lines.profile.interp_x(center):.0f} m E",
                va="top",
                ha="center",
                bbox={"edgecolor": "r"},
            )
            axs.set_xticklabels(
                [f"{self.lines.profile.interp_x(label):.0f}" for label in ticks_loc]
            )
            axs.set_xlabel("Easting (m)")

        elif x_label == "Northing":
            axs.text(
                center,
                y_lims[0],
                f"{self.lines.profile.interp_y(center):.0f} m N",
                va="top",
                ha="center",
                bbox={"edgecolor": "r"},
            )
            axs.set_xticklabels(
                [f"{self.lines.profile.interp_y(label):.0f}" for label in ticks_loc]
            )
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
        plt.show()

    def plot_decay_curve(self, center, plot_trigger):
        """
        Observer of :obj:`geoapps.processing.peak_finder.`:
        """
        if self.pause_refresh or not self.plot_result:
            return

        if (
            self.plot_trigger.value
            or self.refresh.value
            and hasattr(self.lines, "profile")
            and self.tem_checkbox.value
        ):

            if self.decay_figure is None:
                self.decay_figure = plt.figure(figsize=(8, 8))

            else:
                plt.figure(self.decay_figure.number)

            axs = plt.subplot()
            # Find nearest decay to cursor
            group = None
            if getattr(self.lines, "anomalies", None) is not None:
                peaks = np.r_[[group["peak"][0] for group in self.lines.anomalies]]
                if len(peaks) > 0:
                    group = self.lines.anomalies[
                        np.argmin(
                            np.abs(
                                self.lines.profile.locations_resampled[peaks] - center
                            )
                        )
                    ]

            # Get the times of the group and plot the linear regression
            times = []
            if group is not None and group["linear_fit"] is not None:
                times = [
                    channel["time"]
                    for ii, channel in enumerate(self.active_channels.values())
                    if ii in list(group["channels"])
                ]
            if any(times):
                times = np.hstack(times)
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
                    color=group["channel_group"]["color"],
                    marker="^",
                    edgecolors="k",
                )
                axs.grid(True)

                plt.yscale("log")
                axs.set_ylabel("log(V)")
                axs.set_xlabel("Time (sec)")
                axs.set_title("Decay - MADTau")
            else:
                axs.set_ylabel("log(V)")
                axs.set_xlabel("Time (sec)")
                axs.set_title("Too few channels")
        plt.show()

    def scale_update(self, _):
        """
        Observer of :obj:`geoapps.processing.peak_finder.`:
        """
        if self.scale_button.value == "symlog":
            self.scale_panel.children = [
                self.scale_button,
                self.scale_value,
            ]
        else:
            self.scale_panel.children = [self.scale_button]

    def set_data(self, _):
        """
        Observer of :obj:`geoapps.processing.peak_finder.data`
        Populate the list of available channels and refresh groups
        """
        self._group_auto.button_style = "warning"
        if getattr(self, "survey", None) is not None and self.data.value is not None:
            self.pause_refresh = True
            self.active_channels = {}
            for group in self.channel_groups.values():
                for channel in group["properties"]:
                    obj = self.workspace.get_entity(channel)[0]

                    if getattr(obj, "values", None) is not None:
                        self.active_channels[channel] = {"name": obj.name}

            d_min, d_max = np.inf, -np.inf
            thresh_value = np.inf
            if self.tem_checkbox.value:
                system = self.em_system_specs[self.system.value]

            for uid, params in self.active_channels.copy().items():
                obj = self.workspace.get_entity(uid)[0]
                try:
                    if self.tem_checkbox.value:
                        channel = [
                            ch
                            for ch in system["channels"].keys()
                            if ch in params["name"]
                        ]
                        if any(channel):
                            self.active_channels[uid]["time"] = system["channels"][
                                channel[0]
                            ]
                        else:
                            del self.active_channels[uid]

                    self.active_channels[uid]["values"] = (
                        -1.0
                    ) ** self.flip_sign.value * obj.values.copy()
                    thresh_value = np.min(
                        [
                            thresh_value,
                            np.percentile(
                                np.abs(self.active_channels[uid]["values"]), 95
                            ),
                        ]
                    )
                    d_min = np.min([d_min, self.active_channels[uid]["values"].min()])
                    d_max = np.max([d_max, self.active_channels[uid]["values"].max()])
                except KeyError:
                    continue

            self.pause_refresh = False
            self.plot_trigger.value = False

            if d_max > -np.inf:
                self.plot_trigger.value = False
                self.min_value.value = d_min
                self.scale_value.value = thresh_value

            self.line_update(None)

    def trigger_click(self, _):
        param_dict = {}
        ui_json = deepcopy(default_ui_json)
        for key in ui_json:
            try:
                if isinstance(getattr(self, key), Widget) and hasattr(self.params, key):
                    value = getattr(self, key).value

                    if (
                        isinstance(value, uuid.UUID)
                        and self.workspace.get_entity(value)[0] is not None
                    ):
                        value = self.workspace.get_entity(value)[0]

                    param_dict[key] = value

            except AttributeError:
                continue

        for label, group in self._channel_groups.items():
            for member in ["data", "color"]:
                name = f"{label} {member}"
                ui_json[name] = deepcopy(template_dict[member])
                ui_json[name]["group"] = f"Group {label}"
                param_dict[name] = group[member]

        new_workspace = self.get_output_workspace(
            self.export_directory.selected_path, self.ga_group_name.value
        )
        for key, value in param_dict.items():
            if isinstance(value, ObjectBase):
                if new_workspace.get_entity(value.uid)[0] is None:
                    param_dict[key] = value.copy(
                        parent=new_workspace, copy_children=True
                    )
                    line_field = [
                        c for c in param_dict[key].children if c.name == "Line"
                    ]
                    if line_field:
                        param_dict["line_field"] = line_field[0]

        param_dict["geoh5"] = new_workspace
        if self.live_link.value:
            param_dict["monitoring_directory"] = self.monitoring_directory

        ifile = InputFile(
            ui_json=ui_json,
            validations=self.params.validations,
            validation_options={"disabled": True},
        )

        new_params = PeakFinderParams(input_file=ifile, **param_dict)
        new_params.write_input_file()
        self.run(new_params)

        if self.live_link.value:
            print("Live link active. Check your ANALYST session for result.")

    def update_center(self, _):
        """
        Update the center view on group selection
        """
        if hasattr(self.lines, "anomalies"):
            self.center.value = self.group_display.value

    @staticmethod
    def run(params, output_group=None):
        """
        Create an octree mesh from input values
        """

        driver = PeakFinderDriver(params)
        driver.run(output_group)

    def show_decay_trigger(self, _):
        """
        Observer of :obj:`geoapps.processing.peak_finder.`: Add the decay curve plot
        """
        if self.show_decay.value:
            self.decay_panel.children = [self.show_decay, self.decay]
            self.show_decay.description = "Hide decay curve"
        else:
            self.decay_panel.children = [self.show_decay]
            self.show_decay.description = "Show decay curve"


if __name__ == "__main__":
    file = sys.argv[1]
    warnings.warn(
        "'geoapps.peak_finder.application' replaced by "
        "'geoapps.peak_finder.driver' in version 0.7.0. "
        "This warning is likely due to the execution of older ui.json files. Please update."
    )
    params = PeakFinderParams(InputFile(file))
    PeakFinder.run(params)
