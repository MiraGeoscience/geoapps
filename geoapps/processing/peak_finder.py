#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import sys
import uuid
from copy import deepcopy
from os import path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from dask import delayed
from dask.distributed import Client, get_client
from geoh5py.data import ReferencedData
from geoh5py.groups import ContainerGroup
from geoh5py.objects import Curve, Points
from geoh5py.shared import Entity
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
    ToggleButton,
    ToggleButtons,
    VBox,
    Widget,
    interactive_output,
)
from ipywidgets.widgets.widget_selection import TraitError
from tqdm import tqdm

from geoapps.base import BaseApplication
from geoapps.io import InputFile
from geoapps.io.PeakFinder import PeakFinderParams
from geoapps.io.PeakFinder.constants import app_initializer, default_ui_json
from geoapps.selection import LineOptions, ObjectDataSelection
from geoapps.utils import geophysical_systems
from geoapps.utils.formatters import string_name
from geoapps.utils.utils import LineDataDerivatives, hex_to_rgb, running_mean

_default_channel_groups = {
    "early": {"label": ["early"], "color": "#0000FF", "channels": []},
    "middle": {"label": ["middle"], "color": "#FFFF00", "channels": []},
    "late": {"label": ["late"], "color": "#FF0000", "channels": []},
    "early + middle": {
        "label": ["early", "middle"],
        "color": "#00FFFF",
        "channels": [],
    },
    "early + middle + late": {
        "label": ["early", "middle", "late"],
        "color": "#008000",
        "channels": [],
    },
    "middle + late": {
        "label": ["middle", "late"],
        "color": "#FFA500",
        "channels": [],
    },
}


class PeakFinder(ObjectDataSelection):
    """
    Application for the picking of targets along Time-domain EM profiles
    """

    defaults = {}
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

    def __init__(self, ui_json=None, **kwargs):
        app_initializer.update(kwargs)
        if ui_json is not None and path.exists(ui_json):
            self.params = self._param_class(InputFile(ui_json))
        else:
            if "h5file" in app_initializer.keys():
                app_initializer["geoh5"] = app_initializer.pop("h5file")

            self.params = self._param_class(**app_initializer)

        self.defaults.update(self.params.to_dict(ui_json_format=False))
        self.defaults.pop("workspace", None)
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
        super().__init__()
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
        self.tem_checkbox.observe(self.objects_change, names="value")
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

        if any(obj_list) and any(self.params._free_param_dict):
            self._channel_groups = groups_from_params_dict(
                obj_list[0], self.params._free_param_dict
            )

        group_list = []
        for pg, params in self._channel_groups.items():
            group_list += [self.add_group_widget(pg, params)]

        self.groups_panel.children = group_list

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
                channel_groups = self.default_groups_from_property_group(group[0])
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

    @staticmethod
    def default_groups_from_property_group(property_group, start_index=0):
        parent = property_group.parent

        data_list = [
            parent.workspace.get_entity(uid)[0] for uid in property_group.properties
        ]

        start = start_index
        end = len(data_list)
        block = int((end - start) / 3)
        ranges = {
            "early": np.arange(start, start + block).tolist(),
            "middle": np.arange(start + block, start + 2 * block).tolist(),
            "late": np.arange(start + 2 * block, end).tolist(),
        }

        channel_groups = {}
        for ii, (key, default) in enumerate(_default_channel_groups.items()):
            prop_group = parent.find_or_create_property_group(name=key)
            prop_group.properties = []

            for val in default["label"]:
                for ind in ranges[val]:
                    prop_group.properties += [data_list[ind].uid]

            channel_groups[prop_group.name] = {
                "data": prop_group.uid,
                "color": default["color"],
                "label": [ii + 1],
                "properties": prop_group.properties,
            }

        return channel_groups

    def edit_group(self, caller):
        """
        Observer of :obj:`geoapps.processing.PeakFinder.`: Change channels associated with groups
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
        Observer of :obj:`geoapps.processing.PeakFinder.`:
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
        Observer of :obj:`geoapps.processing.PeakFinder.objects`: Reset data and auto-detect AEM system
        """
        if self.workspace.get_entity(self.objects.value):
            self._survey = self.workspace.get_entity(self.objects.value)[0]
            self.update_data_list(None)
            not_tem = True
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
                        not_tem = False
                        break

            if not_tem:
                self.tem_checkbox.value = False
                self.min_channels.disabled = True
                self.show_decay.value = False
                self.system.disabled = True
            else:
                self.tem_checkbox.value = True
                self.min_channels.disabled = False
                self.system.disabled = False

            if self.group_auto:
                self.create_default_groups(None)

            self.set_data(None)

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
        Observer of :obj:`geoapps.processing.PeakFinder.`:
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
        Observer of :obj:`geoapps.processing.PeakFinder.`:
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
        Observer of :obj:`geoapps.processing.PeakFinder.`:
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
        Observer of :obj:`geoapps.processing.PeakFinder.data`
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

        new_workspace = Workspace(
            path.join(
                self.export_directory.selected_path,
                self._ga_group_name.value + ".geoh5",
            )
        )
        obj, data = self.get_selected_entities()

        new_obj = new_workspace.get_entity(obj.uid)[0]
        if new_obj is None:
            obj.copy(parent=new_workspace, copy_children=True)

        self.params.geoh5 = new_workspace.h5file
        self.params.workspace = new_workspace

        param_dict = {}
        for key, value in self.__dict__.items():
            try:
                if isinstance(getattr(self, key), Widget):
                    # setattr(self.params, key, getattr(self, key).value)
                    param_dict[key] = getattr(self, key).value
            except AttributeError:
                continue

        self.params.update(param_dict)
        self.params.line_field = self.lines.data.value
        ui_json = deepcopy(self.params.default_ui_json)
        self.params.group_auto = False
        self.params.write_input_file(
            ui_json=ui_json,
            name=path.join(
                self.export_directory.selected_path,
                self._ga_group_name.value + ".ui.json",
            ),
        )
        self.run(self.params)

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
        print("Reading parameters...")
        try:
            client = get_client()
        except ValueError:
            client = Client()

        workspace = params.workspace
        survey = workspace.get_entity(params.objects)[0]
        prop_group = [pg for pg in survey.property_groups if pg.uid == params.data]

        if params.tem_checkbox:
            system = geophysical_systems.parameters()[params.system]
            normalization = system["normalization"]
        else:
            normalization = [1]

        if output_group is None:
            output_group = ContainerGroup.create(
                workspace, name=string_name(params.ga_group_name)
            )

        line_field = workspace.get_entity(params.line_field)[0]
        lines = np.unique(line_field.values)

        if params.group_auto and any(prop_group):
            channel_groups = PeakFinder.default_groups_from_property_group(
                prop_group[0]
            )
        else:

            channel_groups = groups_from_params_dict(survey, params._free_param_dict)

        active_channels = {}
        for group in channel_groups.values():
            for channel in group["properties"]:
                obj = workspace.get_entity(channel)[0]
                active_channels[channel] = {"name": obj.name}

        for uid, channel_params in active_channels.items():
            obj = workspace.get_entity(uid)[0]
            if params.tem_checkbox:
                channel = [ch for ch in system["channels"].keys() if ch in obj.name]
                if any(channel):
                    channel_params["time"] = system["channels"][channel[0]]
                else:
                    continue
            channel_params["values"] = client.scatter(
                obj.values.copy() * (-1.0) ** params.flip_sign
            )

        print("Submitting parallel jobs:")
        anomalies = []
        locations = client.scatter(survey.vertices.copy())

        for line_id in tqdm(list(lines)):
            line_indices = np.where(line_field.values == line_id)[0]

            anomalies += [
                client.compute(
                    delayed(find_anomalies)(
                        locations,
                        line_indices,
                        active_channels,
                        channel_groups,
                        data_normalization=normalization,
                        smoothing=params.smoothing,
                        min_amplitude=params.min_amplitude,
                        min_value=params.min_value,
                        min_width=params.min_width,
                        max_migration=params.max_migration,
                        min_channels=params.min_channels,
                        minimal_output=True,
                    )
                )
            ]
        (
            channel_group,
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

        print("Processing and collecting results:")
        for future_line in tqdm(anomalies):
            line = future_line.result()
            for group in line:
                if "channel_group" in group.keys() and len(group["cox"]) > 0:
                    channel_group += group["channel_group"]["label"]

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

        print("Exporting...")
        if cox:
            channel_group = np.hstack(channel_group)  # Start count at 1

            # Create reference values and color_map
            group_map, color_map = {}, []
            for ind, (name, group) in enumerate(channel_groups.items()):
                group_map[ind + 1] = name
                color_map += [[ind + 1] + hex_to_rgb(group["color"]) + [1]]

            color_map = np.core.records.fromarrays(
                np.vstack(color_map).T, names=["Value", "Red", "Green", "Blue", "Alpha"]
            )
            points = Points.create(
                params.workspace,
                name="PointMarkers",
                vertices=np.vstack(cox),
                parent=output_group,
            )
            points.entity_type.name = params.ga_group_name
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

            if params.tem_checkbox:
                points.add_data(
                    {
                        "tau": {"values": np.hstack(tau)},
                        "azimuth": {"values": azimuth},
                        "dip": {"values": dip},
                    }
                )

            channel_group_data = points.add_data(
                {
                    "channel_group": {
                        "type": "referenced",
                        "values": np.hstack(channel_group),
                        "value_map": group_map,
                    }
                }
            )
            channel_group_data.entity_type.color_map = {
                "name": "Time Groups",
                "values": color_map,
            }

            if params.tem_checkbox:
                group = points.find_or_create_property_group(
                    name="AzmDip", property_group_type="Dip direction & dip"
                )
                group.properties = [
                    points.get_data("azimuth")[0].uid,
                    points.get_data("dip")[0].uid,
                ]

            # Add structural markers
            if params.structural_markers:

                if params.tem_checkbox:
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
                        params.workspace,
                        name="TickMarkers",
                        vertices=np.vstack(markers),
                        cells=np.arange(len(markers) * 4, dtype="uint32").reshape(
                            (-1, 2)
                        ),
                        parent=output_group,
                    )
                    channel_group_data = curves.add_data(
                        {
                            "channel_group": {
                                "type": "referenced",
                                "values": np.kron(np.hstack(channel_group), np.ones(4)),
                                "value_map": group_map,
                            }
                        }
                    )
                    channel_group_data.entity_type.color_map = {
                        "name": "Time Groups",
                        "values": color_map,
                    }
                inflx_pts = Points.create(
                    params.workspace,
                    name="Inflections_Up",
                    vertices=np.vstack(inflx_up),
                    parent=output_group,
                )
                channel_group_data = inflx_pts.add_data(
                    {
                        "channel_group": {
                            "type": "referenced",
                            "values": np.repeat(
                                np.hstack(channel_group),
                                [ii.shape[0] for ii in inflx_up],
                            ),
                            "value_map": group_map,
                        }
                    }
                )
                channel_group_data.entity_type.color_map = {
                    "name": "Time Groups",
                    "values": color_map,
                }
                inflx_pts = Points.create(
                    params.workspace,
                    name="Inflections_Down",
                    vertices=np.vstack(inflx_dwn),
                    parent=output_group,
                )
                channel_group_data.copy(parent=inflx_pts)

                start_pts = Points.create(
                    params.workspace,
                    name="Starts",
                    vertices=np.vstack(start),
                    parent=output_group,
                )
                channel_group_data.copy(parent=start_pts)

                end_pts = Points.create(
                    params.workspace,
                    name="Ends",
                    vertices=np.vstack(end),
                    parent=output_group,
                )
                channel_group_data.copy(parent=end_pts)

                Points.create(
                    params.workspace,
                    name="Peaks",
                    vertices=np.vstack(peaks),
                    parent=output_group,
                )

        workspace.finalize()
        print("Process completed.")
        print(f"Result exported to: {workspace.h5file}")

        if params.monitoring_directory is not None and path.exists(
            params.monitoring_directory
        ):
            BaseApplication.live_link_output(params.monitoring_directory, output_group)
            print(f"Live link activated!")
            print(
                f"Check your current ANALYST session for results stored in group {output_group.name}."
            )

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
    channel_groups,
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
    Neighbouring anomalies are then grouped and assigned a channel_group label.

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
        "channel_group": [],
    }
    data_uid = list(channels.keys())
    property_groups = [pg for pg in channel_groups.values()]
    group_prop_size = np.r_[[len(grp["properties"]) for grp in channel_groups.values()]]
    for cc, (uid, params) in enumerate(channels.items()):
        if "values" not in list(params.keys()):
            continue

        values = params["values"][line_indices].copy()
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
                anomalies["channel_group"] += [
                    [
                        key
                        for key, channel_group in enumerate(channel_groups.values())
                        if uid in channel_group["properties"]
                    ]
                ]

    if len(anomalies["peak"]) == 0:
        if return_profile:
            return {}, profile
        else:
            return {}

    groups = []

    # Re-cast as numpy arrays
    for key, values in anomalies.items():
        if key == "channel_group":
            continue
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
        # Reject from group if channel gap > 1
        u_gates, u_count = np.unique(anomalies["channel"][near], return_counts=True)
        if len(u_gates) > 1 and np.any((u_gates[1:] - u_gates[:-1]) > 2):
            cutoff = u_gates[np.where((u_gates[1:] - u_gates[:-1]) > 2)[0][0]]
            near = near[anomalies["channel"][near] <= cutoff]  # Remove after cutoff
        # Check for multiple nearest peaks on single channel
        # and keep the nearest
        u_gates, u_count = np.unique(anomalies["channel"][near], return_counts=True)
        for gate in u_gates[np.where(u_count > 1)]:
            mask = np.ones_like(near, dtype="bool")
            sub_ind = anomalies["channel"][near] == gate
            sub_ind[np.where(sub_ind)[0][np.argmin(dist[near][sub_ind])]] = False
            mask[sub_ind] = False
            near = near[mask]

        score = np.zeros(len(channel_groups))
        for ids in near:
            score[anomalies["channel_group"][ids]] += 1

        # Find groups with largest channel overlap
        max_scores = np.where(score == score.max())[0]
        # Keep the group with less properties
        in_group = max_scores[
            np.argmax(score[max_scores] / group_prop_size[max_scores])
        ]
        if score[in_group] < min_channels:
            continue

        channel_group = property_groups[in_group]
        # Remove anomalies not in group
        mask = [
            data_uid[anomalies["channel"][id]] in channel_group["properties"]
            for id in near
        ]
        near = near[mask, ...]
        if len(near) == 0:
            continue
        anomalies["group"][near] = group_id
        gates = anomalies["channel"][near]
        cox = anomalies["peak"][near]
        inflx_dwn = anomalies["inflx_dwn"][near]
        inflx_up = anomalies["inflx_up"][near]
        cox_sort = np.argsort(locs[cox])
        azimuth_near = azimuth[cox]
        dip_direction = azimuth[cox[0]]

        if (
            anomalies["peak_values"][near][cox_sort][0]
            < anomalies["peak_values"][near][cox_sort][-1]
        ):
            dip_direction = (dip_direction + 180) % 360.0

        migration = np.abs(locs[cox[cox_sort[-1]]] - locs[cox[cox_sort[0]]])
        skew = (locs[cox][cox_sort[0]] - locs[inflx_up][cox_sort]) / (
            locs[inflx_dwn][cox_sort] - locs[cox][cox_sort[0]] + 1e-8
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
            for ii, channel in enumerate(channels.values())
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
            "channel_group": channel_group,
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


def groups_from_params_dict(entity: Entity, params_dict: dict):
    """
    Generate a dictionary of groups with associate properties from params.
    """
    count = 0
    channel_groups = {}
    for label, group_params in params_dict.items():
        if group_params["data"] is not None:

            try:
                group_id = group_params["data"]
                if not isinstance(group_id, uuid.UUID):
                    group_id = uuid.UUID(group_id)

            except ValueError:
                group_id = None

            prop_group = [pg for pg in entity.property_groups if pg.uid == group_id]
            if any(prop_group):
                count += 1
                channel_groups[prop_group[0].name] = {
                    "data": prop_group[0].uid,
                    "color": group_params["color"],
                    "label": [count],
                    "properties": prop_group[0].properties,
                }

    return channel_groups


if __name__ == "__main__":
    file = sys.argv[1]
    params = PeakFinderParams(InputFile(file))
    PeakFinder.run(params)
