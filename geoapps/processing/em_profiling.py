import os
import numpy as np
from scipy.spatial import cKDTree
import plotly.graph_objects as go
import plotly.express as px
import time
import matplotlib.pyplot as plt
from geoh5py.workspace import Workspace
from geoh5py.objects import Points, Curve, Surface
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


class EMLineProfiler(BaseApplication):
    """
    Application for the picking of targets along Time-domain EM profiles
    """

    viz_param = '<IParameterList Version="1.0">\n <Colour>4278190335</Colour>\n <Transparency>0</Transparency>\n <Nodesize>9</Nodesize>\n <Nodesymbol>Sphere</Nodesymbol>\n <Scalenodesbydata>false</Scalenodesbydata>\n <Data></Data>\n <Scale>1</Scale>\n <Scalebyabsolutevalue>[NO STRING REPRESENTATION]</Scalebyabsolutevalue>\n <Orientation toggled="true">{\n    "DataGroup": "AzmDip",\n    "ManualWidth": true,\n    "Scale": false,\n    "ScaleLog": false,\n    "Size": 30,\n    "Symbol": "Tablet",\n    "Width": 30\n}\n</Orientation>\n</IParameterList>\n'

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.trigger.description = "Export to GA"
        self.em_system_specs = geophysical_systems.parameters()
        self.system = Dropdown(
            options=[
                key
                for key, specs in self.em_system_specs.items()
                if specs["type"] == "time"
            ],
            description="Time-Domain System:",
        )
        self._groups = base_em_groups()
        self.group_list = Dropdown(description="")
        self.early = np.arange(8, 17).tolist()
        self.middle = np.arange(17, 28).tolist()
        self.late = np.arange(28, 40).tolist()
        self.data_selection = ObjectDataSelection(
            add_groups=True, select_multiple=True, object_types=(Curve,)
        )
        self._objects = self.data_selection.objects
        self._data = self.data_selection.data
        self.data_selection.objects.description = "Survey"
        self.model_selection = ObjectDataSelection(
            h5file=self.h5file, object_types=(Surface,)
        )
        self.model_selection.objects.description = "Surface:"
        self.model_selection.data.description = "Model"
        self.surface_model = None
        self.model_figure = go.Figure()
        self.model_figure.update_layout(
            scene={
                "xaxis_title": "Easting (m)",
                "yaxis_title": "Northing (m)",
                "zaxis_title": "Elevation (m)",
                "yaxis": {"autorange": "reversed"},
                "xaxis": {"autorange": "reversed"},
                "camera": {"eye": dict(x=-1, y=1, z=1.0),},
                "aspectmode": "data",
            },
            width=600,
            height=600,
            autosize=False,
            scene_dragmode="orbit",
        )

        self.marker = {"left": "<", "right": ">"}
        self.lines = LineOptions(
            h5file=self.h5file,
            objects=self.data_selection.objects,
            select_multiple=False,
            find_value=["line", "Line"],
        )
        self.lines.data.description = "Line"

        def survey_selection(_):
            self.survey_selection()

        self.data_selection.objects.observe(survey_selection, names="value")
        self.channels = SelectMultiple(description="Channels")
        self.group_default_early = Text(description="Early", value="9-16")
        self.group_default_middle = Text(description="Middle", value="17-27")
        self.group_default_late = Text(description="Late", value="28-40")

        def reset_default_bounds(_):
            self.reset_default_bounds()

        self.data_channels = {}
        self.data_channel_options = {}

        def get_data(_):
            self.get_data()

        self.data_selection.data.observe(get_data, names="value")
        self.smoothing = IntSlider(
            min=0,
            max=64,
            value=0,
            description="Smoothing",
            continuous_update=False,
            tooltip="Running mean width",
        )
        self.color_maps = Dropdown(
            description="Colormaps",
            options=px.colors.named_colorscales(),
            value="edge",
        )
        self.residual = Checkbox(description="Use residual", value=False)
        self.threshold = FloatSlider(
            value=50,
            min=10,
            max=90,
            step=5,
            continuous_update=False,
            description="Decay threshold (%)",
        )
        self.center = FloatSlider(
            value=0.5,
            min=0,
            max=1.0,
            step=0.001,
            description="Position (%)",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
        )
        self.shift_cox_z = FloatSlider(
            value=0,
            min=0,
            max=1000,
            step=1.0,
            description="Z shift (m)",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
        )
        self.dip_shift = FloatSlider(
            value=0,
            min=-90,
            max=90,
            step=1.0,
            description="Rotate dip (dd)",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
        )
        self.slice_width = FloatSlider(
            value=10.0,
            min=1.0,
            max=500.0,
            step=1.0,
            description="Slice width (m)",
            disabled=False,
            continuous_update=False,
            orientation="vertical",
        )
        self.auto_picker = ToggleButton(description="Pick nearest target", value=False)
        self.pause_plot_refresh = False
        self.reverse_cmap = ToggleButton(description="Flip colormap", value=False)
        self.focus = FloatSlider(
            value=1.0,
            min=0.025,
            max=1.0,
            step=0.005,
            description="Width (%)",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
        )
        self.zoom = VBox([self.center, self.focus])
        self.scale_button = ToggleButtons(
            options=["linear", "symlog",], description="Y-axis scaling",
        )

        def scale_update(_):
            if self.scale_button.value == "symlog":
                scale_panel.children = [
                    self.scale_button,
                    self.scale_value,
                ]
            else:
                scale_panel.children = [self.scale_button]

        self.scale_button.observe(scale_update)
        self.scale_value = FloatLogSlider(
            min=-18,
            max=10,
            step=0.5,
            base=10,
            value=1e-2,
            description="Linear threshold",
            continuous_update=False,
        )
        scale_panel = HBox([self.scale_button])

        def add_group(_):
            self.add_group()

        def channel_setter(caller):
            channel = caller["owner"]
            data_widget = self.data_channel_options[channel.header]
            data_widget.children[0].value = find_value(
                data_widget.children[0].options, [channel.header]
            )

        self.channel_selection = Dropdown(
            description="Time Gate",
            options=self.em_system_specs[self.system.value]["channels"].keys(),
        )

        def system_observer(_):
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

        self.system.observe(system_observer)
        system_observer("")

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
            description="Set System Specs", value=False
        )
        self.system_box_option.observe(system_box_trigger)
        self.system_box = VBox([self.system_box_option])

        self.group_default_early.observe(reset_default_bounds, names="value")
        self.group_default_middle.observe(reset_default_bounds, names="value")
        self.group_default_late.observe(reset_default_bounds, names="value")

        self.group_add = ToggleButton(description="^ Add New Group ^")
        self.group_name = Text(description="Name")
        self.group_color = ColorPicker(
            concise=False, description="Color", value="blue", disabled=False
        )
        self.group_add.observe(add_group)

        def highlight_selection(_):
            self.highlight_selection()

        self.group_list.observe(highlight_selection, names="value")
        self.markers = ToggleButton(description="Show markers")

        self.groups_setter = ToggleButton(description="Set channel groups", value=False)

        def groups_trigger(_):
            self.groups_trigger()

        self.groups_setter.observe(groups_trigger)
        self.groups_widget = VBox([self.groups_setter])

        def trigger_click(_):
            self.trigger_click()

        self.trigger.observe(trigger_click)
        self.x_label = ToggleButtons(
            options=["Distance", "Easting", "Northing"],
            value="Distance",
            description="X-axis label:",
        )

        self.show_model = ToggleButton(
            description="Show model", value=False, button_style="success"
        )

        def plot_data_selection(
            data,
            ind,
            smoothing,
            residual,
            markers,
            scale,
            scale_value,
            center,
            focus,
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
                focus,
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
                "focus": self.focus,
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

        self.show_decay = ToggleButton(description="Show decay", value=False)
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
            focus,
            objects,
            model,
            smoothing,
            slice_width,
            x_label,
            colormap,
            reverse,
            z_shift,
            dip_shift,
        ):
            self.update_line_model()
            self.plot_model_selection(
                ind, center, focus, x_label, colormap, reverse, z_shift, dip_shift
            )

        self.model_section = interactive_output(
            plot_model_selection,
            {
                "ind": self.lines.lines,
                "center": self.center,
                "focus": self.focus,
                "objects": self.model_selection.objects,
                "model": self.model_selection.data,
                "smoothing": self.smoothing,
                "slice_width": self.slice_width,
                "x_label": self.x_label,
                "colormap": self.color_maps,
                "z_shift": self.shift_cox_z,
                "dip_shift": self.dip_shift,
                "reverse": self.reverse_cmap,
            },
        )

        self.model_panel = VBox([self.show_model,])

        def show_model_trigger(_):
            self.show_model_trigger()

        self.show_model.observe(show_model_trigger, names="value")

        self._widget = VBox(
            [
                HBox(
                    [
                        VBox(
                            [
                                self.data_selection.objects,
                                self.data_selection.data,
                                self.system_box,
                                self.groups_widget,
                                HBox([self.auto_picker, self.markers]),
                            ],
                            layout=Layout(width="50%"),
                        ),
                        VBox(
                            [
                                self.lines.widget,
                                self.zoom,
                                self.smoothing,
                                self.residual,
                            ],
                            layout=Layout(width="50%"),
                        ),
                    ]
                ),
                HBox([plotting, self.decay_panel]),
                HBox([self.x_label, self.threshold]),
                scale_panel,
                HBox([self.model_panel], layout=Layout(width="100%")),
                self.trigger_widget,
            ]
        )

        super().__init__(**kwargs)

        self.auto_picker.value = True

    def get_data(self):
        if getattr(self, "survey", None) is not None:
            groups = [p_g.name for p_g in self.survey.property_groups]
            channels = list(self.data_selection.data.value)

            for channel in self.data_selection.data.value:
                if channel in groups:
                    for prop in self.survey.get_property_group(channel).properties:
                        name = self.workspace.get_entity(prop)[0].name
                        if prop not in channels:
                            channels.append(name)

            self.channels.options = channels
            for channel in channels:
                if self.survey.get_data(channel):
                    self.data_channels[channel] = self.survey.get_data(channel)[0]

            # Generate default groups
            self.reset_default_bounds()

            for key, widget in self.data_channel_options.items():
                widget.children[0].options = channels
                widget.children[0].value = find_value(channels, [key])

            self.auto_picker.value = False
            self.auto_picker.value = True

    def survey_selection(self):

        if self.workspace.get_entity(self.data_selection.objects.value):
            self.survey = self.workspace.get_entity(self.data_selection.objects.value)[
                0
            ]

            self.data_selection.data.options = (
                [p_g.name for p_g in self.survey.property_groups]
                + ["^-- Groups --^"]
                + self.survey.get_data_list()
            )

            for aem_system, specs in self.em_system_specs.items():
                if any(
                    [
                        specs["flag"] in channel
                        for channel in self.data_selection.data.options
                    ]
                ):
                    self.system.value = aem_system

            self.line_update()

    def system_box_trigger(self):
        if self.system_box_option.value:
            self.system_box.children = [
                self.system_box_option,
                self.system,
                self.channel_panel,
            ]
        else:
            self.system_box.children = [self.system_box_option]

    def groups_trigger(self):
        if self.groups_setter.value:
            self.groups_widget.children = [
                self.groups_setter,
                self.group_default_early,
                self.group_default_middle,
                self.group_default_late,
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
        # Reset channels
        for group in self.groups.values():
            if len(group["defaults"]) > 0:
                group["channels"] = []
                group["inflx_up"] = []
                group["inflx_dwn"] = []
                group["peaks"] = []
                group["mad_tau"] = []
                group["times"] = []
                group["values"] = []
                group["locations"] = []

        for ind, channel in enumerate(channels):
            for group in self.groups.values():
                if ind in group["gates"]:
                    group["channels"].append(channel)
        self.group_list.options = self.groups.keys()
        self.group_list.value = self.group_list.options[0]

    def add_group(self):
        """
        Add a group to the list of groups
        """
        if self.group_add.value:

            if self.group_name.value not in self.group_list.options:
                self.group_list.options = list(self.group_list.options) + [
                    self.group_name.value
                ]

            self.groups[self.group_name.value] = {
                "color": self.group_color.value,
                "channels": list(self.channels.value),
                "inflx_up": [],
                "inflx_dwn": [],
                "peaks": [],
                "mad_tau": [],
                "times": [],
                "values": [],
                "locations": [],
            }
            self.group_add.value = False

    def trigger_click(self):
        if self.trigger.value:
            for group in self.group_list.value:

                tau = self.groups[group]["mad_tau"]
                dip = self.groups[group]["dip"]
                azimuth = self.groups[group]["azimuth"]
                cox = self.groups[group]["cox"]
                if self.workspace.get_entity(group):
                    points = self.workspace.get_entity(group)[0]
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
                        self.workspace, name=group, vertices=cox.reshape((1, 3))
                    )
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
                    if not os.path.exists(self.live_link_path.value):
                        os.mkdir(self.live_link_path.value)

                    temp_geoh5 = os.path.join(
                        self.live_link_path.value, f"temp{time.time():.3f}.geoh5"
                    )
                    ws_out = Workspace(temp_geoh5)
                    points.copy(parent=ws_out)

            self.trigger.value = False
            self.workspace.finalize()

    def highlight_selection(self):
        """
        Highlight the group choice
        """
        # highlights = []
        # for group in self.group_list.value:
        #     highlights +=
        self.group_color.value = self.groups[self.group_list.value]["color"]
        self.channels.value = self.groups[self.group_list.value]["channels"]

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
        focus,
        groups,
        pick_trigger,
        x_label,
        threshold,
    ):

        self.line_update()

        for group in self.groups.values():
            group["inflx_up"] = []
            group["inflx_dwn"] = []
            group["peaks"] = []
            group["mad_tau"] = []
            group["times"] = []
            group["values"] = []
            group["locations"] = []

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
                (center - focus / 2.0) * self.lines.profile.locations_resampled[-1],
                (center + focus / 2.0) * self.lines.profile.locations_resampled[-1],
            ],
        )

        sub_ind = np.arange(lims[0], lims[1])

        # channels = []
        # for group in self.group_list.value:
        channels = self.groups[self.group_list.value]["channels"]

        if len(channels) == 0:
            channels = self.channels.options

        times = {}
        for channel in self.data_channel_options.values():
            times[channel.children[0].value] = channel.children[1].value

        y_min, y_max = np.inf, -np.inf
        for channel, d in self.data_channels.items():

            if channel not in times.keys():
                continue

            if channel not in channels:
                continue

            if axs is None:
                fig = plt.figure(figsize=(12, 8))
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

                # for ii, group in enumerate(self.group_list.value):
                group = self.group_list.value
                if channel in self.groups[group]["channels"]:
                    self.groups[group]["inflx_up"].append(
                        np.r_[bump_x[inflx_up], bump_v[inflx_up]]
                    )
                    self.groups[group]["peaks"].append(
                        np.r_[bump_x[peak], bump_v[peak]]
                    )
                    self.groups[group]["times"].append(times[channel])
                    self.groups[group]["inflx_dwn"].append(
                        np.r_[bump_x[inflx_dwn], bump_v[inflx_dwn]]
                    )
                    self.groups[group]["locations"].append(bump_x)
                    self.groups[group]["values"].append(bump_v)

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
                        bump_x[:peak],
                        bump_v[:peak],
                        "--",
                        color=self.groups[group]["color"],
                    )
                    # Right
                    axs.plot(
                        bump_x[peak:], bump_v[peak:], color=self.groups[group]["color"],
                    )
                    axs.scatter(
                        self.groups[group]["peaks"][-1][0],
                        self.groups[group]["peaks"][-1][1],
                        s=100,
                        c=self.groups[group]["color"],
                        marker=self.marker[ori],
                    )
                    if ~np.isnan(dip):
                        axs.text(
                            self.groups[group]["peaks"][-1][0],
                            self.groups[group]["peaks"][-1][1],
                            f"{dip:.0f}",
                            va="bottom",
                            ha="center",
                        )
                    axs.scatter(
                        self.groups[group]["inflx_dwn"][-1][0],
                        self.groups[group]["inflx_dwn"][-1][1],
                        s=100,
                        c=self.groups[group]["color"],
                        marker="1",
                    )
                    axs.scatter(
                        self.groups[group]["inflx_up"][-1][0],
                        self.groups[group]["inflx_up"][-1][1],
                        s=100,
                        c=self.groups[group]["color"],
                        marker="2",
                    )

        # for group in self.group_list.value:
        group = self.group_list.value
        if np.any(self.groups[group]["peaks"]):
            peaks = np.vstack(self.groups[group]["peaks"])
            inflx_dwn = np.vstack(self.groups[group]["inflx_dwn"])
            inflx_up = np.vstack(self.groups[group]["inflx_up"])

            ratio = peaks[:, 1] / peaks[0, 1]
            ind = np.where(ratio >= (1 - threshold / 100))[0][-1]

            peaks = np.mean(peaks[: ind + 1, :], axis=0)
            inflx_dwn = np.mean(inflx_dwn[: ind + 1, :], axis=0)
            inflx_up = np.mean(inflx_up[: ind + 1, :], axis=0)

            cox_x = self.lines.profile.interp_x(peaks[0])
            cox_y = self.lines.profile.interp_y(peaks[0])
            cox_z = self.lines.profile.interp_z(peaks[0])
            self.groups[group]["cox"] = np.r_[cox_x, cox_y, cox_z]

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

            self.groups[group]["azimuth"] = azm
            self.groups[group]["dip"] = dip

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

            # for group in self.groups.values():
            if self.groups[group]["peaks"]:
                peaks = np.vstack(self.groups[group]["peaks"])
                ratio = peaks[:, 1] / peaks[0, 1]
                ind = np.where(ratio >= (1 - threshold / 100))[0][-1]
                #                 print(ind)
                axs.plot(
                    peaks[: ind + 1, 0],
                    peaks[: ind + 1, 1],
                    "--",
                    color=self.groups[group]["color"],
                )

            if scale == "symlog":
                plt.yscale("symlog", linthreshy=scale_value)

            x_lims = [
                center_x - focus / 2.0 * self.lines.profile.locations_resampled[-1],
                center_x + focus / 2.0 * self.lines.profile.locations_resampled[-1],
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

            if len(self.groups[group]["peaks"]) == 0:
                return

            peaks = (
                np.vstack(self.groups[group]["peaks"])
                * self.em_system_specs[self.system.value]["normalization"]
            )

            ratio = peaks[:, 1] / peaks[0, 1]
            ind = np.where(ratio >= (1 - self.threshold.value / 100))[0][-1]

            tc = np.hstack(self.groups[group]["times"][: ind + 1])
            vals = np.log(peaks[: ind + 1, 1])

            if tc.shape[0] < 2:
                return
            # Compute linear trend
            A = np.c_[np.ones_like(tc), tc]
            a, c = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, vals))
            d = np.r_[tc.min(), tc.max()]
            vv = d * c + a

            ratio = np.abs((vv[0] - vv[1]) / (d[0] - d[1]))
            #                 angl = np.arctan(ratio**-1.)

            self.groups[group]["mad_tau"] = ratio ** -1.0

            if axs is None:
                plt.figure(figsize=(8, 8))
                axs = plt.subplot()

            axs.plot(
                d,
                np.exp(d * c + a),
                "--",
                linewidth=2,
                color=self.groups[group]["color"],
            )
            axs.text(
                np.mean(d),
                np.exp(np.mean(vv)),
                f"{ratio ** -1.:.2e}",
                color=self.groups[group]["color"],
            )
            #                 plt.yscale('symlog', linthreshy=scale_value)
            #                 axs.set_aspect('equal')
            axs.scatter(
                np.hstack(self.groups[group]["times"]),
                peaks[:, 1],
                color=self.groups[group]["color"],
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
        self, ind, center, focus, x_label, colormap, reverse, z_shift, dip_shift
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

            self.model_figure.data = []
            # Create dip marker
            center_l = center * self.lines.profile.locations_resampled[-1]
            center_x = float(self.lines.profile.interp_x(center_l))
            center_y = float(self.lines.profile.interp_y(center_l))
            center_z = float(self.lines.profile.interp_z(center_l))

            _, ind = tree.query(np.c_[center_x, center_y, center_z])
            self.model_figure.add_trace(
                go.Scatter3d(
                    x=self.lines.model_vertices[ind, 0],
                    y=self.lines.model_vertices[ind, 1],
                    z=self.lines.model_vertices[ind, 2],
                    mode="markers",
                )
            )

            # for group in self.group_list.value:
            group = self.group_list.value
            if not np.any(self.groups[group]["peaks"]):
                return

            _, ind = tree.query(self.groups[group]["cox"].reshape((-1, 3)))
            dip = self.groups[group]["dip"] + dip_shift
            azimuth = self.groups[group]["azimuth"].copy()

            if dip > 90:
                dip = 180 - dip
                azimuth += 180

            vec = rotate_azimuth_dip(azimuth, dip,)
            scaler = 100
            self.model_figure.add_trace(
                go.Cone(
                    x=self.lines.model_vertices[ind, 0],
                    y=self.lines.model_vertices[ind, 1],
                    z=self.lines.model_vertices[ind, 2] - z_shift,
                    u=vec[:, 0] * scaler,
                    v=vec[:, 1] * scaler,
                    w=vec[:, 2] * scaler,
                    showscale=False,
                )
            )
            self.groups[group]["cox"][2] = self.lines.model_vertices[ind, 2]

            simplices = self.lines.model_cells.reshape((-1, 3))
            self.model_figure.add_trace(
                go.Mesh3d(
                    x=self.lines.model_vertices[:, 0],
                    y=self.lines.model_vertices[:, 1],
                    z=self.lines.model_vertices[:, 2],
                    intensity=np.log10(self.lines.model_values),
                    i=simplices[:, 0],
                    j=simplices[:, 1],
                    k=simplices[:, 2],
                    colorscale=colormap,
                )
            )

            if azimuth > 180:
                azm = azimuth + 90
            else:
                azm = azimuth - 90
            vec = rotate_azimuth_dip(azm, 0)
            self.model_figure.update_layout(
                scene={
                    "camera": {
                        "center": dict(x=0, y=0, z=0,),
                        "eye": dict(x=vec[0, 0] * 1.75, y=vec[0, 1] * 1.75, z=0.0,),
                    }
                }
            )

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

        if (
            getattr(self.lines.profile, "line_id", None) is None
            or self.lines.profile.line_id != self.lines.lines.value
        ):

            lims = [
                (self.center.value - self.focus.value / 2.0)
                * self.lines.profile.locations_resampled[-1],
                (self.center.value + self.focus.value / 2.0)
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

        if self.show_model.value:
            self.show_model.description = "Hide model"
        else:
            self.show_model.description = "Show model"

    def reset_default_bounds(self):

        try:
            first, last = np.asarray(
                self.group_default_early.value.split("-"), dtype="int"
            )
            self.early = np.arange(first, last + 1).tolist()
        except ValueError:
            return

        try:
            first, last = np.asarray(
                self.group_default_middle.value.split("-"), dtype="int"
            )
            self.middle = np.arange(first, last + 1).tolist()
        except ValueError:
            return

        try:
            first, last = np.asarray(
                self.group_default_late.value.split("-"), dtype="int"
            )
            self.last = np.arange(first, last + 1).tolist()
        except ValueError:
            return

        for group in self.groups.values():
            gates = []
            if len(group["defaults"]) > 0:
                for default in group["defaults"]:
                    gates += getattr(self, default)
                group["gates"] = gates

        self.set_default_groups(self.channels.options)

    def show_model_trigger(self):
        """
        Add the model widget
        """
        if self.show_model.value:
            self.model_panel.children = [
                self.show_model,
                HBox(
                    [self.model_selection.objects, VBox([self.model_selection.data]),]
                ),
                HBox(
                    [
                        self.model_section,
                        VBox(
                            [
                                self.color_maps,
                                self.reverse_cmap,
                                self.shift_cox_z,
                                self.dip_shift,
                            ]
                        ),
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
            self.decay_panel.children = [
                self.show_decay,
                self.decay,
            ]
            self.show_decay.description = "Hide decay curve"
        else:
            self.decay_panel.children = [self.show_decay]
            self.show_decay.description = "Show decay curve"

    @property
    def data(self):
        return self._data

    @property
    def groups(self):
        return self._groups

    @property
    def objects(self):
        return self._objects

    @property
    def widget(self):
        return self._widget


def base_em_groups():
    return {
        "early": {
            "color": "blue",
            "channels": [],
            "gates": [],
            "defaults": ["early"],
            "inflx_up": [],
            "inflx_dwn": [],
            "peaks": [],
            "times": [],
            "values": [],
            "locations": [],
        },
        "early + middle": {
            "color": "cyan",
            "channels": [],
            "gates": [],
            "defaults": ["early", "middle"],
            "inflx_up": [],
            "inflx_dwn": [],
            "peaks": [],
            "times": [],
            "values": [],
            "locations": [],
        },
        "middle": {
            "color": "green",
            "channels": [],
            "gates": [],
            "defaults": ["middle"],
            "inflx_up": [],
            "inflx_dwn": [],
            "peaks": [],
            "times": [],
            "values": [],
            "locations": [],
        },
        "early + middle + late": {
            "color": "orange",
            "channels": [],
            "gates": [],
            "defaults": ["early", "middle", "late"],
            "inflx_up": [],
            "inflx_dwn": [],
            "peaks": [],
            "times": [],
            "values": [],
            "locations": [],
        },
        "middle + late": {
            "color": "yellow",
            "channels": [],
            "gates": [],
            "defaults": ["middle", "late"],
            "inflx_up": [],
            "inflx_dwn": [],
            "peaks": [],
            "times": [],
            "values": [],
            "locations": [],
        },
        "late": {
            "color": "red",
            "channels": [],
            "gates": [],
            "defaults": ["late"],
            "inflx_up": [],
            "inflx_dwn": [],
            "peaks": [],
            "times": [],
            "values": [],
            "locations": [],
        },
    }
