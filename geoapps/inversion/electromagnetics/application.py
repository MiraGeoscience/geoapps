# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024-2025 Mira Geoscience Ltd.                                '
#                                                                              '
#  This file is part of geoapps.                                               '
#                                                                              '
#  geoapps is distributed under the terms and conditions of the MIT License    '
#  (see LICENSE file at the root of this source code package).                 '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from __future__ import annotations

import json
import multiprocessing
import os
import uuid
from pathlib import Path
from time import time
from uuid import UUID

import numpy as np
from geoh5py.data import Data
from geoh5py.groups import ContainerGroup
from geoh5py.objects import (
    AirborneTEMReceivers,
    BlockModel,
    Curve,
    Octree,
    Points,
    Surface,
)
from geoh5py.shared.utils import fetch_active_workspace, is_uuid
from geoh5py.workspace import Workspace

from geoapps import assets_path
from geoapps.base.application import BaseApplication
from geoapps.base.plot import PlotSelection2D
from geoapps.base.selection import LineOptions, ObjectDataSelection, TopographyOptions
from geoapps.inversion.components.preprocessing import preprocess_data
from geoapps.shared_utils.utils import WindowOptions
from geoapps.utils import geophysical_systems, warn_module_not_found
from geoapps.utils.list import find_value
from geoapps.utils.string import string_2_list


with warn_module_not_found():
    from matplotlib import pyplot as plt

with warn_module_not_found():
    import ipywidgets as widgets
    from ipywidgets.widgets import (
        Button,
        Checkbox,
        Dropdown,
        FloatText,
        HBox,
        IntText,
        Label,
        Layout,
        Text,
        VBox,
    )


class ChannelOptions:
    """
    Options for data channels
    """

    def __init__(self, key, description, **kwargs):
        self._active = Checkbox(
            value=False,
            indent=True,
            description="Active",
        )
        self._label = widgets.Text(description=description)
        self._channel_selection = Dropdown(description="Channel <=> Data:")
        self._channel_selection.header = key
        self._uncertainties = widgets.Text(description="Error (%, floor)")
        self._offsets = widgets.Text(description="Offsets (x,y,z)")
        self._main = VBox(
            [
                self._active,
                self._label,
                self._channel_selection,
                self._uncertainties,
                self._offsets,
            ]
        )

        for obj in self.__dict__:
            if hasattr(getattr(self, obj), "style"):
                getattr(self, obj).style = {"description_width": "initial"}

        for key, value in kwargs.items():
            if hasattr(self, "_" + key):
                try:
                    getattr(self, key).value = value
                except AttributeError:
                    pass

    @property
    def active(self):
        return self._active

    @property
    def label(self):
        return self._label

    @property
    def channel_selection(self):
        return self._channel_selection

    @property
    def uncertainties(self):
        return self._uncertainties

    @property
    def offsets(self):
        return self._offsets

    @property
    def main(self):
        return self._main


class SensorOptions(ObjectDataSelection):
    """
    Define the receiver spatial parameters
    """

    _options = None

    def __init__(self, **kwargs):
        self.defaults.update(**kwargs)
        self._offset = Text(description="(dx, dy, dz) (+ve up)", value="0, 0, 0")
        self._constant = FloatText(
            description="Constant elevation (m)",
        )
        if "offset" in self.defaults:
            self._offset.value = self.defaults["offset"]

        self.option_list = {
            "sensor location + (dx, dy, dz)": self.offset,
            "topo + radar + (dx, dy, dz)": VBox(
                [
                    self.offset,
                ]
            ),
        }
        self.options.observe(self.update_options, names="value")

        super().__init__(**self.defaults)

        self.option_list["topo + radar + (dx, dy, dz)"].children = [
            self.offset,
            self.data,
        ]
        self.data.description = "Radar (Optional):"
        self.data.style = {"description_width": "initial"}

    @property
    def main(self):
        if self._main is None:
            self._main = VBox([self.options, self.option_list[self.options.value]])

        return self._main

    @property
    def offset(self):
        return self._offset

    @property
    def options(self):
        if getattr(self, "_options", None) is None:
            self._options = widgets.RadioButtons(
                options=[
                    "sensor location + (dx, dy, dz)",
                    "topo + radar + (dx, dy, dz)",
                ],
                description="Define by:",
            )
        return self._options

    def update_options(self, _):
        self.main.children = [
            self.options,
            self.option_list[self.options.value],
        ]


class MeshOctreeOptions:
    """
    Widget used for the creation of an octree meshes
    """

    def __init__(self, **kwargs):
        self._core_cell_size = widgets.Text(
            value="25, 25, 25",
            description="Smallest cells",
        )
        self._octree_levels_topo = widgets.Text(
            value="0, 0, 0, 2",
            description="Layers below topo",
        )
        self._octree_levels_obs = widgets.Text(
            value="5, 5, 5, 5",
            description="Layers below data",
        )
        self._depth_core = FloatText(
            value=500,
            description="Minimum depth (m)",
        )
        self._padding_distance = widgets.Text(
            value="0, 0, 0, 0, 0, 0",
            description="Padding [W,E,N,S,D,U] (m)",
        )
        self._max_distance = FloatText(
            value=1000,
            description="Max triangulation length",
        )
        self._main = widgets.VBox(
            [
                Label("octree Mesh"),
                self._core_cell_size,
                self._octree_levels_topo,
                self._octree_levels_obs,
                self._depth_core,
                self._padding_distance,
                self._max_distance,
            ]
        )

        for obj in self.__dict__:
            if hasattr(getattr(self, obj), "style"):
                getattr(self, obj).style = {"description_width": "initial"}

        for key, value in kwargs.items():
            if hasattr(self, "_" + key):
                try:
                    getattr(self, key).value = value
                except AttributeError:
                    pass

    @property
    def core_cell_size(self):
        return self._core_cell_size

    @property
    def depth_core(self):
        return self._depth_core

    @property
    def max_distance(self):
        return self._max_distance

    @property
    def octree_levels_obs(self):
        return self._octree_levels_obs

    @property
    def octree_levels_topo(self):
        return self._octree_levels_topo

    @property
    def padding_distance(self):
        return self._padding_distance

    @property
    def main(self):
        return self._main


class Mesh1DOptions:
    """
    Widget used for the creation of a 1D mesh
    """

    def __init__(self, **kwargs):
        self._hz_expansion = FloatText(
            value=1.05,
            description="Expansion factor:",
        )
        self._hz_min = FloatText(
            value=10.0,
            description="Smallest cell (m):",
        )
        self._n_cells = FloatText(
            value=25.0,
            description="Number of cells:",
        )
        self.cell_count = Label(f"Max depth: {self.count_cells():.2f} m")
        self.n_cells.observe(self.update_hz_count)
        self.hz_expansion.observe(self.update_hz_count)
        self.hz_min.observe(self.update_hz_count)
        self._main = VBox(
            [
                Label("1D Mesh"),
                self.hz_min,
                self.hz_expansion,
                self.n_cells,
                self.cell_count,
            ]
        )

        for obj in self.__dict__:
            if hasattr(getattr(self, obj), "style"):
                getattr(self, obj).style = {"description_width": "initial"}

        for key, value in kwargs.items():
            if hasattr(self, "_" + key):
                try:
                    getattr(self, key).value = value
                except AttributeError:
                    pass

    def count_cells(self):
        return (
            self.hz_min.value * self.hz_expansion.value ** np.arange(self.n_cells.value)
        ).sum()

    def update_hz_count(self, _):
        self.cell_count.value = f"Max depth: {self.count_cells():.2f} m"

    @property
    def hz_expansion(self):
        """"""
        return self._hz_expansion

    @property
    def hz_min(self):
        """"""
        return self._hz_min

    @property
    def n_cells(self):
        """"""
        return self._n_cells

    @property
    def main(self):
        return self._main


class ModelOptions(ObjectDataSelection):
    """
    Widgets for the selection of model options
    """

    defaults = {"objects": None, "data": None}

    def __init__(self, **kwargs):
        self._units = "Units"
        self._object_types = (BlockModel, Octree, Surface, Curve, Points)
        self._options = widgets.RadioButtons(
            options=["Model", "Value"],
            value="Value",
            disabled=False,
        )
        self._options.observe(self.update_panel, names="value")
        self.objects.description = "Object"
        self.data.description = "Values"
        self._value = FloatText(description=self.units)
        self._description = Label()

        super().__init__(**kwargs)

        self.selection_widget = self.main
        self._main = widgets.VBox(
            [self._description, widgets.VBox([self._options, self._value])]
        )

    def update_panel(self, _):
        if self._options.value == "Model":
            self._main.children[1].children = [self._options, self.selection_widget]
            self._main.children[1].children[1].layout.visibility = "visible"
        elif self._options.value == "Value":
            self._main.children[1].children = [self._options, self._value]
            self._main.children[1].children[1].layout.visibility = "visible"
        else:
            self._main.children[1].children[1].layout.visibility = "hidden"

    @property
    def description(self):
        return self._description

    @property
    def options(self):
        return self._options

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, value):
        self._units = value
        self._value.description = value

    @property
    def value(self):
        return self._value


class InversionOptions(BaseApplication):
    """
    Collection of widgets controlling the inversion parameters
    """

    def __init__(self, **kwargs):
        self.defaults.update(**kwargs)
        self._chi_factor = FloatText(
            value=1, description="Target misfit", disabled=False
        )
        self._uncert_mode = widgets.RadioButtons(
            options=[
                "Estimated (%|data| + background)",
                "User input (%|data| + floor)",
            ],
            value="User input (%|data| + floor)",
            disabled=False,
        )
        self._lower_bound = widgets.Text(
            value=None,
            description="Lower bound value",
        )
        self._upper_bound = widgets.Text(
            value=None,
            description="Upper bound value",
        )
        self._ignore_values = widgets.Text(
            value="<0",
            description="Data (i.e. <0 = no negatives)",
        )
        self._max_iterations = IntText(value=10, description="Max beta Iterations")
        self._max_cg_iterations = IntText(value=30, description="Max CG Iterations")
        self._tol_cg = FloatText(value=1e-3, description="CG Tolerance")
        self._n_cpu = IntText(
            value=int(multiprocessing.cpu_count() / 2), description="Max CPUs"
        )
        self._max_ram = FloatText(value=2.0, description="Max RAM (Gb)")
        self._beta_start_options = widgets.RadioButtons(
            options=["value", "ratio"],
            value="ratio",
            description="Starting tradeoff (beta):",
        )
        self._beta_start = FloatText(value=1e2, description="phi_d/phi_m")
        self._beta_start_options.observe(self.initial_beta_change)
        self._beta_start_panel = HBox([self._beta_start_options, self._beta_start])
        self._optimization = VBox(
            [
                self._max_iterations,
                self._chi_factor,
                self._beta_start_panel,
                self._max_cg_iterations,
                self._tol_cg,
                self._n_cpu,
                self._max_ram,
            ]
        )
        self._starting_model = ModelOptions(**kwargs)
        self._susceptibility_model = ModelOptions(
            units="SI", description="Background susceptibility", **kwargs
        )
        self._susceptibility_model.options.options = ["None", "Model", "Value"]
        self._reference_model = ModelOptions(**kwargs)
        self._reference_model.options.options = [
            "None",
            "Best-fitting halfspace",
            "Model",
            "Value",
        ]
        self.reference_model.options.observe(self.update_ref)
        self._alphas = widgets.Text(
            value="1, 1, 1, 1",
            description="Scaling alpha_(s, x, y, z)",
        )
        self._norms = widgets.Text(
            value="2, 2, 2, 2",
            description="Norms p_(s, x, y, z)",
            continuous_update=False,
        )
        self.bound_panel = VBox([self._upper_bound, self._lower_bound])
        self._mesh = MeshOctreeOptions()

        super().__init__(**self.defaults)

        self.inversion_options = {
            "uncertainties": self._uncert_mode,
            "starting model": self._starting_model.main,
            "background susceptibility": self._susceptibility_model.main,
            "regularization": VBox(
                [self._reference_model.main, self._alphas, self._norms]
            ),
            "upper-lower bounds": self.bound_panel,
            "mesh": self.mesh.main,
            "ignore values": VBox([self._ignore_values]),
            "optimization": self._optimization,
        }
        self.option_choices = widgets.Dropdown(
            options=list(self.inversion_options),
            value=list(self.inversion_options)[0],
            disabled=False,
        )
        self.option_choices.observe(self.inversion_option_change, names="value")
        self._main = widgets.VBox(
            [
                HBox([widgets.Label("inversion Options")]),
                HBox(
                    [
                        self.option_choices,
                        self.inversion_options[self.option_choices.value],
                    ],
                ),
            ],
            layout=Layout(width="100%"),
        )

    def inversion_option_change(self, _):
        self._main.children[1].children = [
            self.option_choices,
            self.inversion_options[self.option_choices.value],
        ]

    def initial_beta_change(self, _):
        if self._beta_start_options.value == "ratio":
            self._beta_start.description = "phi_d/phi_m"
        else:
            self._beta_start.description = ""

    @property
    def alphas(self):
        return self._alphas

    @property
    def beta_start(self):
        return self._beta_start

    @property
    def beta_start_options(self):
        return self._beta_start_options

    @property
    def chi_factor(self):
        return self._chi_factor

    @property
    def ignore_values(self):
        return self._ignore_values

    @property
    def lower_bound(self):
        return self._lower_bound

    @property
    def upper_bound(self):
        return self._upper_bound

    @property
    def max_iterations(self):
        return self._max_iterations

    @property
    def max_cg_iterations(self):
        return self._max_cg_iterations

    @property
    def n_cpu(self):
        """
        ipywidgets.IntText()
        """
        return self._n_cpu

    @property
    def max_ram(self):
        """
        ipywidgets.IntText()
        """
        return self._max_ram

    @property
    def mesh(self):
        return self._mesh

    @property
    def norms(self):
        return self._norms

    @property
    def optimization(self):
        return self._optimization

    @property
    def reference_model(self):
        return self._reference_model

    @property
    def starting_model(self):
        return self._starting_model

    @property
    def susceptibility_model(self):
        return self._susceptibility_model

    @property
    def tol_cg(self):
        return self._tol_cg

    @property
    def uncert_mode(self):
        return self._uncert_mode

    def update_ref(self, _):
        alphas = string_2_list(self.alphas.value)
        if self.reference_model.options.value == "None":
            alphas[0] = 0.0
        else:
            alphas[0] = 1.0
        self.alphas.value = ", ".join(list(map(str, alphas)))

    def update_workspace(self, workspace):
        self.starting_model.workspace = workspace
        self.reference_model.workspace = workspace
        self.susceptibility_model.workspace = workspace


def get_inversion_output(h5file, group_name):
    """
    Recover an inversion iterations from a ContainerGroup comments.
    """
    with Workspace(h5file) as workspace:
        out = {"time": [], "iteration": [], "phi_d": [], "phi_m": [], "beta": []}

        if workspace.get_entity(group_name):
            group = workspace.get_entity(group_name)[0]

            for comment in group.comments.values:
                if "Iteration" in comment["Author"]:
                    out["iteration"] += [np.int(comment["Author"].split("_")[1])]
                    out["time"] += [comment["Date"]]
                    values = json.loads(comment["Text"])
                    out["phi_d"] += [float(values["phi_d"])]
                    out["phi_m"] += [float(values["phi_m"])]
                    out["beta"] += [float(values["beta"])]

            if len(out["iteration"]) > 0:
                out["iteration"] = np.hstack(out["iteration"])
                ind = np.argsort(out["iteration"])
                out["iteration"] = out["iteration"][ind]
                out["phi_d"] = np.hstack(out["phi_d"])[ind]
                out["phi_m"] = np.hstack(out["phi_m"])[ind]
                out["time"] = np.hstack(out["time"])[ind]

    return out


def plot_convergence_curve(h5file):
    """"""
    with Workspace(h5file, mode="r") as workspace:
        names = [
            group.name
            for group in workspace.groups
            if isinstance(group, ContainerGroup)
        ]
    objects = widgets.Dropdown(
        options=names,
        value=names[0],
        description="inversion Group:",
    )

    def plot_curve(objects):
        with Workspace(h5file, mode="r") as workspace:
            inversion = workspace.get_entity(objects)[0]
            result = None
            if getattr(inversion, "comments", None) is not None:
                if inversion.comments.values is not None:
                    result = get_inversion_output(workspace.h5file, objects)
                    iterations = result["iteration"]
                    phi_d = result["phi_d"]
                    phi_m = result["phi_m"]

                    ax1 = plt.subplot()
                    ax2 = ax1.twinx()
                    ax1.plot(iterations, phi_d, linewidth=3, c="k")
                    ax1.set_xlabel("Iterations")
                    ax1.set_ylabel(r"$\phi_d$", size=16)
                    ax2.plot(iterations, phi_m, linewidth=3, c="r")
                    ax2.set_ylabel(r"$\phi_m$", size=16)

        return result

    interactive_plot = widgets.interactive(plot_curve, objects=objects)

    return interactive_plot


def inversion_defaults():
    """
    Get defaults for EM1D inversion
    """
    defaults = {
        "units": {"EM1D": "S/m"},
        "property": {
            "EM1D": "conductivity",
        },
        "reference_value": {"EM1D": 1e-3},
        "starting_value": {
            "EM1D": 1e-3,
        },
    }

    return defaults


app_initializer = {
    "geoh5": str(assets_path() / "FlinFlon.geoh5"),
    "objects": UUID("{656acd40-25de-4865-814c-cb700f6ee51a}"),
    "data": UUID("{2d165431-63bd-4e07-9db8-5b44acf8c9bf}"),
    "resolution": 50,
    "window_width": 1500,
    "window_height": 1500,
    "ga_group_name": "EM1DInversion_",
    "inversion_parameters": {"max_iterations": 25},
    "topography": {
        "options": "Object",
        "objects": UUID("{ab3c2083-6ea8-4d31-9230-7aad3ec09525}"),
        "data": UUID("{a603a762-f6cb-4b21-afda-3160e725bf7d}"),
    },
    "lines": {
        "data": UUID("{b6927372-22d2-4ca1-9c23-372411ddc772}"),
        "lines": [8, 9, 10, 11, 12],
    },
    "sensor": {"offset": "0, 0, 0", "options": "sensor location + (dx, dy, dz)"},
}


class InversionApp(PlotSelection2D):
    _select_multiple = False
    _add_groups = "only"
    _sensor = None
    _lines = None
    _topography = None
    _inversion_parameters = None

    def __init__(self, plot_result=True, **kwargs):
        self.defaults.update(**app_initializer)
        self.defaults.update(**kwargs)
        self._uncertainties = Dropdown(
            description="Uncertainties: ",
        )
        self.em_system_specs = geophysical_systems.parameters()
        self._data_count = (Label("Data Count: 0"),)
        self._forward_only = Checkbox(
            value=False,
            description="Forward only",
        )
        self._starting_channel = (IntText(value=None, description="Starting Channel"),)
        self._system = Dropdown(
            options=["Airborne TEM Survey"] + list(self.em_system_specs),
            description="Survey Type: ",
        )
        self._write = Button(
            description="Write input",
            button_style="warning",
            icon="check",
        )
        self.data_channel_choices = widgets.Dropdown()
        self.data_channel_panel = widgets.VBox(
            [self.data_channel_choices], layout=Layout(width="50%")
        )
        self.trigger.observe(self.trigger_click, names="value")
        self.write.on_click(self.write_trigger)
        self.system.observe(self.system_observer, names="value")
        self.mesh_1D = Mesh1DOptions(**self.defaults)
        self.objects.observe(self.object_observer, names="value")
        self.data.observe(self.update_component_panel, names="value")
        self.data_channel_choices.observe(
            self.data_channel_choices_observer, names="value"
        )
        self.plotting_data = None
        super().__init__(plot_result=plot_result, **self.defaults)

        if "lines" in app_initializer:
            self.lines.__populate__(**self.defaults["lines"])

        self.output_panel = VBox([self.export_directory, self.write, self.trigger])
        self.inversion_parameters.update_ref(None)

    @property
    def data_count(self):
        """"""
        return self._data_count

    @property
    def forward_only(self):
        """"""
        return self._forward_only

    @property
    def inversion_parameters(self):
        if getattr(self, "_inversion_parameters", None) is None:
            self._inversion_parameters = InversionOptions(
                workspace=self._workspace, **self.defaults["inversion_parameters"]
            )

        return self._inversion_parameters

    @property
    def lines(self):
        if getattr(self, "_lines", None) is None:
            self._lines = LineOptions(workspace=self._workspace, objects=self._objects)
            self._lines.lines.observe(self.update_selection, names="value")
            self._lines.data.description = "Field"
            self._lines.lines.description = "ID:"

        return self._lines

    @property
    def main(self):
        if getattr(self, "_main", None) is None:
            self._main = VBox(
                [
                    self.project_panel,
                    HBox(
                        [
                            VBox([self.objects, self.data, self._uncertainties]),
                            VBox(
                                [
                                    VBox(
                                        [
                                            Label("Geophysical System"),
                                            self.system,
                                        ]
                                    ),
                                ],
                            ),
                        ],
                        layout=Layout(border="solid"),
                    ),
                    VBox(
                        [
                            Label("Input Data"),
                            HBox(
                                [
                                    self.data_channel_panel,
                                    self.window_selection,
                                ],
                            ),
                            Label("Line Selection"),
                            self.lines.main,
                        ],
                        layout=Layout(border="solid"),
                    ),
                    HBox(
                        [
                            VBox(
                                [
                                    Label("Topography"),
                                    self.topography.main,
                                ]
                            ),
                            VBox(
                                [
                                    Label("Sensor Location"),
                                    self.sensor.main,
                                ]
                            ),
                        ],
                        layout=Layout(border="solid"),
                    ),
                    VBox(
                        [
                            Label("inversion Parameters"),
                            self.forward_only,
                            self.inversion_parameters.main,
                        ],
                        layout=Layout(border="solid"),
                    ),
                    VBox(
                        [
                            Label("Output"),
                            self.ga_group_name,
                            self.output_panel,
                        ],
                        layout=Layout(border="solid"),
                    ),
                ]
            )
        return self._main

    @property
    def sensor(self):
        if getattr(self, "_sensor", None) is None:
            self._sensor = SensorOptions(
                workspace=self._workspace,
                objects=self._objects,
                **self.defaults["sensor"],
            )
        return self._sensor

    @property
    def starting_channel(self):
        """"""
        return self._starting_channel

    @property
    def system(self):
        """"""
        return self._system

    @property
    def topography(self):
        if getattr(self, "_topography", None) is None:
            self._topography = TopographyOptions(
                workspace=self._workspace, **self.defaults["topography"]
            )

        return self._topography

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
        assert isinstance(workspace, Workspace), (
            f"Workspace must be of class {Workspace}"
        )
        self.base_workspace_changes(workspace)

        # Refresh the list of objects
        self.update_objects_list()

        # Check for startup
        if getattr(self, "_inversion_parameters", None) is not None:
            self.inversion_parameters.update_workspace(workspace)
            self.lines.workspace = workspace
            self.sensor.workspace = workspace
            self.topography.workspace = workspace

    @property
    def write(self):
        """"""
        return self._write

    def trigger_click(self, _):
        """"""
        os.system(
            "start cmd.exe @cmd /k "
            + 'python -m geoapps.inversion.electromagnetics.driver "'
            + self.trigger.file_name
        )
        self.trigger.button_style = ""

    def system_observer(self, _):
        """
        Change the application on change of system
        """

        self.sensor.offset.value = ", ".join(
            [
                str(offset)
                for offset in self.em_system_specs[self.system.value]["bird_offset"]
            ]
        )

        # if start_channel is None:
        self.inversion_parameters.reference_model.options.options = [
            "Best-fitting halfspace",
            "Model",
            "Value",
        ]
        self.inversion_parameters.reference_model.options.value = (
            "Best-fitting halfspace"
        )
        self.inversion_parameters.lower_bound.value = "1e-5"
        self.inversion_parameters.upper_bound.value = "10"
        self.inversion_parameters.ignore_values.value = "<0"

        # Switch mesh options
        self.inversion_parameters._mesh = (  # pylint: disable=protected-access
            self.mesh_1D
        )
        self.inversion_parameters.inversion_options["mesh"] = self.mesh_1D.main
        flag = "EM1D"

        self.inversion_parameters.reference_model.value.description = (
            inversion_defaults()["units"][flag]
        )
        self.inversion_parameters.reference_model.value.value = inversion_defaults()[
            "reference_value"
        ][flag]
        self.inversion_parameters.reference_model.description.value = (
            "Reference " + inversion_defaults()["property"][flag]
        )
        self.inversion_parameters.starting_model.value.description = (
            inversion_defaults()["units"][flag]
        )
        self.inversion_parameters.starting_model.value.value = inversion_defaults()[
            "starting_value"
        ][flag]
        self.inversion_parameters.starting_model.description.value = (
            "Starting " + inversion_defaults()["property"][flag]
        )

        def channel_setter(caller):
            channel = caller["owner"]
            data_widget = self.data_channel_choices.data_channel_options[channel.header]

            entity = self.workspace.get_entity(self.objects.value)[0]
            if entity is None:
                return

            if channel.value is None or channel.value not in [
                child.uid for child in entity.children
            ]:
                data_widget.children[0].value = False
            else:
                data_widget.children[0].value = True
            # Trigger plot update
            if self.data_channel_choices.value == channel.header:
                self.plotting_data = channel.value
                self.refresh.value = False
                self.refresh.value = True

        data_channel_options = {}
        if self.system.value != "Airborne TEM Survey":
            if self.em_system_specs[self.system.value]["type"] == "time":
                labels = ["Time (s)"] * len(
                    self.em_system_specs[self.system.value]["channels"]
                )
            else:
                labels = ["Frequency (Hz)"] * len(
                    self.em_system_specs[self.system.value]["channels"]
                )

            start_channel = self.em_system_specs[self.system.value][
                "channel_start_index"
            ]
            tx_offsets = self.em_system_specs[self.system.value]["tx_offsets"]
            uncertainties = self.em_system_specs[self.system.value]["uncertainty"]
            system_specs = {}
            for key, time_gate in self.em_system_specs[self.system.value][
                "channels"
            ].items():
                system_specs[key] = f"{time_gate:.5e}"

            for ind, (key, channel) in enumerate(system_specs.items()):
                if ind + 1 < start_channel:
                    continue

                if len(tx_offsets) > 1:
                    offsets = tx_offsets[ind]
                else:
                    offsets = tx_offsets[0]

                try:
                    channel_options = ChannelOptions(
                        key,
                        labels[ind],
                        uncertainties=", ".join(
                            [str(uncert) for uncert in uncertainties[ind][:2]]
                        ),
                        offsets=", ".join([str(offset) for offset in offsets]),
                    )

                except IndexError:
                    continue
                channel_options.channel_selection.observe(channel_setter, names="value")
                data_channel_options[key] = channel_options.main
                data_channel_options[key].children[1].value = channel

        if len(data_channel_options) > 0:
            self.data_channel_choices.options = list(data_channel_options)
            self.data_channel_choices.value = list(data_channel_options)[0]
            self.data_channel_choices.data_channel_options = data_channel_options
            self.data_channel_panel.children = [
                self.data_channel_choices,
                data_channel_options[self.data_channel_choices.value],
            ]
        else:
            self.data_channel_choices.options = []
            self.data_channel_choices.data_channel_options = None
            self.data_channel_panel.children = [self.data_channel_choices]

        self.update_component_panel(None)

        if self.em_system_specs[self.system.value]["type"] == "frequency":
            self.inversion_parameters.option_choices.options = list(
                self.inversion_parameters.inversion_options
            )
        else:
            self.inversion_parameters.option_choices.options = [
                key
                for key in self.inversion_parameters.inversion_options
                if key != "background susceptibility"
            ]

        self.write.button_style = "warning"
        self.trigger.button_style = "danger"

    def object_observer(self, _):
        self.resolution.indices = None
        entity = self._workspace.get_entity(self.objects.value)[0]
        if entity is None:
            return

        self.update_data_list(None)
        self.sensor.update_data_list(None)
        self.lines.update_data_list(None)
        self.lines.update_line_list(None)

        if isinstance(entity, AirborneTEMReceivers):
            self.system.value = "Airborne TEM Survey"
        else:
            for aem_system, specs in self.em_system_specs.items():
                if aem_system == "Airborne TEM Survey":
                    continue

                if any([specs["flag"] in data.name for data in entity.children]):
                    self.system.value = aem_system
                    break

        self.system_observer(None)

        if getattr(self.data_channel_choices, "data_channel_options", None) is not None:
            for (
                key,
                data_widget,
            ) in self.data_channel_choices.data_channel_options.items():
                data_widget.children[2].options = self.data.options
                value = find_value(self.data.options, [key])
                data_widget.children[2].value = value

        self.write.button_style = "warning"
        self.trigger.button_style = "danger"

    def update_component_panel(self, _):
        obj, data_list = self.get_selected_entities()

        if obj is None:
            return

        options = [
            [data.name, data.uid] for data in data_list if isinstance(data, Data)
        ]

        self._uncertainties.options = self.data.options
        if getattr(self.data_channel_choices, "data_channel_options", None) is not None:
            for (
                key,
                data_widget,
            ) in self.data_channel_choices.data_channel_options.items():
                data_widget.children[2].options = options
                data_widget.children[2].value = find_value(options, [key])

    def data_channel_choices_observer(self, _):
        if getattr(
            self.data_channel_choices, "data_channel_options", None
        ) is not None and self.data_channel_choices.value in (
            self.data_channel_choices.data_channel_options
        ):
            data_widget = self.data_channel_choices.data_channel_options[
                self.data_channel_choices.value
            ]
            self.data_channel_panel.children = [self.data_channel_choices, data_widget]

            if (
                self.workspace.get_entity(self.objects.value)[0] is not None
                and data_widget.children[2].value is None
            ):
                _, data_list = self.get_selected_entities()
                options = [[data.name, data.uid] for data in data_list]
                data_widget.children[2].value = find_value(
                    options, [self.data_channel_choices.value]
                )

            self.plotting_data = data_widget.children[2].value
            self.refresh.value = False
            self.refresh.value = True

        self.write.button_style = "warning"
        self.trigger.button_style = "danger"

    def update_selection(self, _):
        self.highlight_selection = {self.lines.data.value: self.lines.lines.value}
        self.refresh.value = False
        self.refresh.value = True

    def write_trigger(self, _):
        input_dict = {
            "out_group": self.ga_group_name.value,
            "mesh": "Mesh",
            "system": self.system.value,
            "lines": {str(self.lines.data.value): self.lines.lines.value},
            "mesh 1D": [
                self.mesh_1D.hz_min.value,
                self.mesh_1D.hz_expansion.value,
                self.mesh_1D.n_cells.value,
            ],
            "uncertainty_mode": self.inversion_parameters.uncert_mode.value,
            "chi_factor": self.inversion_parameters.chi_factor.value,
            "max_iterations": self.inversion_parameters.max_iterations.value,
            "max_cg_iterations": self.inversion_parameters.max_cg_iterations.value,
            "n_cpu": self.inversion_parameters.n_cpu.value,
            "max_ram": self.inversion_parameters.max_ram.value,
        }

        if self.inversion_parameters.beta_start_options.value == "value":
            input_dict["initial_beta"] = self.inversion_parameters.beta_start.value
        else:
            input_dict["initial_beta_ratio"] = (
                self.inversion_parameters.beta_start.value
            )

        input_dict["tol_cg"] = self.inversion_parameters.tol_cg.value
        input_dict["ignore_values"] = self.inversion_parameters.ignore_values.value
        input_dict["resolution"] = self.resolution.value
        input_dict["window"] = {
            "center_x": self.window_center_x.value,
            "center_y": self.window_center_y.value,
            "width": self.window_width.value,
            "height": self.window_height.value,
            "azimuth": self.window_azimuth.value,
        }
        input_dict["alphas"] = string_2_list(self.inversion_parameters.alphas.value)

        ref_type = self.inversion_parameters.reference_model.options.value.lower()

        if ref_type == "model":
            input_dict["reference_model"] = {
                ref_type: {
                    str(self.inversion_parameters.reference_model.objects.value): str(
                        self.inversion_parameters.reference_model.data.value
                    )
                }
            }
        else:
            input_dict["reference_model"] = {
                ref_type: self.inversion_parameters.reference_model.value.value
            }
        start_type = self.inversion_parameters.starting_model.options.value.lower()

        if start_type == "model":
            input_dict["starting_model"] = {
                start_type: {
                    str(self.inversion_parameters.starting_model.objects.value): str(
                        self.inversion_parameters.starting_model.data.value
                    )
                }
            }
        else:
            input_dict["starting_model"] = {
                start_type: self.inversion_parameters.starting_model.value.value
            }

        if self.inversion_parameters.susceptibility_model.options.value != "None":
            susc_type = (
                self.inversion_parameters.susceptibility_model.options.value.lower()
            )

            if susc_type == "model":
                input_dict["susceptibility_model"] = {
                    susc_type: {
                        str(
                            self.inversion_parameters.susceptibility_model.objects.value
                        ): str(
                            self.inversion_parameters.susceptibility_model.data.value
                        )
                    }
                }
            else:
                input_dict["susceptibility_model"] = {
                    susc_type: self.inversion_parameters.susceptibility_model.value.value
                }

        input_dict["model_norms"] = string_2_list(self.inversion_parameters.norms.value)

        if len(input_dict["model_norms"]) not in [4, 12]:
            print(
                f"Norm values should be 4 or 12 values. List of {len(input_dict['model_norms'])} values provided"
            )

        if len(self.inversion_parameters.lower_bound.value) > 0:
            input_dict["lower_bound"] = string_2_list(
                self.inversion_parameters.lower_bound.value
            )

        if len(self.inversion_parameters.upper_bound.value) > 0:
            input_dict["upper_bound"] = string_2_list(
                self.inversion_parameters.upper_bound.value
            )

        input_dict["data"] = {"type": "GA_object", "name": str(self.objects.value)}

        if getattr(self.data_channel_choices, "data_channel_options", None) is not None:
            channel_param = {}

            for (
                key,
                data_widget,
            ) in self.data_channel_choices.data_channel_options.items():
                if data_widget.children[0].value is False:
                    continue
                channel_param[key] = {}
                channel_param[key]["value"] = string_2_list(
                    data_widget.children[1].value
                )
                channel_param[key]["name"] = str(data_widget.children[2].value)
                channel_param[key]["uncertainties"] = string_2_list(
                    data_widget.children[3].value
                )
                channel_param[key]["offsets"] = string_2_list(
                    data_widget.children[4].value
                )

            input_dict["data"]["channels"] = channel_param
        else:
            input_dict["data"]["channels"] = str(self.data.value)

        if self.sensor.options.value == "sensor location + (dx, dy, dz)":
            input_dict["receivers_offset"] = {
                "constant": string_2_list(self.sensor.offset.value)
            }
        else:
            input_dict["receivers_offset"] = {
                "radar_drape": string_2_list(self.sensor.offset.value)
                + [str(self.sensor.data.value)]
            }

        if self.topography.options.value == "Object":
            if self.topography.objects.value is None:
                input_dict["topography"] = None
            else:
                input_dict["topography"] = {
                    "GA_object": {
                        "name": str(self.topography.objects.value),
                        "data": str(self.topography.data.value),
                    }
                }
        elif self.topography.options.value == "Relative to Sensor":
            input_dict["topography"] = {"draped": self.topography.offset.value}
        else:
            input_dict["topography"] = {"constant": self.topography.constant.value}

        if self.forward_only.value:
            input_dict["forward_only"] = []

        checks = [key for key, val in input_dict.items() if val is None]

        if len(list(input_dict["data"]["channels"])) == 0:
            checks += ["'Channel' for at least one data component."]

        if len(checks) > 0:
            print(f"Required value for {checks}")
            self.trigger.button_style = "danger"
            return

        if (
            self.system.value == "Airborne TEM Survey"
            and self._uncertainties.value is None
        ):
            print("TEM survey requires an Uncertainty group.")
            return

        time_stamp = time()
        with Workspace.create(
            str(
                Path(self.export_directory.selected_path)
                / (self.ga_group_name.value + f"{time_stamp:.0f}.geoh5")
            )
        ) as new_workspace:
            with fetch_active_workspace(self.workspace):
                obj, _ = self.get_selected_entities()
                new_obj = obj.copy(parent=new_workspace, copy_children=False)

                prop_group = obj.find_or_create_property_group(
                    name=self.data.uid_name_map[self.data.value]
                )
                data_list = []
                for prop in prop_group.properties:
                    data = self.workspace.get_entity(prop)[0]
                    data_list.append(data.copy(parent=new_obj))

                data_group = new_obj.add_data_to_group(data_list, prop_group.name)

                if isinstance(input_dict["data"]["channels"], str):
                    input_dict["data"]["channels"] = str(data_group.uid)

                if self._uncertainties.value is not None:
                    prop_group = obj.find_or_create_property_group(
                        name=self.data.uid_name_map[self._uncertainties.value]
                    )
                    data_list = []
                    for prop in prop_group.properties:
                        data = self.workspace.get_entity(prop)[0]
                        data_list.append(data.copy(parent=new_obj))

                    uncert_group = new_obj.add_data_to_group(data_list, prop_group.name)
                    input_dict["uncertainty_channel"] = str(uncert_group.uid)

                _, data = self.sensor.get_selected_entities()
                for d in data:
                    if d is None:
                        continue
                    d.copy(parent=new_obj)

                _, data = self.lines.get_selected_entities()
                for d in data:
                    if d is None:
                        continue
                    d.copy(parent=new_obj)

                for elem in [
                    self.topography,
                    self.inversion_parameters.starting_model,
                    self.inversion_parameters.reference_model,
                    self.inversion_parameters.susceptibility_model,
                ]:
                    obj, data = elem.get_selected_entities()

                    if obj is not None:
                        new_obj = new_workspace.get_entity(obj.uid)[0]
                        if new_obj is None:
                            new_obj = obj.copy(
                                parent=new_workspace, copy_children=False
                            )

                        for d in data:
                            if d is None:
                                continue
                            d.copy(parent=new_obj)

            # Get data_dict for pre-processing
            data_object = new_workspace.get_entity(
                uuid.UUID(input_dict["data"]["name"])
            )[0]
            components = []
            data_dict = {}
            if isinstance(input_dict["data"]["channels"], dict):
                for key, value in input_dict["data"]["channels"].items():
                    comp = key
                    components.append(comp)
                    data_dict[comp + "_channel"] = {
                        "name": new_workspace.get_entity(uuid.UUID(value["name"]))[
                            0
                        ].name
                    }
                    if is_uuid(value["uncertainties"]):
                        data_dict[comp + "_uncertainty"] = {
                            "name": new_workspace.get_entity(
                                uuid.UUID(value["uncertainties"])
                            )[0].name
                        }

            window_options = WindowOptions(
                center_x=self.window_center_x.value,
                center_y=self.window_center_y.value,
                width=self.window_width.value,
                height=self.window_height.value,
                azimuth=self.window_azimuth.value,
            )

            # Pre-processing
            update_dict = preprocess_data(
                new_workspace,
                input_dict,
                data_object,
                window_options,
                drape_options=None,
                resolution=self.resolution.value,
                components=components,
                data_dict=data_dict,
            )
            # Update input dict from pre-processing
            line_data = new_workspace.get_entity(self.lines.data.value)[0]
            input_dict["data"]["name"] = str(update_dict["data_object"])
            survey = new_workspace.get_entity(update_dict["data_object"])[0]
            new_line_uid = survey.get_entity(line_data.name)[0].uid
            input_dict["lines"] = {
                str(new_line_uid): input_dict["lines"][str(line_data.uid)]
            }

            if isinstance(input_dict["data"]["channels"], dict):
                for comp in components:
                    if comp + "_channel" in update_dict:
                        input_dict["data"]["channels"][comp]["name"] = str(
                            update_dict[comp + "_channel"]
                        )
                    if comp + "_uncertainty" in update_dict:
                        input_dict["data"]["channels"][comp]["uncertainties"] = str(
                            update_dict[comp + "_uncertainty"]
                        )
            else:
                new_group = survey.get_entity(data_group.name)[0]
                input_dict["data"]["channels"] = str(new_group.uid)
                new_group = survey.get_entity(uncert_group.name)[0]
                input_dict["uncertainty_channel"] = str(new_group.uid)

        input_dict["workspace"] = input_dict["save_to_geoh5"] = str(
            Path(new_workspace.h5file).resolve()
        )

        file = f"{Path(self.export_directory.selected_path) / self.ga_group_name.value}{time_stamp:.0f}.json"
        self.trigger.file_name = file

        with open(file, "w", encoding="utf8") as f:
            json.dump(input_dict, f, indent=4)

        self.write.button_style = ""
        self.trigger.button_style = "success"
