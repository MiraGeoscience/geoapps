import json
import os

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from geoh5py.groups import ContainerGroup
from geoh5py.objects import BlockModel, Curve, Grid2D, Octree, Points, Surface
from geoh5py.workspace import Workspace
from ipywidgets.widgets import Dropdown, HBox, Label, Layout, Text, VBox

from geoapps.functions.base import Widget
from geoapps.functions.plotting import plot_plan_data_selection
from geoapps.functions.utils import find_value, rotate_xy
from geoapps.functions.selection import object_data_selection_widget, LineOptions


class ChannelOptions(Widget):
    """
    Options for data channels
    """

    def __init__(self, key, description, **kwargs):

        self._active = widgets.Checkbox(value=False, indent=True, description="Active",)
        self._label = widgets.Text(
            description=description, style={"description_width": "initial"},
        )
        self._channel_selection = Dropdown(
            description="Channel", style={"description_width": "initial"}
        )
        self._channel_selection.header = key

        self._uncertainties = widgets.Text(
            description="Error (%, floor)", style={"description_width": "initial"},
        )
        self._offsets = widgets.Text(
            description="Offsets (x,y,z)", style={"description_width": "initial"},
        )

        self._widget = VBox(
            [
                self._active,
                self._label,
                self._channel_selection,
                self._uncertainties,
                self._offsets,
            ]
        )

        super().__init__(**kwargs)

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
    def widget(self):
        return self._widget


class ObjectDataOptions(Widget):
    """
    General widget to get a list of objects and corresponding data and
    property groups from geoh5.
    """

    def __init__(self, h5file, **kwargs):

        self.h5file = h5file
        self.workspace = Workspace(h5file)

        all_obj = [
            entity.name
            for entity in self.workspace.all_objects()
            if isinstance(entity, (Curve, Grid2D, Surface, Points))
        ]

        all_names = [name for name in sorted(all_obj)]

        self._objects = Dropdown(description="Object:", options=all_names)
        self._data_channels = widgets.SelectMultiple(
            description="Groups/Data: ", style={"description_width": "initial"}
        )

        def object_observer(_):
            self.set_data_channels()

        self._objects.observe(object_observer)

        self._widget = VBox([self._objects, self._data_channels])

        super().__init__(**kwargs)

        if "objects" in kwargs.keys() and kwargs["objects"] in self._objects.options:
            self._objects.value = kwargs["objects"]
        # else:
        #     self._objects.value = self._objects.options[0]

        object_observer("")

        if "data_channels" in kwargs.keys():
            for channel in kwargs["data_channels"]:
                if channel in self._data_channels.options:
                    self._data_channels.value = list(self._data_channels.value) + [
                        channel
                    ]

    def set_data_channels(self):
        self._data_channels.options = self.get_comp_list() + self.get_data_list()

    def get_comp_list(self):
        component_list = []

        if self.workspace.get_entity(self._objects.value):
            obj = self.workspace.get_entity(self._objects.value)[0]
            for pg in obj.property_groups:
                component_list.append(pg.name)

        return component_list

    def get_data_list(self):
        data_list = []
        if self.workspace.get_entity(self._objects.value):
            obj = self.workspace.get_entity(self._objects.value)[0]
            data_list = obj.get_data_list()

        return data_list

    @property
    def widget(self):
        return self._widget

    @property
    def objects(self):
        return self._objects

    @property
    def data_channels(self):
        return self._data_channels


class SensorOptions(Widget):
    """
    Define the receiver spatial parameters
    """

    def __init__(self, h5file, objects, **kwargs):
        self.workspace = Workspace(h5file)

        self._objects = objects
        _, self._value = object_data_selection_widget(h5file, objects=objects.value)

        self._value.description = "Height Channel"
        self._value.style = {"description_width": "initial"}

        self._offset = Text(description="[dx, dy, dz]", value="0, 0, 0")

        self._constant = widgets.FloatText(
            description="Constant elevation (m)",
            style={"description_width": "initial"},
        )
        if "offset" in kwargs.keys():
            self._offset.value = kwargs["value"]

        self._options = {
            "(x, y, z) + offset(x,y,z)": self._offset,
            "(x, y, topo + radar) + offset(x,y,z)": VBox([self._offset, self._value]),
        }
        self._options_button = widgets.RadioButtons(
            options=[
                "(x, y, z) + offset(x,y,z)",
                "(x, y, topo + radar) + offset(x,y,z)",
            ],
            description="Define by:",
        )

        def update_options(_):
            self.update_options()

        self._options_button.observe(update_options)
        self._widget = VBox(
            [self._options_button, self._options[self._options_button.value]]
        )

        super().__init__(**kwargs)

        def update_list(_):
            self.update_list()

        self._objects.observe(update_list)

        self.update_list()

        if "value" in kwargs.keys() and kwargs["value"] in self._value.options:
            self._value.value = kwargs["value"]

    @property
    def objects(self):
        return self._objects

    @property
    def offset(self):
        return self._offset

    @property
    def options(self):
        return self._options

    @property
    def options_button(self):
        return self._options_button

    def update_list(self):
        if self.workspace.get_entity(self._objects.value):
            obj = self.workspace.get_entity(self._objects.value)[0]
            data_list = obj.get_data_list() + [""]

            self._value.options = [
                name for name in data_list if "visual" not in name.lower()
            ]

            self._value.value = ""

    def update_options(self):
        self._widget.children = [
            self._options_button,
            self._options[self._options_button.value],
        ]

    @property
    def value(self):
        return self._value

    @property
    def widget(self):
        return self._widget


class TopographyOptions(Widget):
    """
    Define the topography used by the inversion
    """

    def __init__(self, h5file, **kwargs):
        self.h5file = h5file

        self._objects, self._value = object_data_selection_widget(self.h5file)

        def update_list(_):
            self.update_list()

        self._objects.observe(update_list, names="value")
        self._objects.value = find_value(self._objects.options, ["topo", "dem", "dtm"])

        self._panel = VBox([self._objects, self._value])
        self._offset = widgets.FloatText(
            description="Vertical offset (m)", style={"description_width": "initial"},
        )
        self._constant = widgets.FloatText(
            description="Constant elevation (m)",
            style={"description_width": "initial"},
        )
        if "offset" in kwargs.keys():
            self._offset.value = kwargs["value"]

        self._options = {
            "Object": self._panel,
            "Drape Height": self._offset,
            "Constant": self._constant,
            "None": widgets.Label("No topography"),
        }
        self._options_button = widgets.RadioButtons(
            options=["Object", "Drape Height", "Constant"], description="Define by:",
        )

        def update_options(_):
            self.update_options()

        self._options_button.observe(update_options)
        self._widget = VBox(
            [self._options_button, self._options[self._options_button.value]]
        )

        super().__init__(**kwargs)

        if "objects" in kwargs.keys() and kwargs["objects"] in self._objects.options:
            self._objects.value = kwargs["objects"]

        if "value" in kwargs.keys() and kwargs["value"] in self._value.options:
            self._objects.value = kwargs["objects"]

    @property
    def panel(self):
        return self._panel

    @property
    def constant(self):
        return self._constant

    @property
    def objects(self):
        return self._objects

    @property
    def offset(self):
        return self._offset

    @property
    def options(self):
        return self._options

    @property
    def options_button(self):
        return self._options_button

    def update_list(self):
        w_s = Workspace(self.h5file)
        if w_s.get_entity(self.objects.value):
            data_list = w_s.get_entity(self.objects.value)[0].get_data_list()
            self._value.options = [
                name for name in data_list if "visual" not in name.lower()
            ] + ["Vertices"]
            self._value.value = find_value(
                self._value.options, ["dem", "topo", "dtm"], default="Vertices"
            )

    def update_options(self):
        self._widget.children = [
            self._options_button,
            self._options[self._options_button.value],
        ]

    @property
    def value(self):
        return self._value

    @property
    def widget(self):
        return self._widget


class OctreeMeshOptions(Widget):
    """
    Widget used for the creation of an octree meshes
    """

    def __init__(self, **kwargs):
        self._core_cell_size = widgets.Text(
            value="25, 25, 25",
            description="Smallest cells",
            style={"description_width": "initial"},
        )
        self._octree_levels_topo = widgets.Text(
            value="0, 0, 0, 2",
            description="Layers below topo",
            style={"description_width": "initial"},
        )
        self._octree_levels_obs = widgets.Text(
            value="5, 5, 5, 5",
            description="Layers below data",
            style={"description_width": "initial"},
        )
        self._depth_core = widgets.FloatText(
            value=500,
            description="Minimum depth (m)",
            style={"description_width": "initial"},
        )
        self._padding_distance = widgets.Text(
            value="0, 0, 0, 0, 0, 0",
            description="Padding [W,E,N,S,D,U] (m)",
            style={"description_width": "initial"},
        )

        self._max_distance = widgets.FloatText(
            value=1000,
            description="Max triangulation length",
            style={"description_width": "initial"},
        )

        self._widget = widgets.VBox(
            [
                Label("Octree Mesh"),
                self._core_cell_size,
                self._octree_levels_topo,
                self._octree_levels_obs,
                self._depth_core,
                self._padding_distance,
                self._max_distance,
            ]
        )

        super().__init__(**kwargs)

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
    def widget(self):
        return self._widget


class ModelOptions(Widget):
    """
    Widgets for the selection of model options
    """

    def __init__(self, name_list, **kwargs):

        self._options = widgets.RadioButtons(
            options=["Model", "Value"], value="Value", disabled=False,
        )

        def update_panel(_):
            self.update_panel()

        self._options.observe(update_panel, names="value")
        self._list = widgets.Dropdown(description="3D Model", options=name_list)
        self._value = widgets.FloatText(description="Units")
        self._description = Label()
        self._widget = widgets.VBox(
            [self._description, widgets.VBox([self._options, self._value])]
        )

        super().__init__(**kwargs)

        for key, value in kwargs.items():
            if key == "units":
                self._value.description = value
            elif key == "label":
                self._description.value = value

    def update_panel(self):

        if self._options.value == "Model":
            self._widget.children[1].children = [self._options, self._list]
            self._widget.children[1].children[1].layout.visibility = "visible"
        elif self._options.value == "Value":
            self._widget.children[1].children = [self._options, self._value]
            self._widget.children[1].children[1].layout.visibility = "visible"
        else:
            self._widget.children[1].children[1].layout.visibility = "hidden"

    @property
    def description(self):
        return self._description

    @property
    def list(self):
        return self._list

    @property
    def options(self):
        return self._options

    @property
    def value(self):
        return self._value

    @property
    def widget(self):
        return self._widget


class InversionOptions(Widget):
    """
    Collection of widgets controlling the inversion parameters
    """

    def __init__(self, h5file, **kwargs):

        self.workspace = Workspace(h5file)

        model_list = []
        for obj in self.workspace.all_objects():
            if isinstance(obj, (BlockModel, Octree, Surface)):
                for data in obj.children:
                    if (
                        getattr(data, "values", None) is not None
                        and data.name != "Visual Parameters"
                    ):
                        model_list += [data.name]

        self._output_name = widgets.Text(
            value="Inversion_", description="Save to:", disabled=False
        )

        self._chi_factor = widgets.FloatText(
            value=1, description="Target misfit", disabled=False
        )
        self._uncert_mode = widgets.RadioButtons(
            options=[
                "Estimated (%|data| + background)",
                r"User input (\%|data| + floor)",
            ],
            value=r"User input (\%|data| + floor)",
            disabled=False,
        )
        self._lower_bound = widgets.Text(value=None, description="Lower bound value",)
        self._upper_bound = widgets.Text(value=None, description="Upper bound value",)

        self._ignore_values = widgets.Text(value="<0", tooltip="Dummy value",)
        self._max_iterations = widgets.IntText(
            value=10, description="Max beta Iterations"
        )
        self._max_cg_iterations = widgets.IntText(
            value=10, description="Max CG Iterations"
        )
        self._tol_cg = widgets.FloatText(value=1e-4, description="CG Tolerance")

        self._beta_start_options = widgets.RadioButtons(
            options=["value", "ratio"],
            value="ratio",
            description="Starting tradeoff (beta):",
        )
        self._beta_start = widgets.FloatText(value=1e2, description="phi_d/phi_m")

        def initial_beta_change(_):
            self.initial_beta_change()

        self._beta_start_options.observe(initial_beta_change)

        self._beta_start_panel = HBox([self._beta_start_options, self._beta_start])
        self._optimization = VBox(
            [
                self._max_iterations,
                self._chi_factor,
                self._beta_start_panel,
                self._max_cg_iterations,
                self._tol_cg,
            ]
        )
        self._starting_model = ModelOptions(model_list)
        self._susceptibility_model = ModelOptions(
            model_list, units="SI", label="Background susceptibility"
        )
        self._susceptibility_model.options.options = ["None", "Model", "Value"]
        self._reference_model = ModelOptions(model_list)
        self._reference_model.options.options = [
            "None",
            "Best-fitting halfspace",
            "Model",
            "Value",
        ]
        self._alphas = widgets.Text(
            value="1, 1, 1, 1", description="Scaling alpha_(s, x, y, z)",
        )
        self._norms = widgets.Text(
            value="2, 2, 2, 2", description="Norms p_(s, x, y, z)",
        )

        def check_max_iterations(_):
            self.check_max_iterations()

        self._norms.observe(check_max_iterations)

        self._mesh = OctreeMeshOptions()
        self.inversion_options = {
            "output name": self._output_name,
            "uncertainties": self._uncert_mode,
            "starting model": self._starting_model.widget,
            "background susceptibility": self._susceptibility_model.widget,
            "regularization": VBox(
                [self._reference_model.widget, self._alphas, self._norms]
            ),
            "upper-lower bounds": VBox([self._upper_bound, self._lower_bound]),
            "mesh": self._mesh.widget,
            "ignore values (<0 = no negatives)": self._ignore_values,
            "optimization": self._optimization,
        }
        self.option_choices = widgets.Dropdown(
            options=list(self.inversion_options.keys()),
            value=list(self.inversion_options.keys())[0],
            disabled=False,
        )

        def inversion_option_change(_):
            self.inversion_option_change()

        self.option_choices.observe(inversion_option_change)

        self._widget = widgets.VBox(
            [
                widgets.HBox([widgets.Label("Inversion Options")]),
                widgets.HBox(
                    [
                        self.option_choices,
                        self.inversion_options[self.option_choices.value],
                    ],
                ),
            ],
            layout=Layout(width="100%"),
        )

        super().__init__(**kwargs)

        for obj in self.__dict__:
            if hasattr(getattr(self, obj), "style"):
                getattr(self, obj).style = {"description_width": "initial"}

    def check_max_iterations(self):
        if not all([val == 2 for val in string_2_list(self._norms.value)]):
            self._max_iterations.value = 30
        else:
            self._max_iterations.value = 10

    def inversion_option_change(self):
        self._widget.children[1].children = [
            self.option_choices,
            self.inversion_options[self.option_choices.value],
        ]

    def initial_beta_change(self):
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
    def mesh(self):
        return self._mesh

    @property
    def norms(self):
        return self._norms

    @property
    def output_name(self):
        return self._output_name

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

    @property
    def widget(self):
        return self._widget


def get_inversion_output(h5file, group_name):
    """
    Recover an inversion iterations from a ContainerGroup comments.
    """
    workspace = Workspace(h5file)
    out = {"time": [], "iteration": [], "phi_d": [], "phi_m": [], "beta": []}

    if workspace.get_entity(group_name):
        group = workspace.get_entity(group_name)[0]

        for comment in group.comments.values:
            if "Iteration" in comment["Author"]:
                out["iteration"] += [np.int(comment["Author"].split("_")[1])]
                out["time"] += [comment["Date"]]

                values = json.loads(comment["Text"])

                out["phi_d"] += [np.float(values["phi_d"])]
                out["phi_m"] += [np.float(values["phi_m"])]
                out["beta"] += [np.float(values["beta"])]

        if len(out["iteration"]) > 0:
            out["iteration"] = np.hstack(out["iteration"])
            ind = np.argsort(out["iteration"])
            out["iteration"] = out["iteration"][ind]
            out["phi_d"] = np.hstack(out["phi_d"])[ind]
            out["phi_m"] = np.hstack(out["phi_m"])[ind]
            out["time"] = np.hstack(out["time"])[ind]

    return out


def plot_convergence_curve(h5file):
    """

    """
    workspace = Workspace(h5file)

    names = [
        group.name
        for group in workspace.all_groups()
        if isinstance(group, ContainerGroup)
    ]

    objects = widgets.Dropdown(
        options=names,
        value=names[0],
        description="Inversion Group:",
        style={"description_width": "initial"},
    )

    def plot_curve(objects):

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


def inversion_widgets(h5file, **kwargs):
    dir_path = os.path.dirname(os.path.realpath(os.path.realpath(__file__)))
    with open(os.path.join(dir_path, "AEM_systems.json")) as aem_systems:
        em_system_specs = json.load(aem_systems)

    # Load all known em systems
    widget_list = {
        "azimuth": widgets.FloatSlider(
            min=-90,
            max=90,
            value=0,
            steps=5,
            description="Orientation",
            continuous_update=False,
        ),
        "center_x": widgets.FloatSlider(
            min=np.inf,
            max=np.inf,
            steps=10,
            description="Easting",
            continuous_update=False,
        ),
        "center_y": widgets.FloatSlider(
            min=np.inf,
            max=np.inf,
            steps=10,
            description="Northing",
            continuous_update=False,
            orientation="vertical",
        ),
        "data_count": Label("Data Count: 0", tooltip="Keep <1500 for speed"),
        "forward_only": widgets.Checkbox(
            value=False,
            description="Forward only",
            tooltip="Forward response of reference model",
        ),
        "hz_expansion": widgets.FloatText(value=1.05, description="Expansion factor:",),
        "hz_min": widgets.FloatText(value=10.0, description="Smallest cell (m):",),
        "inducing_field": widgets.Text(
            description="Inducing Field [Amp, Inc, Dec]",
            style={"description_width": "initial"},
        ),
        "n_cells": widgets.FloatText(value=25.0, description="Number of cells:",),
        "resolution": widgets.FloatText(description="Resolution (m)"),
        "run": widgets.ToggleButton(
            value=False, description="Run SimPEG", button_style="danger", icon="check"
        ),
        "starting_channel": widgets.IntText(value=None, description="Starting Channel"),
        "system": Dropdown(
            options=["Magnetics", "Gravity"] + list(em_system_specs.keys()),
            description="Survey Type: ",
        ),
        "width_x": widgets.FloatSlider(
            min=np.inf,
            max=np.inf,
            steps=10,
            description="Width",
            continuous_update=False,
        ),
        "width_y": widgets.FloatSlider(
            min=np.inf,
            max=np.inf,
            steps=10,
            description="Height",
            continuous_update=False,
            orientation="vertical",
        ),
        "write": widgets.ToggleButton(
            value=False,
            description="Write input",
            button_style="warning",
            tooltip="Write json input file",
            icon="check",
        ),
        "zoom_extent": widgets.ToggleButton(
            value=False,
            description="Zoom on selection",
            tooltip="Keep plot extent on selection",
            icon="check",
        ),
    }

    for widget in widget_list.values():
        widget.style = {"description_width": "initial"}

    for attr, item in kwargs.items():
        try:
            if (
                hasattr(widget_list[attr], "options")
                and item not in widget_list[attr].options
            ):
                continue
            widget_list[attr].value = item
        except KeyError:
            continue

    return widget_list


def inversion_defaults():
    """
    Get defaults for gravity, magnetics and EM1D inversions
    """
    defaults = {
        "units": {"Gravity": "g/cc", "Magnetics": "SI", "EM1D": "S/m"},
        "property": {
            "Gravity": "density",
            "Magnetics": "effective susceptibility",
            "EM1D": "conductivity",
        },
        "reference_value": {"Gravity": 0.0, "Magnetics": 0.0, "EM1D": 1e-3},
        "starting_value": {"Gravity": 1e-4, "Magnetics": 1e-4, "EM1D": 1e-3},
    }

    return defaults


def string_2_list(string):
    """
    Convert a list of numbers separated by comma to a list of floats
    """
    return [np.float(val) for val in string.split(",") if len(val) > 0]


def inversion_app(h5file, **kwargs):

    # Load all known em systems
    dir_path = os.path.dirname(os.path.realpath(os.path.realpath(__file__)))
    with open(os.path.join(dir_path, "AEM_systems.json")) as aem_systems:
        em_system_specs = json.load(aem_systems)

    w_l = inversion_widgets(h5file, **kwargs)
    system = w_l["system"]
    defaults = inversion_defaults()
    workspace = Workspace(h5file)
    o_d = ObjectDataOptions(h5file)

    all_obj = [
        entity.name
        for entity in workspace.all_objects()
        if isinstance(entity, (Curve, Grid2D, Surface, Points))
    ]

    all_names = [name for name in sorted(all_obj)]

    dsep = os.path.sep
    inv_dir = dsep.join(os.path.dirname(os.path.abspath(h5file)).split(dsep))

    if len(inv_dir) > 0:
        inv_dir += dsep
    else:
        inv_dir = os.getcwd() + dsep

    def run_unclick(_):
        if w_l["run"].value:

            if system.value in ["Gravity", "Magnetics"]:
                os.system(
                    "start cmd.exe @cmd /k "
                    + 'python functions/pf_inversion.py "'
                    + inv_dir
                    + f'\\{inversion_parameters.output_name.value}.json"'
                )
            else:
                os.system(
                    "start cmd.exe @cmd /k "
                    + 'python functions/em1d_inversion.py "'
                    + inv_dir
                    + f'\\{inversion_parameters.output_name.value}.json"'
                )

            w_l["run"].value = False
            w_l["run"].button_style = ""

    w_l["run"].observe(run_unclick)

    # # OCTREE MESH
    inversion_parameters = InversionOptions(h5file)
    mesh = inversion_parameters.mesh

    # 1D MESH
    def count_cells():
        return (
            w_l["hz_min"].value
            * w_l["hz_expansion"].value ** np.arange(w_l["n_cells"].value)
        ).sum()

    cell_count = Label(f"Max depth: {count_cells():.2f} m")

    def update_hz_count(_):
        cell_count.value = f"Max depth: {count_cells():.2f} m"
        w_l["write"].button_style = "warning"
        w_l["run"].button_style = "danger"

    w_l["n_cells"].observe(update_hz_count)
    w_l["hz_expansion"].observe(update_hz_count)
    w_l["hz_min"].observe(update_hz_count)
    hz_panel = VBox(
        [
            Label("1D Mesh"),
            w_l["hz_min"],
            w_l["hz_expansion"],
            w_l["n_cells"],
            cell_count,
        ]
    )

    # Check parameters change
    def update_options(_):
        w_l["write"].button_style = "warning"
        w_l["run"].button_style = "danger"

    for item in inversion_parameters.__dict__.values():
        if isinstance(item, widgets.Widget):
            item.observe(update_options)
        elif isinstance(item, Widget):
            for val in item.__dict__.values():
                if isinstance(val, widgets.Widget):
                    val.observe(update_options)

    def update_ref(_):
        alphas = string_2_list(inversion_parameters.alphas.value)
        if inversion_parameters.reference_model.options.value == "None":
            alphas[0] = 0.0
        else:
            alphas[0] = 1.0

        inversion_parameters.alphas.value = ", ".join(list(map(str, alphas)))
        w_l["write"].button_style = "warning"
        w_l["run"].button_style = "danger"

    inversion_parameters.reference_model.options.observe(update_ref)

    def channel_setter(caller):

        channel = caller["owner"]
        data_widget = data_channel_choices.data_channel_options[channel.header]

        entity = workspace.get_entity(o_d.objects.value)[0]
        if channel.value is None or not entity.get_data(channel.value):
            data_widget.children[0].value = False
            if system.value in ["Magnetics", "Gravity"]:
                data_widget.children[3].value = "0, 1"
        else:
            data_widget.children[0].value = True
            if system.value in ["Magnetics", "Gravity"]:
                values = entity.get_data(channel.value)[0].values
                if values is not None and isinstance(values[0], float):
                    data_widget.children[
                        3
                    ].value = f"0, {np.percentile(values[values > 2e-18], 5):.2f}"

        # Trigger plot update
        if data_channel_choices.value == channel.header:
            data_channel_choices.value = None
            data_channel_choices.value = channel.header

    def system_observer(_, start_channel=w_l["starting_channel"].value):

        if system.value in ["Magnetics", "Gravity"]:
            if system.value == "Magnetics":
                data_type_list = ["tmi", "bxx", "bxy", "bxz", "byy", "byz", "bzz"]
                labels = ["tmi", "bxx", "bxy", "bxz", "byy", "byz", "bzz"]

            else:
                data_type_list = ["gz", "gxx", "gxy", "gxz", "gyy", "gyz", "gzz"]
                labels = ["gz", "gxx", "gxy", "gxz", "gyy", "gyz", "gzz"]

            tx_offsets = [[0, 0, 0]]
            uncertainties = [[0, 1]] * len(data_type_list)

            system_specs = {}
            for key in data_type_list:
                system_specs[key] = key

            # Remove line_id from choices
            spatial_choices.options = list(spatial_options.keys())[:2]

            # Switch mesh options
            inversion_parameters.inversion_options[
                "mesh"
            ] = inversion_parameters.mesh.widget

            inversion_parameters.reference_model.options.options = [
                "None",
                "Model",
                "Value",
            ]
            inversion_parameters.reference_model.options.value = "Value"
            flag = system.value

            inversion_parameters.lower_bound.value = ""
            inversion_parameters.upper_bound.value = ""
            inversion_parameters.ignore_values.value = "-99999"

        else:
            tx_offsets = em_system_specs[system.value]["tx_offsets"]
            sensor.offset.value = ", ".join(
                [str(offset) for offset in em_system_specs[system.value]["bird_offset"]]
            )
            uncertainties = em_system_specs[system.value]["uncertainty"]

            # if start_channel is None:
            start_channel = em_system_specs[system.value]["channel_start_index"]
            if em_system_specs[system.value]["type"] == "time":
                labels = ["Time (s)"] * len(em_system_specs[system.value]["channels"])
            else:
                labels = ["Frequency (Hz)"] * len(
                    em_system_specs[system.value]["channels"]
                )

            system_specs = {}
            for key, time in em_system_specs[system.value]["channels"].items():
                system_specs[key] = f"{time:.5e}"

            spatial_choices.options = list(spatial_options.keys())

            inversion_parameters.reference_model.options.options = [
                "Best-fitting halfspace",
                "Model",
                "Value",
            ]
            inversion_parameters.reference_model.options.value = (
                "Best-fitting halfspace"
            )
            inversion_parameters.lower_bound.value = "1e-5"
            inversion_parameters.upper_bound.value = "10"
            inversion_parameters.ignore_values.value = "<0"
            # Switch mesh options
            inversion_parameters.inversion_options["mesh"] = hz_panel
            flag = "EM1D"

        inversion_parameters.reference_model.value.description = defaults["units"][flag]
        inversion_parameters.reference_model.value.value = defaults["reference_value"][
            flag
        ]
        inversion_parameters.reference_model.description.value = (
            "Reference " + defaults["property"][flag]
        )
        inversion_parameters.starting_model.value.description = defaults["units"][flag]
        inversion_parameters.starting_model.value.value = defaults["starting_value"][
            flag
        ]
        inversion_parameters.starting_model.description.value = (
            "Starting " + defaults["property"][flag]
        )

        spatial_choices.value = spatial_choices.options[0]

        data_channel_options = {}
        for ind, (key, channel) in enumerate(system_specs.items()):
            if ind + 1 < start_channel:
                continue

            if len(tx_offsets) > 1:
                offsets = tx_offsets[ind]
            else:
                offsets = tx_offsets[0]

            channel_selection = Dropdown(
                description="Channel", style={"description_width": "initial"}
            )
            channel_selection.header = key
            channel_selection.observe(channel_setter, names="value")

            channel_options = ChannelOptions(
                key,
                labels[ind],
                uncertainties=", ".join(
                    [str(uncert) for uncert in uncertainties[ind][:2]]
                ),
                offsets=", ".join([str(offset) for offset in offsets]),
            )

            channel_options.channel_selection.observe(channel_setter, names="value")

            data_channel_options[key] = channel_options.widget

            if system.value not in ["Magnetics", "Gravity"]:
                data_channel_options[key].children[1].value = channel
            else:
                data_channel_options[key].children[1].layout.visibility = "hidden"
                data_channel_options[key].children[4].layout.visibility = "hidden"

        if len(data_channel_options) > 0:
            data_channel_choices.options = list(data_channel_options.keys())
            data_channel_choices.value = list(data_channel_options.keys())[0]
            data_channel_choices.data_channel_options = data_channel_options
            data_channel_panel.children = [
                data_channel_choices,
                data_channel_options[data_channel_choices.value],
            ]

        update_component_panel("")

        if (
            system.value not in ["Magnetics", "Gravity"]
            and em_system_specs[system.value]["type"] == "frequency"
        ):
            inversion_parameters.option_choices.options = list(
                inversion_parameters.inversion_options.keys()
            )
        else:
            inversion_parameters.option_choices.options = [
                key
                for key in inversion_parameters.inversion_options.keys()
                if key != "background susceptibility"
            ]

        w_l["write"].button_style = "warning"
        w_l["run"].button_style = "danger"

        if system.value == "Magnetics":
            survey_type_panel.children = [system, w_l["inducing_field"]]
        else:
            survey_type_panel.children = [system]

    def object_observer(_):

        w_l["resolution"].indices = None

        if workspace.get_entity(o_d.objects.value):
            obj = workspace.get_entity(o_d.objects.value)[0]
            data_list = obj.get_data_list()
            lines.update_list()
            topography.update_list()

            for aem_system, specs in em_system_specs.items():
                if any([specs["flag"] in channel for channel in data_list]):
                    system.value = aem_system

            system_observer("")

            if hasattr(data_channel_choices, "data_channel_options"):
                for (
                    key,
                    data_widget,
                ) in data_channel_choices.data_channel_options.items():
                    data_widget.children[2].options = o_d.data_channels.options
                    value = find_value(o_d.data_channels.options, [key])
                    data_widget.children[2].value = value

            w_l["write"].button_style = "warning"
            w_l["run"].button_style = "danger"

    o_d.objects.observe(object_observer, names="value")
    system.observe(system_observer, names="value")

    def get_data_list(entity):
        groups = [p_g.name for p_g in entity.property_groups]
        data_list = []
        if o_d.data_channels.value is not None:
            for component in o_d.data_channels.value:
                if component in groups:
                    data_list += [
                        workspace.get_entity(data)[0].name
                        for data in entity.get_property_group(component).properties
                    ]
                elif component in entity.get_data_list():
                    data_list += [component]
        return data_list

    def update_component_panel(_):
        if workspace.get_entity(o_d.objects.value):
            entity = workspace.get_entity(o_d.objects.value)[0]
            data_list = get_data_list(entity)

            if hasattr(data_channel_choices, "data_channel_options"):
                for (
                    key,
                    data_widget,
                ) in data_channel_choices.data_channel_options.items():
                    data_widget.children[2].options = data_list
                    value = find_value(data_list, [key])
                    data_widget.children[2].value = value

    o_d.data_channels.observe(update_component_panel, names="value")

    def data_channel_choices_observer(_):
        if hasattr(
            data_channel_choices, "data_channel_options"
        ) and data_channel_choices.value in (
            data_channel_choices.data_channel_options.keys()
        ):
            data_widget = data_channel_choices.data_channel_options[
                data_channel_choices.value
            ]
            data_channel_panel.children = [data_channel_choices, data_widget]

            if (
                workspace.get_entity(o_d.objects.value)
                and data_widget.children[2].value is None
            ):
                entity = workspace.get_entity(o_d.objects.value)[0]
                data_list = get_data_list(entity)
                value = find_value(data_list, [data_channel_choices.value])
                data_widget.children[2].value = value

        w_l["write"].button_style = "warning"
        w_l["run"].button_style = "danger"

    data_channel_choices = widgets.Dropdown(
        description="Component:", style={"description_width": "initial"}
    )

    data_channel_choices.observe(data_channel_choices_observer, names="value")
    data_channel_panel = widgets.VBox([data_channel_choices])

    survey_type_panel = VBox([system])

    # Spatial parameters
    # Topography definition
    topography = TopographyOptions(h5file)

    # Define bird parameters
    sensor = SensorOptions(h5file, o_d.objects)

    # LINE ID
    lines = LineOptions(h5file, o_d.objects)

    # SPATIAL PARAMETERS DROPDOWN
    spatial_options = {
        "Topography": topography.widget,
        "Receivers": sensor.widget,
        "Line ID": lines.widget,
    }

    spatial_choices = widgets.Dropdown(
        options=list(spatial_options.keys()),
        value=list(spatial_options.keys())[0],
        disabled=False,
    )

    spatial_panel = VBox([spatial_choices, spatial_options[spatial_choices.value]])

    def spatial_option_change(_):
        spatial_panel.children = [
            spatial_choices,
            spatial_options[spatial_choices.value],
        ]
        w_l["write"].button_style = "warning"
        w_l["run"].button_style = "danger"

    spatial_choices.observe(spatial_option_change)

    # Data selection and plotting from object extent
    lim_x = [1e8, 0]
    lim_y = [1e8, 0]
    for name in all_names:
        obj = workspace.get_entity(name)[0]
        if obj.vertices is not None:
            lim_x[0], lim_x[1] = (
                np.min([lim_x[0], obj.vertices[:, 0].min()]),
                np.max([lim_x[1], obj.vertices[:, 0].max()]),
            )
            lim_y[0], lim_y[1] = (
                np.min([lim_y[0], obj.vertices[:, 1].min()]),
                np.max([lim_y[1], obj.vertices[:, 1].max()]),
            )
        elif hasattr(obj, "centroids"):
            lim_x[0], lim_x[1] = (
                np.min([lim_x[0], obj.centroids[:, 0].min()]),
                np.max([lim_x[1], obj.centroids[:, 0].max()]),
            )
            lim_y[0], lim_y[1] = (
                np.min([lim_y[0], obj.centroids[:, 1].min()]),
                np.max([lim_y[1], obj.centroids[:, 1].max()]),
            )

    w_l["center_x"].min, w_l["center_x"].max, w_l["center_x"].value = (
        lim_x[0],
        lim_x[1],
        np.mean(lim_x),
    )
    w_l["center_y"].min, w_l["center_y"].max, w_l["center_y"].value = (
        lim_y[0],
        lim_y[1],
        np.mean(lim_y),
    )
    w_l["width_x"].min, w_l["width_x"].max, w_l["width_x"].value = (
        100,
        lim_x[1] - lim_x[0],
        lim_x[1] - lim_x[0],
    )
    w_l["width_y"].min, w_l["width_y"].max, w_l["width_y"].value = (
        100,
        lim_y[1] - lim_y[0],
        lim_y[1] - lim_y[0],
    )

    def update_octree_param(_):
        dl = w_l["resolution"].value
        mesh.core_cell_size.value = f"{dl/2:.0f}, {dl/2:.0f}, {dl/2:.0f}"
        mesh.depth_core.value = np.ceil(
            np.min([w_l["width_x"].value, w_l["width_y"].value]) / 2.0
        )

        mesh.padding_distance.value = ", ".join(
            list(
                map(
                    str,
                    [
                        np.ceil(w_l["width_x"].value / 2),
                        np.ceil(w_l["width_x"].value / 2),
                        np.ceil(w_l["width_y"].value / 2),
                        np.ceil(w_l["width_y"].value / 2),
                        0,
                        0,
                    ],
                )
            )
        )
        w_l["resolution"].indices = None

    w_l["width_x"].observe(update_octree_param)
    w_l["width_y"].observe(update_octree_param)
    w_l["resolution"].observe(update_octree_param, names="value")
    w_l["resolution"].indices = None

    def plot_selection(
        entity_name,
        data_choice,
        resolution,
        line_ids,
        center_x,
        center_y,
        width_x,
        width_y,
        azimuth,
        zoom_extent,
        marker_size,
    ):
        if workspace.get_entity(entity_name):
            obj = workspace.get_entity(entity_name)[0]

            name = None
            if hasattr(
                data_channel_choices, "data_channel_options"
            ) and data_choice in (data_channel_choices.data_channel_options.keys()):
                name = (
                    data_channel_choices.data_channel_options[data_choice]
                    .children[2]
                    .value
                )

            if (
                obj.get_data(name)
                and isinstance(obj.get_data(name)[0].values, np.ndarray)
                and isinstance(obj.get_data(name)[0].values[0], float)
            ):

                data_obj = obj.get_data(name)[0]

                plt.figure(figsize=(10, 10))
                ax1 = plt.subplot()
                corners = np.r_[
                    np.c_[-1.0, -1.0],
                    np.c_[-1.0, 1.0],
                    np.c_[1.0, 1.0],
                    np.c_[1.0, -1.0],
                    np.c_[-1.0, -1.0],
                ]
                corners[:, 0] *= width_x / 2
                corners[:, 1] *= width_y / 2
                corners = rotate_xy(corners, [0, 0], -azimuth)
                ax1.plot(corners[:, 0] + center_x, corners[:, 1] + center_y, "k")

                _, _, indices, line_selection = plot_plan_data_selection(
                    obj,
                    data_obj,
                    **{
                        "ax": ax1,
                        "highlight_selection": {lines.value.value: line_ids},
                        "resolution": resolution,
                        "window": {
                            "center": [center_x, center_y],
                            "size": [width_x, width_y],
                            "azimuth": azimuth,
                        },
                        "zoom_extent": zoom_extent,
                        "marker_size": marker_size,
                        "indices": w_l["resolution"].indices,
                    },
                )

                if w_l["resolution"].indices is None:
                    w_l["resolution"].indices = indices
                data_count = 0
                for widget in data_channel_choices.data_channel_options.values():
                    if system.value in ["Magnetics", "Gravity"]:
                        data_count += widget.children[0].value * indices.sum()
                    else:
                        data_count += widget.children[0].value * line_selection.sum()

                if system.value in ["Magnetics", "Gravity"]:
                    values = np.abs(data_obj.values).reshape(indices.shape, order="F")[
                        indices
                    ]
                    w_l[
                        "data_count"
                    ].value = f"Data Count: {data_count}, 10th PCT |d|: {np.percentile(values[values > 2e-18], 5):.2f}"
                else:
                    w_l["data_count"].value = f"Data Count: {data_count}"
                w_l["write"].button_style = "warning"
                w_l["run"].button_style = "danger"

    marker_size = widgets.IntSlider(
        min=1, max=100, value=3, description="Markers", continuous_update=False,
    )
    plot_window = widgets.interactive_output(
        plot_selection,
        {
            "entity_name": o_d.objects,
            "data_choice": data_channel_choices,
            "line_ids": lines.lines,
            "resolution": w_l["resolution"],
            "center_x": w_l["center_x"],
            "center_y": w_l["center_y"],
            "width_x": w_l["width_x"],
            "width_y": w_l["width_y"],
            "azimuth": w_l["azimuth"],
            "zoom_extent": w_l["zoom_extent"],
            "marker_size": marker_size,
        },
    )
    selection_panel = VBox(
        [
            Label("Window & Downsample"),
            VBox(
                [
                    w_l["resolution"],
                    w_l["data_count"],
                    marker_size,
                    HBox(
                        [w_l["center_y"], w_l["width_y"], plot_window],
                        layout=Layout(align_items="center"),
                    ),
                    VBox(
                        [
                            w_l["width_x"],
                            w_l["center_x"],
                            w_l["azimuth"],
                            w_l["zoom_extent"],
                        ],
                        layout=Layout(align_items="center"),
                    ),
                ],
                layout=Layout(align_items="center"),
            ),
        ],
        layout=Layout(width="50%"),
    )

    def write_unclick(_):
        if w_l["write"].value is False:
            return

        input_dict = {
            "out_group": inversion_parameters.output_name.value,
            "workspace": h5file,
            "save_to_geoh5": h5file,
        }
        if system.value in ["Gravity", "Magnetics"]:
            input_dict["inversion_type"] = system.value.lower()

            if input_dict["inversion_type"] == "magnetics":
                input_dict["inducing_field_aid"] = string_2_list(
                    w_l["inducing_field"].value
                )
            # Octree mesh parameters
            input_dict["core_cell_size"] = string_2_list(mesh.core_cell_size.value)
            input_dict["octree_levels_topo"] = string_2_list(
                mesh.octree_levels_topo.value
            )
            input_dict["octree_levels_obs"] = string_2_list(
                mesh.octree_levels_obs.value
            )
            input_dict["depth_core"] = {"value": mesh.depth_core.value}
            input_dict["max_distance"] = mesh.max_distance.value
            p_d = string_2_list(mesh.padding_distance.value)
            input_dict["padding_distance"] = [
                [p_d[0], p_d[1]],
                [p_d[2], p_d[3]],
                [p_d[4], p_d[5]],
            ]

        else:
            input_dict["system"] = system.value
            input_dict["lines"] = {
                lines.value.value: [str(line) for line in lines.lines.value]
            }

            input_dict["mesh 1D"] = [
                w_l["hz_min"].value,
                w_l["hz_expansion"].value,
                w_l["n_cells"].value,
            ]
        input_dict["chi_factor"] = inversion_parameters.chi_factor.value
        input_dict["max_iterations"] = inversion_parameters.max_iterations.value
        input_dict["max_cg_iterations"] = inversion_parameters.max_cg_iterations.value

        if inversion_parameters.beta_start_options.value == "value":
            input_dict["initial_beta"] = inversion_parameters.beta_start.value
        else:
            input_dict["initial_beta_ratio"] = inversion_parameters.beta_start.value

        input_dict["tol_cg"] = inversion_parameters.tol_cg.value
        input_dict["ignore_values"] = inversion_parameters.ignore_values.value
        input_dict["resolution"] = w_l["resolution"].value
        input_dict["window"] = {
            "center": [w_l["center_x"].value, w_l["center_y"].value],
            "size": [w_l["width_x"].value, w_l["width_y"].value],
            "azimuth": w_l["azimuth"].value,
        }
        input_dict["alphas"] = string_2_list(inversion_parameters.alphas.value)

        input_dict["reference_model"] = {
            inversion_parameters.reference_model.widget.children[1]
            .children[0]
            .value.lower(): inversion_parameters.reference_model.widget.children[1]
            .children[1]
            .value
        }

        input_dict["starting_model"] = {
            inversion_parameters.starting_model.widget.children[1]
            .children[0]
            .value.lower(): inversion_parameters.starting_model.widget.children[1]
            .children[1]
            .value
        }

        if inversion_parameters.susceptibility_model.options.value != "None":
            input_dict["susceptibility"] = {
                inversion_parameters.susceptibility_model.widget.children[1]
                .children[0]
                .value.lower(): inversion_parameters.susceptibility_model.widget.children[
                    1
                ]
                .children[1]
                .value
            }

        input_dict["model_norms"] = string_2_list(inversion_parameters.norms.value)

        if len(inversion_parameters.lower_bound.value) > 1:
            input_dict["lower_bound"] = string_2_list(
                inversion_parameters.lower_bound.value
            )

        if len(inversion_parameters.upper_bound.value) > 1:
            input_dict["upper_bound"] = string_2_list(
                inversion_parameters.upper_bound.value
            )

        input_dict["data"] = {}
        input_dict["data"]["type"] = "GA_object"
        input_dict["data"]["name"] = o_d.objects.value

        if hasattr(data_channel_choices, "data_channel_options"):
            channel_param = {}

            for key, data_widget in data_channel_choices.data_channel_options.items():
                if data_widget.children[0].value is False:
                    continue

                channel_param[key] = {}
                channel_param[key]["name"] = data_widget.children[2].value
                channel_param[key]["uncertainties"] = string_2_list(
                    data_widget.children[3].value
                )
                channel_param[key]["offsets"] = string_2_list(
                    data_widget.children[4].value
                )

                if system.value not in ["Gravity", "Magnetics"]:
                    channel_param[key]["value"] = string_2_list(
                        data_widget.children[1].value
                    )
                if (
                    system.value in ["Gravity", "Magnetics"]
                    and w_l["azimuth"].value != 0
                    and key not in ["tmi", "gz"]
                ):
                    print(
                        f"Gradient data with rotated window is currently not supported"
                    )
                    w_l["run"].button_style = "danger"
                    return

            input_dict["data"]["channels"] = channel_param

        input_dict["uncertainty_mode"] = inversion_parameters.uncert_mode.value

        if sensor.options_button.value == "(x, y, z) + offset(x,y,z)":
            input_dict["receivers_offset"] = {
                "constant": string_2_list(sensor.offset.value)
            }
        else:
            input_dict["receivers_offset"] = {
                "radar_drape": string_2_list(sensor.offset.value) + [sensor.value.value]
            }

        if topography.options_button.value == "Object":
            if topography.objects.value is None:
                input_dict["topography"] = None
            else:
                input_dict["topography"] = {
                    "GA_object": {
                        "name": topography.objects.value,
                        "data": topography.value.value,
                    }
                }
        elif topography.options_button.value == "Drape Height":
            input_dict["topography"] = {"drapped": topography.offset.value}
        else:
            input_dict["topography"] = {"constant": topography.constant.value}

        if w_l["forward_only"].value:
            input_dict["forward_only"] = []

        checks = [key for key, val in input_dict.items() if val is None]

        if len(list(input_dict["data"]["channels"].keys())) == 0:
            checks += ["'Channel' for at least one data component."]

        if len(checks) > 0:
            print(f"Required value for {checks}")
            w_l["run"].button_style = "danger"
        else:
            w_l["write"].button_style = ""
            file = inv_dir + f"{inversion_parameters.output_name.value}.json"
            with open(file, "w") as f:
                json.dump(input_dict, f, indent=4)
            w_l["run"].button_style = "success"

        w_l["write"].value = False
        w_l["write"].button_style = ""
        w_l["run"].button_style = "success"

    w_l["write"].observe(write_unclick)

    object_observer("")
    update_ref("")

    for attr, item in kwargs.items():
        try:
            if hasattr(w_l[attr], "options") and item not in w_l[attr].options:
                continue
            w_l[attr].value = item
        except KeyError:
            continue
    for attr, item in kwargs.items():
        try:
            if getattr(o_d, attr, None) is not None:
                widget = getattr(o_d, attr)
                if hasattr(widget, "options") and item not in widget.options:
                    continue
                widget.value = item
        except AttributeError:
            continue

    return VBox(
        [
            HBox(
                [
                    VBox(
                        [
                            VBox([Label("Input Data"), o_d.widget]),
                            VBox([Label(""), survey_type_panel]),
                            VBox([Label("Data Components"), data_channel_panel]),
                            VBox([Label("Spatial Information"), spatial_panel]),
                        ]
                    ),
                    selection_panel,
                ]
            ),
            inversion_parameters.widget,
            w_l["forward_only"],
            w_l["write"],
            w_l["run"],
        ]
    )
