import json
import os

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from geoh5py.groups import ContainerGroup
from geoh5py.objects import BlockModel, Curve, Grid2D, Octree, Points, Surface
from geoh5py.workspace import Workspace
from ipywidgets.widgets import Dropdown, HBox, Label, Layout, Text, VBox

from .plotting import plot_plan_data_selection
from .utils import find_value, rotate_xy


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


def inversion_widgets():

    widget_list = {
        "objects": Dropdown(description="Object:"),
        "resolution": widgets.FloatText(description="Resolution (m)"),
        "starting_channel": widgets.IntText(value=0, description="Starting Channel"),
        "inducing_field": widgets.Text(description="Inducing Field [Amp, Inc, Dec]"),
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
        "ignore_values": widgets.Text(value="<0", tooltip="Dummy value",),
        "azimuth": widgets.FloatSlider(
            min=-90,
            max=90,
            value=0,
            steps=5,
            description="Orientation",
            continuous_update=False,
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
        "data_count": Label("Data Count: 0", tooltip="Keep <1500 for speed"),
        "core_cell_size": widgets.Text(
            value="25, 25, 25", description="Smallest cells",
        ),
        "max_distance": widgets.FloatText(
            value=1000, description="Max triangulation length",
        ),
        "max_iterations": widgets.IntText(value=10,),
        "octree_levels_topo": widgets.Text(
            value="0, 0, 0, 2", description="Layers below topo",
        ),
        "octree_levels_obs": widgets.Text(
            value="5, 5, 5", description="Layers below data",
        ),
        "depth_core": widgets.FloatText(value=500, description="Minimum depth (m)",),
        "hz_min": widgets.FloatText(value=10.0, description="Smallest cell (m):",),
        "padding_distance": widgets.Text(
            value="0, 0, 0, 0, 0, 0", description="Padding [W,E,N,S,D,U] (m)",
        ),
        "hz_expansion": widgets.FloatText(value=1.05, description="Expansion factor:",),
        "n_cells": widgets.FloatText(value=25.0, description="Number of cells:",),
        "uncert_mode": widgets.RadioButtons(
            options=[
                "Estimated (%|data| + background)",
                r"User input (\%|data| + floor)",
            ],
            value=r"User input (\%|data| + floor)",
            disabled=False,
        ),
        "chi_factor": widgets.FloatText(
            value=1, description="Target misfit", disabled=False
        ),
        "alpha_values": widgets.Text(
            value="1, 1, 1, 1",
            description="Scaling alpha_(s, x, y, z)",
            disabled=False,
            style={"description_width": "initial"},
        ),
        "norms": widgets.Text(
            value="2, 2, 2, 2",
            description="Norms p_(s, x, y, z)",
            disabled=False,
            style={"description_width": "initial"},
        ),
        "lower_bound": widgets.Text(
            value=None,
            description="Lower bound value",
            style={"description_width": "initial"},
        ),
        "upper_bound": widgets.Text(
            value=None,
            description="Upper bound value",
            style={"description_width": "initial"},
        ),
        "run": widgets.ToggleButton(
            value=False, description="Run SimPEG", button_style="danger", icon="check"
        ),
        "write": widgets.ToggleButton(
            value=False,
            description="Write input",
            button_style="warning",
            tooltip="Write json input file",
            icon="check",
        ),
        "forward_only": widgets.Checkbox(
            value=False,
            description="Forward only",
            tooltip="Forward response of reference model",
        ),
        "out_group": widgets.Text(
            value="Inversion_", description="Save to:", disabled=False
        ),
    }

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
    return np.asarray(string.split(",")).astype(float).tolist()


def inversion_widget(h5file, **kwargs):

    w_l = inversion_widgets()
    defaults = inversion_defaults()

    for widget in w_l.values():
        widget.style = {"description_width": "initial"}

    for attr, item in kwargs.items():
        try:
            if hasattr(w_l[attr], "options") and item not in w_l[attr].options:
                continue
            w_l[attr].value = item
        except KeyError:
            continue

    workspace = Workspace(h5file)

    curves = [entity.name for entity in workspace.all_objects()]
    names = [name for name in sorted(curves)]

    all_obj = [
        entity.name
        for entity in workspace.all_objects()
        if isinstance(entity, (Curve, Grid2D, Surface, Points))
    ]

    all_names = [name for name in sorted(all_obj)]

    model_list = []
    for obj in workspace.all_objects():
        if isinstance(obj, (BlockModel, Octree, Surface)):
            for data in obj.children:
                if getattr(data, "values", None) is not None:
                    model_list += [data.name]

    dsep = os.path.sep
    inv_dir = dsep.join(os.path.dirname(os.path.abspath(h5file)).split(dsep))

    if len(inv_dir) > 0:
        inv_dir += dsep
    else:
        inv_dir = os.getcwd() + dsep

    # Load all known em systems
    dir_path = os.path.dirname(os.path.realpath("./functions/AEM_systems.json"))
    with open(os.path.join(dir_path, "AEM_systems.json")) as aem_systems:
        em_system_specs = json.load(aem_systems)

    def run_unclick(_):
        if w_l["run"].value:

            if system.value in ["Gravity", "Magnetics"]:
                os.system(
                    "start cmd.exe @cmd /k "
                    + 'python functions/pf_inversion.py "'
                    + inv_dir
                    + f'\\{w_l["out_group"].value}.json"'
                )
            else:
                os.system(
                    "start cmd.exe @cmd /k "
                    + 'python functions/em1d_inversion.py "'
                    + inv_dir
                    + f'\\{w_l["out_group"].value}.json"'
                )

            w_l["run"].value = False
            w_l["run"].button_style = ""

    w_l["run"].observe(run_unclick)

    def update_options(_):
        w_l["write"].button_style = "warning"
        w_l["run"].button_style = "danger"

    w_l["forward_only"].observe(update_options)

    w_l["chi_factor"].observe(update_options)
    w_l["alpha_values"].observe(update_options)
    w_l["norms"].observe(update_options)
    w_l["lower_bound"].observe(update_options)
    w_l["upper_bound"].observe(update_options)

    bound_panel = VBox([w_l["lower_bound"], w_l["upper_bound"]])

    def update_ref(_):
        alphas = string_2_list(w_l["alpha_values"].value)
        if ref_type.value == "Model":
            ref_mod_panel.children[1].children = [ref_type, ref_mod_list]
            ref_mod_panel.children[1].children[1].layout.visibility = "visible"
            alphas[0] = 1.0

        elif ref_type.value == "Value":
            ref_mod_panel.children[1].children = [ref_type, ref_mod_value]
            ref_mod_panel.children[1].children[1].layout.visibility = "visible"
            alphas[0] = 1.0
        elif ref_type.value == "Best-fitting halfspace":
            ref_mod_panel.children[1].children[1].layout.visibility = "hidden"
            alphas[0] = 1.0
        else:
            ref_mod_panel.children[1].children[1].layout.visibility = "hidden"
            alphas[0] = 0.0

        w_l["alpha_values"].value = ", ".join(list(map(str, alphas)))
        w_l["write"].button_style = "warning"
        w_l["run"].button_style = "danger"

    def update_start(_):
        if start_type.value == "Model":
            start_mod_panel.children[1].children = [start_type, start_mod_list]
            start_mod_panel.children[1].children[1].layout.visibility = "visible"
        elif start_type.value == "Value":
            start_mod_panel.children[1].children = [start_type, start_mod_value]
            start_mod_panel.children[1].children[1].layout.visibility = "visible"
        else:
            start_mod_panel.children[1].children[1].layout.visibility = "hidden"

        w_l["write"].button_style = "warning"
        w_l["run"].button_style = "danger"

    def update_susc(_):

        if susc_type.value == "Model":
            susc_mod_panel.children[1].children = [susc_type, susc_mod_list]
            susc_mod_panel.children[1].children[1].layout.visibility = "visible"
        elif susc_type.value == "Value":
            susc_mod_panel.children[1].children = [susc_type, susc_mod_value]
            susc_mod_panel.children[1].children[1].layout.visibility = "visible"
        else:
            susc_mod_panel.children[1].children[1].layout.visibility = "hidden"

        w_l["write"].button_style = "warning"
        w_l["run"].button_style = "danger"

    ref_type = widgets.RadioButtons(
        options=["Best-fitting halfspace", "Model", "Value"],
        value="Best-fitting halfspace",
        disabled=False,
    )

    ref_type.observe(update_ref)
    ref_mod_list = widgets.Dropdown(description="3D Model", options=model_list,)
    ref_mod_value = widgets.FloatText(description="S/m", value=1e-3,)
    ref_mod_panel = widgets.VBox(
        [Label("Reference"), widgets.VBox([ref_type, ref_mod_value])]
    )

    start_type = widgets.RadioButtons(
        options=["Model", "Value"], value="Value", disabled=False
    )
    start_type.observe(update_start)
    start_mod_value = widgets.FloatText(description="S/m", value=1e-3,)
    start_mod_list = widgets.Dropdown(description="3D Model", options=model_list,)
    start_mod_panel = widgets.VBox(
        [Label("Starting"), widgets.VBox([start_type, start_mod_value])]
    )
    susc_type = widgets.RadioButtons(
        options=["None", "Model", "Value"], value="None", disabled=False
    )

    susc_type.observe(update_susc)
    susc_mod_value = widgets.FloatText(description="SI", value=0.0,)
    susc_mod_list = widgets.Dropdown(description="3D Model", options=model_list,)
    susc_mod_panel = widgets.VBox(
        [Label("Susceptibility model"), widgets.VBox([susc_type])]
    )

    # Mesh parameters
    # OCTREE MESH
    mesh_panel = widgets.VBox(
        [
            Label("Octree Mesh"),
            widgets.VBox(
                [
                    w_l["core_cell_size"],
                    w_l["octree_levels_topo"],
                    w_l["octree_levels_obs"],
                    w_l["depth_core"],
                    w_l["padding_distance"],
                    w_l["max_distance"],
                ]
            ),
        ]
    )

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
    regularization_panel = VBox([ref_mod_panel, w_l["alpha_values"], w_l["norms"]])
    inversion_options = {
        "output name": w_l["out_group"],
        "target misfit": w_l["chi_factor"],
        "uncertainties": w_l["uncert_mode"],
        "starting model": start_mod_panel,
        "background susceptibility": susc_mod_panel,
        "regularization": regularization_panel,
        "upper-lower bounds": bound_panel,
        "mesh": mesh_panel,
        "ignore values (<0 = no negatives)": w_l["ignore_values"],
        "max iterations": w_l["max_iterations"],
    }

    option_choices = widgets.Dropdown(
        options=list(inversion_options.keys()),
        value=list(inversion_options.keys())[0],
        disabled=False,
    )

    def inv_option_change(_):
        inversion_panel.children[1].children = [
            option_choices,
            inversion_options[option_choices.value],
        ]

    option_choices.observe(inv_option_change)

    inversion_panel = widgets.VBox(
        [
            widgets.HBox([widgets.Label("Inversion Options")]),
            widgets.HBox([option_choices, inversion_options[option_choices.value]],),
        ],
        layout=Layout(width="100%"),
    )

    def get_comp_list(entity):
        component_list = []

        for pg in entity.property_groups:
            component_list.append(pg.name)

        return component_list

    def channel_setter(caller):

        channel = caller["owner"]
        data_widget = system.data_channel_options[channel.header]

        entity = workspace.get_entity(w_l["objects"].value)[0]
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
            inversion_options["mesh"] = mesh_panel

            ref_type.options = ["None", "Model", "Value"]
            ref_type.value = "Value"
            flag = system.value

            w_l["ignore_values"].value = "-99999"

        else:
            tx_offsets = em_system_specs[system.value]["tx_offsets"]
            bird_offset.value = ", ".join(
                [str(offset) for offset in em_system_specs[system.value]["bird_offset"]]
            )
            uncertainties = em_system_specs[system.value]["uncertainty"]

            if start_channel is None:
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

            ref_type.options = ["Best-fitting halfspace", "Model", "Value"]
            ref_type.value = "Best-fitting halfspace"

            w_l["lower_bound"].value = "1e-5"
            w_l["upper_bound"].value = "10"
            w_l["ignore_values"].value = "<0"
            # Switch mesh options
            inversion_options["mesh"] = hz_panel
            flag = "EM1D"

        ref_mod_value.description = defaults["units"][flag]
        ref_mod_value.value = defaults["reference_value"][flag]
        ref_mod_panel.children[0].value = "Reference " + defaults["property"][flag]
        start_mod_value.description = defaults["units"][flag]
        start_mod_value.value = defaults["starting_value"][flag]
        start_mod_panel.children[0].value = "Starting " + defaults["property"][flag]

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

            data_channel_options[key] = VBox(
                [
                    widgets.Checkbox(value=False, indent=True, description="Active",),
                    widgets.Text(
                        description=labels[ind], style={"description_width": "initial"},
                    ),
                    channel_selection,
                    widgets.Text(
                        value=", ".join(
                            [str(uncert) for uncert in uncertainties[ind][:2]]
                        ),
                        description="Error (%, floor)",
                        style={"description_width": "initial"},
                    ),
                    widgets.Text(
                        value=", ".join([str(offset) for offset in offsets]),
                        description="Offset",
                        style={"description_width": "initial"},
                    ),
                ]
            )

            if system.value not in ["Magnetics", "Gravity"]:
                data_channel_options[key].children[1].value = channel
            else:
                data_channel_options[key].children[1].layout.visibility = "hidden"
                data_channel_options[key].children[4].layout.visibility = "hidden"

        if len(data_channel_options) > 0:
            data_channel_choices.options = list(data_channel_options.keys())
            data_channel_choices.value = list(data_channel_options.keys())[0]
            system.data_channel_options = data_channel_options
            data_channel_panel.children = [
                data_channel_choices,
                data_channel_options[data_channel_choices.value],
            ]

        update_component_panel("")

        if (
            system.value not in ["Magnetics", "Gravity"]
            and em_system_specs[system.value]["type"] == "frequency"
        ):
            option_choices.options = list(inversion_options.keys())
        else:
            option_choices.options = [
                key
                for key in inversion_options.keys()
                if key != "background susceptibility"
            ]

        w_l["write"].button_style = "warning"
        w_l["run"].button_style = "danger"

        if system.value == "Magnetics":
            survey_type_panel.children = [w_l["objects"], system, w_l["inducing_field"]]
        else:
            survey_type_panel.children = [w_l["objects"], system]

    def object_observer(_):

        w_l["resolution"].indices = None

        if workspace.get_entity(w_l["objects"].value):
            obj = workspace.get_entity(w_l["objects"].value)[0]
            data_list = obj.get_data_list()

            sensor_value.options = [
                name for name in data_list if "visual" not in name.lower()
            ] + ["Vertices"]

            line_field.options = data_list
            line_field.value = find_value(data_list, ["line"])
            line_field_observer("")

            update_topo_list("")
            for aem_system, specs in em_system_specs.items():
                if any([specs["flag"] in channel for channel in data_list]):
                    system.value = aem_system

            if get_comp_list(obj):
                components.options = get_comp_list(obj) + data_list
                components.value = [get_comp_list(obj)[0]]
            else:
                components.options = data_list

            system_observer("")

            if hasattr(system, "data_channel_options"):
                for key, data_widget in system.data_channel_options.items():
                    data_widget.children[2].options = components.options
                    value = find_value(components.options, [key])
                    data_widget.children[2].value = value

            w_l["write"].button_style = "warning"
            w_l["run"].button_style = "danger"

    w_l["objects"].options = all_names
    w_l["objects"].observe(object_observer, names="value")

    systems = ["Magnetics", "Gravity"] + list(em_system_specs.keys())
    system = Dropdown(options=systems, description="Survey Type: ",)
    system.observe(system_observer, names="value")

    def get_data_list(entity):
        groups = [p_g.name for p_g in entity.property_groups]
        data_list = []
        if components.value is not None:
            for component in components.value:
                if component in groups:
                    data_list += [
                        workspace.get_entity(data)[0].name
                        for data in entity.get_property_group(component).properties
                    ]
                elif component in entity.get_data_list():
                    data_list += [component]
        return data_list

    def update_component_panel(_):
        if workspace.get_entity(w_l["objects"].value):
            entity = workspace.get_entity(w_l["objects"].value)[0]
            data_list = get_data_list(entity)

            if hasattr(system, "data_channel_options"):
                for key, data_widget in system.data_channel_options.items():
                    data_widget.children[2].options = data_list
                    value = find_value(data_list, [key])
                    data_widget.children[2].value = value

    components = widgets.SelectMultiple(
        description="Data Channels: ", style={"description_width": "initial"}
    )
    components.observe(update_component_panel, names="value")

    def data_channel_choices_observer(_):
        if hasattr(system, "data_channel_options") and data_channel_choices.value in (
            system.data_channel_options.keys()
        ):
            data_widget = system.data_channel_options[data_channel_choices.value]
            data_channel_panel.children = [data_channel_choices, data_widget]

            if (
                workspace.get_entity(w_l["objects"].value)
                and data_widget.children[2].value is None
            ):
                entity = workspace.get_entity(w_l["objects"].value)[0]
                data_list = get_data_list(entity)
                value = find_value(data_list, [data_channel_choices.value])
                data_widget.children[2].value = value

        w_l["write"].button_style = "warning"
        w_l["run"].button_style = "danger"

    data_channel_choices = widgets.Dropdown(
        description="Data Component:", style={"description_width": "initial"}
    )

    data_channel_choices.observe(data_channel_choices_observer, names="value")
    data_channel_panel = widgets.VBox([data_channel_choices])

    survey_type_panel = VBox([w_l["objects"], system])

    # Spatial parameters
    # Topography definition
    def update_topo_list(_):
        if workspace.get_entity(topo_objects.value):
            obj = workspace.get_entity(topo_objects.value)[0]
            topo_value.options = [
                name for name in obj.get_data_list() if "visual" not in name.lower()
            ] + ["Vertices"]
            topo_value.value = find_value(
                topo_value.options, ["dem", "topo", "dtm"], default="Vertices"
            )

            w_l["write"].button_style = "warning"
            w_l["run"].button_style = "danger"

    def update_topo_options(_):
        topo_options_panel.children = [
            topo_options_button,
            topo_options[topo_options_button.value],
        ]
        w_l["write"].button_style = "warning"
        w_l["run"].button_style = "danger"

    topo_objects = Dropdown(
        options=all_names,
        value=find_value(names, ["topo", "dem", "dtm"]),
        description="Object:",
    )
    topo_objects.observe(update_topo_list, names="value")
    topo_value = Dropdown(description="Channel: ",)
    topo_panel = VBox([topo_objects, topo_value])
    topo_offset = widgets.FloatText(
        value=-30,
        description="Vertical offset (m)",
        style={"description_width": "initial"},
    )
    topo_options = {"Object": topo_panel, "Drape Height": topo_offset}
    topo_options_button = widgets.RadioButtons(
        options=["Object", "Drape Height"], description="Define by:",
    )
    topo_options_button.observe(update_topo_options)
    topo_options_panel = VBox(
        [topo_options_button, topo_options[topo_options_button.value]]
    )
    # Define bird parameters
    sensor_value = Dropdown(description="Channel: ",)

    bird_offset = Text(description="[dx, dy, dz]", value="0, 0, 0")

    sensor_options = {
        "(x,y,z) + offset(x,y,z)": [bird_offset],
        "(x, y, z = topo + constant)": [bird_offset],
        "(x, y, z = topo + radar)": [bird_offset, sensor_value],
    }

    sensor_options_button = widgets.RadioButtons(
        options=[
            "(x,y,z) + offset(x,y,z)",
            "(x, y, z = topo + constant)",
            "(x, y, z = topo + radar)",
        ],
        description="Define by:",
    )

    def update_sensor_options(_):
        if topo_value.value is None:
            sensor_options_button.value = "(x,y,z) + offset(x,y,z)"

        sensor_options_panel.children = [sensor_options_button] + sensor_options[
            sensor_options_button.value
        ]

        w_l["write"].button_style = "warning"
        w_l["run"].button_style = "danger"

    sensor_options_button.observe(update_sensor_options)

    sensor_options_panel = VBox(
        [sensor_options_button] + sensor_options[sensor_options_button.value]
    )

    # LINE ID
    line_field = Dropdown(description="Lines field",)

    def line_field_observer(_):
        if (
            w_l["objects"].value is not None
            and line_field.value is not None
            and "line" in line_field.value.lower()
        ):
            entity = workspace.get_entity(w_l["objects"].value)[0]
            if entity.get_data(line_field.value):
                lines.options = np.unique(entity.get_data(line_field.value)[0].values)

                if lines.options[1]:
                    lines.value = [lines.options[1]]
                if lines.options[0]:
                    lines.value = [lines.options[0]]
            w_l["write"].button_style = "warning"
            w_l["run"].button_style = "danger"

    line_field.observe(line_field_observer)
    lines = widgets.SelectMultiple(description="Select data:",)

    line_id_panel = VBox([line_field, lines])

    # SPATIAL PARAMETERS DROPDOWN
    spatial_options = {
        "Topography": topo_options_panel,
        "Receivers": sensor_options_panel,
        "Line ID": line_id_panel,
    }

    spatial_choices = widgets.Dropdown(
        options=list(spatial_options.keys()),
        value=list(spatial_options.keys())[0],
        disabled=False,
    )

    spatial_panel = VBox(
        [
            Label("Spatial Information"),
            VBox([spatial_choices, spatial_options[spatial_choices.value]]),
        ]
    )

    def spatial_option_change(_):
        spatial_panel.children[1].children = [
            spatial_choices,
            spatial_options[spatial_choices.value],
        ]
        w_l["write"].button_style = "warning"
        w_l["run"].button_style = "danger"

    spatial_choices.observe(spatial_option_change)

    # Data selection and plotting from object extent
    lim_x = [1e8, 0]
    lim_y = [1e8, 0]
    for name in names:
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
        w_l["core_cell_size"].value = f"{dl:.0f}, {dl:.0f}, {dl:.0f}"
        w_l["depth_core"].value = np.ceil(
            np.min([w_l["width_x"].value, w_l["width_y"].value]) / 2.0
        )

        w_l["padding_distance"].value = ", ".join(
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
            if hasattr(system, "data_channel_options") and data_choice in (
                system.data_channel_options.keys()
            ):
                name = system.data_channel_options[data_choice].children[2].value

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

                _, indices, line_selection = plot_plan_data_selection(
                    obj,
                    data_obj,
                    **{
                        "ax": ax1,
                        "highlight_selection": {line_field.value: line_ids},
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
                for widget in system.data_channel_options.values():
                    if system.value in ["Magnetics", "Gravity"]:
                        data_count += widget.children[0].value * indices.sum()
                    else:
                        data_count += widget.children[0].value * line_selection.sum()

                w_l["data_count"].value = f"Data Count: {data_count}"
                w_l["write"].button_style = "warning"
                w_l["run"].button_style = "danger"

    zoom_extent = widgets.ToggleButton(
        value=False,
        description="Zoom on selection",
        tooltip="Keep plot extent on selection",
        icon="check",
    )
    marker_size = widgets.IntSlider(
        min=1,
        max=50,
        value=3,
        description="Markers",
        continuous_update=False,
        orientation="vertical",
    )

    plot_window = widgets.interactive_output(
        plot_selection,
        {
            "entity_name": w_l["objects"],
            "data_choice": data_channel_choices,
            "line_ids": lines,
            "resolution": w_l["resolution"],
            "center_x": w_l["center_x"],
            "center_y": w_l["center_y"],
            "width_x": w_l["width_x"],
            "width_y": w_l["width_y"],
            "azimuth": w_l["azimuth"],
            "zoom_extent": zoom_extent,
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
                    HBox(
                        [w_l["center_y"], w_l["width_y"], plot_window, marker_size],
                        layout=Layout(align_items="center"),
                    ),
                    VBox(
                        [w_l["width_x"], w_l["center_x"], w_l["azimuth"], zoom_extent],
                        layout=Layout(align_items="center"),
                    ),
                ],
                layout=Layout(align_items="center"),
            ),
        ]
    )

    # Inversion options
    def write_unclick(_):
        if w_l["write"].value is False:
            return

        input_dict = {}
        input_dict["out_group"] = w_l["out_group"].value
        input_dict["workspace"] = h5file
        input_dict["save_to_geoh5"] = h5file
        if system.value in ["Gravity", "Magnetics"]:
            input_dict["inversion_type"] = system.value.lower()

            if input_dict["inversion_type"] == "magnetics":
                input_dict["inducing_field_aid"] = string_2_list(
                    w_l["inducing_field"].value
                )
            # Octree mesh parameters
            input_dict["core_cell_size"] = string_2_list(w_l["core_cell_size"].value)
            input_dict["octree_levels_topo"] = string_2_list(
                w_l["octree_levels_topo"].value
            )
            input_dict["octree_levels_obs"] = string_2_list(
                w_l["octree_levels_obs"].value
            )
            input_dict["depth_core"] = {"value": w_l["depth_core"].value}
            input_dict["max_distance"] = w_l["max_distance"].value
            p_d = string_2_list(w_l["padding_distance"].value)
            input_dict["padding_distance"] = [
                [p_d[0], p_d[1]],
                [p_d[2], p_d[3]],
                [p_d[4], p_d[5]],
            ]

        else:
            input_dict["system"] = system.value
            input_dict["lines"] = {
                line_field.value: [str(line) for line in lines.value]
            }

            input_dict["mesh 1D"] = [
                w_l["hz_min"].value,
                w_l["hz_expansion"].value,
                w_l["n_cells"].value,
            ]
        input_dict["chi_factor"] = w_l["chi_factor"].value
        input_dict["max_iterations"] = w_l["max_iterations"].value
        input_dict["ignore_values"] = w_l["ignore_values"].value
        input_dict["resolution"] = w_l["resolution"].value
        input_dict["window"] = {
            "center": [w_l["center_x"].value, w_l["center_y"].value],
            "size": [w_l["width_x"].value, w_l["width_y"].value],
            "azimuth": w_l["azimuth"].value,
        }
        input_dict["alphas"] = string_2_list(w_l["alpha_values"].value)

        input_dict["reference_model"] = {
            ref_mod_panel.children[1]
            .children[0]
            .value.lower(): ref_mod_panel.children[1]
            .children[1]
            .value
        }

        input_dict["starting_model"] = {
            start_mod_panel.children[1]
            .children[0]
            .value.lower(): start_mod_panel.children[1]
            .children[1]
            .value
        }

        if susc_type.value != "None":
            input_dict["susceptibility"] = susc_mod_panel.children[1].children[1].value

        input_dict["model_norms"] = string_2_list(w_l["norms"].value)

        if len(w_l["lower_bound"].value) > 1:
            input_dict["lower_bound"] = string_2_list(w_l["lower_bound"].value)

        if len(w_l["upper_bound"].value) > 1:
            input_dict["upper_bound"] = string_2_list(w_l["upper_bound"].value)

        input_dict["data"] = {}
        input_dict["data"]["type"] = "GA_object"
        input_dict["data"]["name"] = w_l["objects"].value
        if hasattr(system, "data_channel_options"):
            channel_param = {}
            for key, data_widget in system.data_channel_options.items():
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

            input_dict["data"]["channels"] = channel_param

        input_dict["uncertainty_mode"] = w_l["uncert_mode"].value

        if sensor_options_button.value == "(x,y,z) + offset(x,y,z)":
            input_dict["receivers_offset"] = {
                "constant": string_2_list(bird_offset.value)
            }
        elif sensor_options_button.value == "(x, y, z = topo + constant)":
            input_dict["receivers_offset"] = {
                "constant_drape": string_2_list(bird_offset.value)
            }
        else:
            input_dict["receivers_offset"] = {
                "radar_drape": string_2_list(bird_offset.value) + [sensor_value.value]
            }

        if topo_options_button.value == "Object":
            if topo_objects.value is None:
                input_dict["topography"] = None
            else:
                input_dict["topography"] = {
                    "GA_object": {"name": topo_objects.value, "data": topo_value.value}
                }
        else:
            input_dict["topography"] = {"drapped": topo_offset.value}

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
            file = inv_dir + f"{w_l['out_group'].value}.json"
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

    return VBox(
        [
            VBox([survey_type_panel, HBox([components, data_channel_panel])]),
            spatial_panel,
            selection_panel,
            inversion_panel,
            w_l["forward_only"],
            w_l["write"],
            w_l["run"],
        ]
    )
