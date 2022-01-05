#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import discretize
import ipywidgets as widgets
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from geoh5py.groups import ContainerGroup
from geoh5py.objects import Surface
from geoh5py.workspace import Workspace

from geoapps.selection import ObjectDataSelection
from geoapps.utils.utils import RectangularBlock, tensor_2_block_model


def block_model_widget(h5file, inducing_field="50000, 90, 0"):
    workspace = Workspace(h5file)
    names = list(workspace.list_objects_name.values())

    def set_axes_equal(ax):
        """
        Source:
        https://stackoverflow.com/questions/13685386/

        Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  This is one possible solution to Matplotlib's
        ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

        Input
          ax: a matplotlib axis, e.g., as output from plt.gca().
        """

        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5 * max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    def plot_layout(objects, data, blocks, dip, azimuth, update):

        if workspace.get_entity(objects):
            obj = workspace.get_entity(objects)[0]
        else:
            return

        if obj.get_data(data):
            obs = obj.get_data(data)[0]
        else:
            obs = None

        plt.figure(figsize=(10, 12))
        axs = plt.subplot(projection="3d")
        axs.view_init(dip, ((450 - azimuth) % 360) + 180)
        if getattr(obj, "vertices", None) is not None:

            if isinstance(getattr(obs, "values", None), np.float):
                values = obs.values
            else:
                values = None

            axs.scatter(
                obj.vertices[:, 0],
                obj.vertices[:, 1],
                obj.vertices[:, 2],
                s=4,
                c=values,
                cmap="Spectral_r",
            )
        elif getattr(obj, "centroids", None) is not None:

            if isinstance(obs.values[0], np.float):
                values = obs.values
            else:
                values = None

            axs.scatter(
                obj.centroids[:, 0],
                obj.centroids[:, 1],
                obj.centroids[:, 2],
                s=4,
                c=values,
                cmap="Spectral_r",
            )

        for block in blocks:
            param = blocks_widgets[block].children
            block = RectangularBlock(
                dip=param[0].value,
                azimuth=param[1].value,
                center=[param[2].value, param[3].value, param[4].value],
                length=param[5].value,
                width=param[6].value,
                depth=param[7].value,
            )
            axs.plot_trisurf(
                block.vertices[:, 0],
                block.vertices[:, 1],
                block.vertices[:, 2],
                triangles=block.triangles,
            )

        set_axes_equal(axs)

    object_selection = ObjectDataSelection(h5file=h5file, interactive=True)

    xyz = []
    for obj_name in object_selection.objects.options:
        if workspace.get_entity(obj_name):
            obj = workspace.get_entity(obj_name)[0]
            if getattr(obj, "vertices", None) is not None:
                xyz += [obj.vertices]
            else:
                xyz += [obj.centroids]

    xyz = np.vstack(xyz)

    azimuth = widgets.FloatSlider(
        min=-180,
        max=180,
        value=0,
        step=5,
        description="Camera azimuth",
        continuous_update=False,
    )

    dip = widgets.FloatSlider(
        min=-90,
        max=90,
        value=15,
        step=5,
        description="Camera dip",
        continuous_update=False,
    )

    # Pre-build a list of blocks
    update = widgets.ToggleButton(value=False)

    def update_view(_):
        if update.value:
            update.value = False
        else:
            update.value = True

    blocks_widgets = {}
    for ii in range(10):
        blocks_widgets[ii + 1] = widgets.VBox(
            [
                widgets.FloatSlider(
                    min=-90,
                    max=90,
                    value=0,
                    description="Dip",
                    continuous_update=False,
                ),
                widgets.FloatSlider(
                    min=-180,
                    max=180,
                    value=0,
                    description="Strike",
                    continuous_update=False,
                ),
                widgets.FloatSlider(
                    min=xyz[:, 0].min(),
                    max=xyz[:, 0].max(),
                    value=np.mean(xyz[:, 0]),
                    description="X center:",
                    continuous_update=False,
                ),
                widgets.FloatSlider(
                    min=xyz[:, 1].min(),
                    max=xyz[:, 1].max(),
                    value=np.mean(xyz[:, 1]),
                    description="Y center:",
                    continuous_update=False,
                ),
                widgets.FloatSlider(
                    min=-1000,
                    max=xyz[:, 2].max(),
                    value=np.mean(xyz[:, 2]) - 500,
                    description="Z center:",
                    continuous_update=False,
                ),
                widgets.FloatSlider(
                    min=10,
                    max=10000,
                    value=1000,
                    description="Length:",
                    continuous_update=False,
                ),
                widgets.FloatSlider(
                    min=10,
                    max=10000,
                    value=1000,
                    description="Width:",
                    continuous_update=False,
                ),
                widgets.FloatSlider(
                    min=10,
                    max=10000,
                    value=1000,
                    description="Depth:",
                    continuous_update=False,
                ),
                widgets.FloatSlider(
                    min=0,
                    max=1.0,
                    value=0.01,
                    step=0.001,
                    description="Susceptibility:",
                    continuous_update=False,
                ),
            ]
        )

        for child in blocks_widgets[ii + 1].children:
            child.observe(update_view)

    block_add = widgets.SelectMultiple(
        options=list(blocks_widgets.keys()), description="Select blocks"
    )

    def block_add_update(_):
        block_list.options = block_add.value

    block_add.observe(block_add_update)

    block_list = widgets.Dropdown(
        options=[],
    )

    block_panel = widgets.VBox(
        [
            block_list,
        ]
    )

    def block_param_update(_):
        if block_list.value is not None:
            block_panel.children = [block_list, blocks_widgets[block_list.value]]
        else:
            block_panel.children = [
                block_list,
            ]

    block_list.observe(block_param_update)

    interactive_plot = widgets.interactive_output(
        plot_layout,
        {
            "objects": object_selection.objects,
            "data": object_selection.data,
            "blocks": block_add,
            "azimuth": azimuth,
            "dip": dip,
            "update": update,
        },
    )

    sim_name = widgets.Text(
        value="Simulation_",
        description="Name:",
        disabled=False,
    )

    core_cell_size = widgets.Text(
        value="100, 100, 100",
        description="Model discretization (m)",
        disabled=False,
    )

    topography = widgets.Dropdown(
        options=[None] + names,
        description="Topography",
    )

    # Pre-build a list of blocks
    export_ga = widgets.ToggleButton(value=False, description="Export", icon="check")

    def export_trigger(_):
        if getattr(export_ga, "data", None) is not None and export_ga.value:

            # Create a group
            out_group = ContainerGroup.create(workspace, name=sim_name.value)

            obj = workspace.get_entity(object_selection.objects.value)[0]
            obj_out = obj.copy(parent=out_group)

            # Export data on Points
            nC = len(components.value)

            for ind, comp in enumerate(components.value):
                obj_out.add_data(
                    {sim_name.value + "_" + comp: {"values": export_ga.data[ind::nC]}}
                )

            # Export mesh and model
            tensor_2_block_model(
                workspace,
                export_ga.mesh,
                name=sim_name.value + "mesh",
                parent=out_group,
                data={sim_name.value + "_model": export_ga.model},
            )

            # Export blocks as surfaces
            for block in export_ga.blocks:
                Surface.create(
                    workspace,
                    vertices=block.vertices,
                    cells=block.triangles,
                    parent=out_group,
                )
            export_ga.value = False

    export_ga.observe(export_trigger)

    plot_now = widgets.ToggleButton(
        value=True, description="Plot simulation", icon="check"
    )

    def run_simulation(_):

        if forward.value:
            obj = workspace.get_entity(object_selection.objects.value)[0]

            if getattr(obj, "vertices", None) is not None:
                xyz = obj.vertices
            else:
                xyz = obj.centroids

            nodes = []
            out_blocks = []
            for block in block_add.value:
                param = blocks_widgets[block].children
                block = RectangularBlock(
                    dip=param[0].value,
                    azimuth=param[1].value,
                    center=[param[2].value, param[3].value, param[4].value],
                    length=param[5].value,
                    width=param[6].value,
                    depth=param[7].value,
                    susc=param[8].value,
                )
                out_blocks.append(block)
                nodes.append(block.vertices)

            h = np.asarray(core_cell_size.value.split(",")).astype(float).tolist()
            paddings = [[h[0], h[0]], [h[1], h[1]], [h[2], h[2]]]

            # Create a mesh
            mesh = discretize.utils.meshutils.mesh_builder_xyz(
                np.vstack(nodes), h, padding_distance=paddings
            )

            # Create a model
            model = np.zeros(mesh.nC)
            for block in out_blocks:
                ind = Utils.ModelBuilder.PolygonInd(mesh, block.vertices)
                model[ind] = block.susc

            # Cut air cells
            if topography.value is not None:
                topo = workspace.get_entity(topography.value)[0]
                ind = Utils.modelutils.surface2ind_topo(mesh, topo.vertices)
                model[ind == False] = -1

            # Create a problem
            active = model > 0
            nC = int(active.sum())

            if survey_type.value == "Magnetics":
                rxLoc = PF.BaseMag.RxObs(xyz)
                srcField = PF.BaseMag.SrcField(
                    [rxLoc],
                    param=np.asarray(inducing_field.value.split(","))
                    .astype(float)
                    .tolist(),
                )
                survey = PF.BaseMag.LinearSurvey(srcField, components=components.value)
                prob = PF.Magnetics.MagneticIntegral(
                    mesh,
                    chiMap=Maps.IdentityMap(nP=nC),
                    actInd=active,
                    forwardOnly=True,
                    verbose=False,
                )
                prob.pair(survey)
                d = prob.fields(model[active])

            else:
                rxLoc = PF.BaseGrav.RxObs(xyz)
                srcField = PF.BaseGrav.SrcField([rxLoc])
                survey = PF.BaseGrav.LinearSurvey(srcField, components=components.value)
                prob = PF.Gravity.GravityIntegral(
                    mesh,
                    rhoMap=Maps.IdentityMap(nP=nC),
                    actInd=active,
                    forwardOnly=True,
                    verbose=False,
                )
                prob.pair(survey)
                d = prob.fields(model[active])

            # Save latest simulation to widget for export
            export_ga.data = d
            export_ga.mesh = mesh
            export_ga.model = model
            export_ga.blocks = out_blocks
        forward.value = False

    # Pre-build a list of blocks
    forward = widgets.ToggleButton(value=False, description="Run forward", icon="check")
    forward.observe(run_simulation)

    forward_panel = widgets.VBox(
        [
            widgets.Label("FORWARD SIMULATION"),
            topography,
            core_cell_size,
            sim_name,
            forward,
            plot_now,
            export_ga,
        ]
    )

    def update_data_options(_):
        if survey_type.value == "Magnetics":
            components.options = ["tmi", "bxx", "bxy", "bxz", "byy", "byz", "bzz"]
            components.value = ["tmi"]
            phys_prop = "Susceptibility SI"
            ranges = [0.0, 1.0]
        else:
            components.options = ["gz", "gxx", "gxy", "gxz", "gyy", "gyz", "gzz"]
            components.value = ["gz"]
            phys_prop = "Residual Density (g/cc)"
            ranges = [-1.0, 1.0]

        for widget in blocks_widgets.values():
            widget.children[-1].description = phys_prop
            widget.children[-1].min = ranges[0]
            widget.children[-1].max = ranges[1]

    components = widgets.SelectMultiple(
        description="Components",
        options=["tmi", "bxx", "bxy", "bxz", "byy", "byz", "bzz"],
        value=["tmi"],
    )

    components_panel = widgets.VBox([components])

    def update_survey_type(_):
        if survey_type.value == "Magnetics":
            survey_type_panel.children = [
                survey_type,
                object_selection.main,
                components_panel,
                inducing_field,
            ]
        else:
            survey_type_panel.children = [
                survey_type,
                object_selection.main,
                components_panel,
            ]

        update_data_options("")

    survey_type = widgets.Dropdown(
        options=["Magnetics", "Gravity"],
        value="Magnetics",
        description="Survey Type:",
    )
    inducing_field = widgets.Text(
        value=inducing_field,
        description="Inducing Field [Amp, Inc, Dec]",
    )
    survey_type_panel = widgets.VBox(
        [survey_type, object_selection.main, components_panel, inducing_field]
    )
    survey_type.observe(update_survey_type)

    def plot_simulation(plot, run):

        objects = object_selection.objects.value
        data = object_selection.data.value

        if plot:
            if workspace.get_entity(objects):
                obj = workspace.get_entity(objects)[0]
                if getattr(obj, "vertices", None) is not None:
                    xyz = obj.vertices
                else:
                    xyz = obj.centroids

                if obj.get_data(data):
                    obs = obj.get_data(data)[0]
                else:
                    return
            else:
                return

            nC = len(components.value)

            if (
                getattr(export_ga, "data", None) is not None
                and export_ga.data.shape[0] == nC * xyz.shape[0]
            ):
                trian = tri.Triangulation(xyz[:, 0], xyz[:, 1])

                plt.figure(figsize=(8, 4 * int(np.ceil(nC / 2) + 1)))
                axs = plt.subplot(int(np.ceil(nC / 2) + 1), 2, 1)
                if isinstance(obs.values[0], np.float):
                    im = axs.tricontourf(
                        trian,
                        obs.values,
                        cmap="Spectral_r",
                        levels=100,
                        vmin=obs.values.min(),
                        vmax=obs.values.max(),
                    )
                    plt.colorbar(im)
                    axs.set_title(data)
                    axs.set_yticklabels([])
                    axs.set_xticklabels([])
                    axs.set_aspect("equal")

                for ind, comp in enumerate(components.value):
                    axs = plt.subplot(int(np.ceil(nC / 2) + 1), 2, ind + 2)
                    im = axs.tricontourf(
                        trian,
                        export_ga.data[ind::nC],
                        cmap="Spectral_r",
                        levels=100,
                        vmin=obs.values.min(),
                        vmax=obs.values.max(),
                    )
                    plt.colorbar(im)
                    axs.set_yticklabels([])
                    axs.set_xticklabels([])
                    axs.set_title(comp)
                    axs.set_aspect("equal")

    return widgets.VBox(
        [
            widgets.HBox(
                [
                    widgets.VBox([survey_type_panel, forward_panel]),
                    widgets.VBox([block_add, block_panel]),
                ]
            ),
            widgets.HBox(
                [
                    widgets.VBox(
                        [
                            azimuth,
                            dip,
                            interactive_plot,
                        ]
                    ),
                    widgets.interactive_output(
                        plot_simulation, {"plot": plot_now, "run": forward}
                    ),
                ]
            ),
        ]
    )
