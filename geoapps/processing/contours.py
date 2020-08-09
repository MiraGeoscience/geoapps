import re

import ipywidgets as widgets
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from geoh5py.objects import Curve, Grid2D, Points, Surface
from geoh5py.workspace import Workspace
from ipywidgets.widgets import HBox, Label, Layout, VBox
from scipy.interpolate import LinearNDInterpolator

from geoapps.plotting import format_labels
from geoapps.selection import object_data_selection_widget


def contour_values_widget(h5file, **kwargs):
    """
    Application for 2D contouring of spatial data.
    """

    workspace = Workspace(h5file)

    def compute_plot(entity_name, data_name, contour_vals):

        entity = workspace.get_entity(entity_name)[0]

        if entity.get_data(data_name):
            data = entity.get_data(data_name)[0]
        else:
            return

        if data.entity_type.color_map is not None:
            new_cmap = data.entity_type.color_map.values
            map_vals = new_cmap["Value"].copy()
            cmap = colors.ListedColormap(
                np.c_[
                    new_cmap["Red"] / 255,
                    new_cmap["Green"] / 255,
                    new_cmap["Blue"] / 255,
                ]
            )
            color_norm = colors.BoundaryNorm(map_vals, cmap.N)

        else:
            cmap = None
            color_norm = None

        if contour_vals != "":
            vals = re.split(",", contour_vals)
            cntrs = []
            for val in vals:
                if ":" in val:
                    param = np.asarray(re.split(":", val), dtype="int")

                    if len(param) == 2:
                        cntrs += [np.arange(param[0], param[1])]
                    else:

                        cntrs += [np.arange(param[0], param[2], param[1])]

                else:
                    cntrs += [np.float(val)]
            contour_vals = np.unique(np.sort(np.hstack(cntrs)))
        else:
            contour_vals = None

        plt.figure(figsize=(10, 10))
        axs = plt.subplot()
        contour_sets = None
        if isinstance(entity, Grid2D):
            xx = entity.centroids[:, 0].reshape(entity.shape, order="F")
            yy = entity.centroids[:, 1].reshape(entity.shape, order="F")
            if len(data.values) == entity.n_cells:
                grid_data = data.values.reshape(xx.shape, order="F")

                axs.pcolormesh(
                    xx, yy, grid_data, cmap=cmap, norm=color_norm, shading="auto"
                )
                format_labels(xx, yy, axs)
                if contour_vals is not None:
                    contour_sets = axs.contour(
                        xx,
                        yy,
                        grid_data,
                        len(contour_vals),
                        levels=contour_vals,
                        colors="k",
                        linewidths=0.5,
                    )

        elif isinstance(entity, (Points, Curve, Surface)):

            if len(data.values) == entity.n_vertices:
                xx = entity.vertices[:, 0]
                yy = entity.vertices[:, 1]
                axs.scatter(xx, yy, 5, data.values, cmap=cmap)
                if contour_vals is not None:
                    contour_sets = axs.tricontour(
                        xx,
                        yy,
                        data.values,
                        levels=contour_vals,
                        linewidths=0.5,
                        colors="k",
                    )
                format_labels(xx, yy, axs)

        else:
            contours.contours = None

        contours.contours = contour_sets

    def save_selection(_):
        if export.value:

            entity = workspace.get_entity(objects.value)[0]
            data_name = data.value

            # TODO
            #  Create temporary workspace and write to trigger LIVE LINK
            # temp_geoh5 = os.path.join(os.path.dirname(
            #     os.path.abspath(workspace.h5file)), "Temp", "temp.geoh5")
            # ws_out = Workspace(temp_geoh5)

            if contours.contours is not None:

                vertices, cells, values = [], [], []
                count = 0
                for segs, level in zip(
                    contours.contours.allsegs, contours.contours.levels
                ):
                    for poly in segs:
                        n_v = len(poly)
                        vertices.append(poly)
                        cells.append(
                            np.c_[
                                np.arange(count, count + n_v - 1),
                                np.arange(count + 1, count + n_v),
                            ]
                        )
                        values.append(np.ones(n_v) * level)

                        count += n_v
                if vertices:
                    vertices = np.vstack(vertices)

                    if z_value.value:
                        vertices = np.c_[vertices, np.hstack(values)]
                    else:

                        if isinstance(entity, (Points, Curve, Surface)):
                            z_interp = LinearNDInterpolator(
                                entity.vertices[:, :2], entity.vertices[:, 2]
                            )
                            vertices = np.c_[vertices, z_interp(vertices)]
                        else:
                            vertices = np.c_[
                                vertices,
                                np.ones(vertices.shape[0]) * entity.origin["z"],
                            ]

                    curve = Curve.create(
                        entity.workspace,
                        name=export_as.value,
                        vertices=vertices,
                        cells=np.vstack(cells).astype("uint32"),
                    )
                    curve.add_data({contours.value: {"values": np.hstack(values)}})

                    # objects.options = list(entity.workspace.list_objects_name.values())
                    # objects.value = entity.name
                    # data.options = entity.get_data_list()
                    # data.value = data_name

                export.value = False

    if "contours" in kwargs.keys():
        contours = kwargs["contours"]
    else:
        contours = ""

    contours = widgets.Text(
        value=contours, description="Contours", disabled=False, continuous_update=False
    )

    def updateContours(_):
        if data.value is not None:
            export_as.value = data.value + "_" + contours.value

    contours.observe(updateContours, names="value")
    contours.contours = None

    objects, data = object_data_selection_widget(h5file)

    if "objects" in kwargs.keys() and kwargs["objects"] in objects.options:
        objects.value = kwargs["objects"]

    if "data" in kwargs.keys() and kwargs["data"] in data.options:
        data.value = kwargs["data"]

    export = widgets.ToggleButton(
        value=False,
        description="Export to GA",
        button_style="danger",
        tooltip="Description",
        icon="check",
    )

    export.observe(save_selection, names="value")

    export_as = widgets.Text(indent=False,)
    updateContours("")

    def updateName(_):
        export_as.value = data.value + "_" + contours.value

    data.observe(updateName, names="value")

    z_value = widgets.Checkbox(
        value=False, indent=False, description="Assign Z from values"
    )

    out = widgets.interactive_output(
        compute_plot,
        {"entity_name": objects, "data_name": data, "contour_vals": contours},
    )

    contours.value = contours.value
    return widgets.VBox(
        [
            widgets.HBox(
                [
                    VBox([Label("Input options:"), objects, data, contours]),
                    VBox(
                        [Label("Output options:"), export_as, z_value, export],
                        layout=Layout(width="50%"),
                    ),
                ]
            ),
            out,
        ]
    )
