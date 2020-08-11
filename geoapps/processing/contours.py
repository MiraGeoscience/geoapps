import re
import matplotlib.pyplot as plt
import ipywidgets as widgets
import numpy as np
from geoh5py.objects import Curve, Grid2D, Points, Surface
from ipywidgets.widgets import Label, Layout, VBox
from scipy.interpolate import LinearNDInterpolator
from geoapps.plotting import PlotSelection2D


def contour_values_widget(h5file, **kwargs):
    """
    Application for 2D contouring of spatial data.
    """

    plot_selection = PlotSelection2D(h5file, **kwargs)

    def compute_plot(contour_values):

        entity, data = plot_selection.selection.get_selected_entities()

        if data is None:
            return
        if contour_values is not None:
            plot_selection.contours.value = contour_values

    def save_selection(_):
        entity, _ = plot_selection.selection.get_selected_entities()
        if export.value:

            # TODO
            #  Create temporary workspace and write to trigger LIVE LINK
            # temp_geoh5 = os.path.join(os.path.dirname(
            #     os.path.abspath(workspace.h5file)), "Temp", "temp.geoh5")
            # ws_out = Workspace(temp_geoh5)

            if getattr(plot_selection.contours, "contour_set", None) is not None:
                contour_set = plot_selection.contours.contour_set

                vertices, cells, values = [], [], []
                count = 0
                for segs, level in zip(contour_set.allsegs, contour_set.levels):
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
        if selection.data.value is not None:
            export_as.value = selection.data.value + "_" + contours.value

    contours.observe(updateContours, names="value")

    selection = plot_selection.selection

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
        if selection.data.value is not None:
            export_as.value = selection.data.value + "_" + contours.value
        else:
            export_as.value = contours.value

    selection.data.observe(updateName, names="value")

    z_value = widgets.Checkbox(
        value=False, indent=False, description="Assign Z from values"
    )

    out = widgets.interactive_output(compute_plot, {"contour_values": contours},)

    contours.value = contours.value
    return widgets.VBox(
        [
            widgets.HBox(
                [
                    VBox([Label("Input options:"), plot_selection.widget]),
                    VBox(
                        [
                            Label("Output options:"),
                            contours,
                            export_as,
                            z_value,
                            export,
                        ],
                        layout=Layout(width="50%"),
                    ),
                ]
            ),
            out,
        ]
    )
