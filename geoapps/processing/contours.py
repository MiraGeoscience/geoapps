import numpy as np
from geoh5py.workspace import Workspace
from geoh5py.io import H5Writer
from geoh5py.objects import Curve, Grid2D, Points, Surface
from scipy.interpolate import LinearNDInterpolator
from geoapps.base import BaseApplication
from geoapps.plotting import PlotSelection2D
from ipywidgets import (
    Text,
    Checkbox,
    VBox,
    HBox,
    interactive_output,
    Label,
    Layout,
    Widget,
)

defaults = {
    "objects": "Gravity_Magnetics_drape60m",
    "data": "Airborne_TMI",
    "contours": "-400:2000:100,-240",
    "resolution": 50,
}


class ContourValues(BaseApplication):
    """
    Application for 2D contouring of spatial data.
    """

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self._plot_selection = PlotSelection2D(workspace=self.workspace)
        self._objects = self.plot_selection.objects
        self._data = self.plot_selection.data
        self._contours = Text(
            value="", description="Contours", disabled=False, continuous_update=False
        )
        self._export_as = Text(value="Contours", indent=False)

        def update_name(_):
            self.update_name()

        self.plot_selection.selection.data.observe(update_name, names="value")
        self._z_value = Checkbox(
            value=False, indent=False, description="Assign Z from values"
        )

        for key, value in defaults.items():
            try:
                getattr(self, key).value = value
            except:
                pass

        out = interactive_output(self.compute_plot, {"contour_values": self.contours},)

        def save_selection(_):
            self.save_selection()

        self.trigger.observe(save_selection, names="value")
        self.trigger.description = "Export to GA"
        self.trigger.button_style = "danger"

        for key in self.plot_selection.__dict__:
            if isinstance(getattr(self.plot_selection, key, None), Widget):
                getattr(self.plot_selection, key, None).observe(
                    save_selection, names="value"
                )
        self.export_as.observe(save_selection, names="value")
        self._widget = VBox(
            [
                self.project_panel,
                HBox(
                    [
                        VBox(
                            [
                                Label("Input options:"),
                                self.plot_selection.selection.widget,
                                self.contours,
                                self.plot_selection.plot_widget,
                            ]
                        ),
                        VBox(
                            [
                                Label("Output options:"),
                                self.export_as,
                                self.z_value,
                                self.trigger_widget,
                            ],
                            layout=Layout(width="50%"),
                        ),
                    ]
                ),
                out,
            ]
        )

        self.__populate__(**kwargs)

    @property
    def contours(self):
        """
        :obj:`ipywidgets.Text`: String defining sets of contours.
        Contours can be defined over an interval `50:200:10` and/or at a fix value `215`.
        Any combination of the above can be used:
        50:200:10, 215 => Contours between values 50 and 200 every 10, with a contour at 215.
        """
        return self._contours

    @property
    def export(self):
        """
        :obj:`ipywidgets.ToggleButton`: Write contours to the target geoh5
        """
        return self._export

    @property
    def export_as(self):
        """
        :obj:`ipywidgets.Text`: Name given to the Curve object
        """
        return self._export_as

    @property
    def plot_selection(self):
        """
        :obj:`geoapps.selection.PlotSelection2D`: Selection and 2D plot of an object with data to be contoured
        """
        return self._plot_selection

    @property
    def widget(self):
        """
        :obj:`ipywidgets.VBox`: Pre-defined application layout
        """
        return self._widget

    @property
    def z_value(self):
        """
        :obj:`ipywidgets.Checkbox`: Assign z-coordinate based on contour values
        """
        return self._z_value

    def compute_plot(self, contour_values):
        """
        Get current selection and trigger update
        """
        entity, data = self.plot_selection.selection.get_selected_entities()
        if data is None:
            return
        if contour_values is not None:
            self.plot_selection.contours.value = contour_values
        self.save_selection()

    def update_contours(self):
        """
        Assign
        """
        if self.plot_selection.selection.data.value is not None:
            self.export_as.value = (
                self.plot_selection.selection.data.value + "_" + self.contours.value
            )

    def update_name(self):
        if self.plot_selection.selection.data.value is not None:
            self.export_as.value = self.plot_selection.selection.data.value
        else:
            self.export_as.value = "Contours"

    def save_selection(self):
        entity, _ = self.plot_selection.selection.get_selected_entities()

        workspace = Workspace(self.h5file)

        if self.trigger.value:

            if getattr(self.plot_selection.contours, "contour_set", None) is not None:
                contour_set = self.plot_selection.contours.contour_set

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
                    if self.z_value.value:
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

                    if workspace.get_entity(self.export_as.value):

                        curve = workspace.get_entity(self.export_as.value)[0]
                        curve._children = []
                        curve.vertices = vertices
                        curve.cells = np.vstack(cells).astype("uint32")

                        # Remove directly on geoh5
                        project_handle = H5Writer.fetch_h5_handle(self.h5file, entity)
                        base = list(project_handle.keys())[0]
                        obj_handle = project_handle[base]["Objects"]
                        for key in obj_handle[H5Writer.uuid_str(curve.uid)][
                            "Data"
                        ].keys():
                            del project_handle[base]["Data"][key]
                        del obj_handle[H5Writer.uuid_str(curve.uid)]

                    else:
                        curve = Curve.create(
                            workspace,
                            name=self.export_as.value,
                            vertices=vertices,
                            cells=np.vstack(cells).astype("uint32"),
                        )

                    if self.live_link.value:
                        self.live_link_output(
                            curve, data={self.contours.value: np.hstack(values)}
                        )
                        self.trigger.value = False
                    else:
                        curve.add_data(
                            {self.contours.value: {"values": np.hstack(values)}}
                        )
                        workspace = Workspace(self.h5file)
                        self.trigger.value = False
