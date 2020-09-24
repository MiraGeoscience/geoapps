import numpy as np
from matplotlib import collections
from geoh5py.workspace import Workspace
from geoh5py.io import H5Writer
from geoh5py.objects import Grid2D, Curve
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line
from ipywidgets import (
    FloatSlider,
    HBox,
    IntSlider,
    Layout,
    ToggleButton,
    Text,
    VBox,
    interactive_output,
)
from geoapps.utils import filter_xy
from geoapps.plotting import PlotSelection2D


class EdgeDetectionApp(PlotSelection2D):
    """
    Widget for Grid2D objects for the automated detection of line features.
    The application relies on the Canny and Hough trandforms from the
    Scikit-Image library.

    :param grid: Grid2D object
    :param data: Children data object for the provided grid

    Optional
    --------

    :param sigma [Canny]: standard deviation of the Gaussian filter
    :param threshold [Hough]: Value threshold
    :param line_length [Hough]: Minimum accepted pixel length of detected lines
    :param line_gap [Hough]: Maximum gap between pixels to still form a line.
    """

    defaults = {
        "h5file": "../../assets/FlinFlon.geoh5",
        "objects": "Gravity_Magnetics_drape60m",
        "data": "Airborne_Gxx",
        "resolution": 50,
        "sigma": 0.5,
        "compute": True,
        "window": {"azimuth": -20,},
        "ga_group_name": "Edges",
    }
    object_types = (Grid2D,)

    def __init__(self, **kwargs):

        self._compute = ToggleButton(
            value=False,
            description="Compute",
            button_style="warning",
            tooltip="Description",
            icon="check",
        )
        self._export_as = Text(value="Edges", description="Save as:", disabled=False,)
        self._line_length = IntSlider(
            min=1,
            max=100,
            step=1,
            value=1,
            continuous_update=False,
            description="Line Length",
        )
        self._line_gap = IntSlider(
            min=1,
            max=100,
            step=1,
            value=1,
            continuous_update=False,
            description="Line Gap",
        )
        self._sigma = FloatSlider(
            min=0.0,
            max=10,
            step=0.1,
            value=1.0,
            continuous_update=False,
            description="Sigma",
        )
        self._threshold = IntSlider(
            min=1,
            max=100,
            step=1,
            value=1,
            continuous_update=False,
            description="Threshold",
        )
        self._window_size = IntSlider(
            min=16,
            max=512,
            value=64,
            continuous_update=False,
            description="Window size",
        )
        super().__init__(**kwargs)

        out = interactive_output(self.compute_trigger, {"compute": self.compute},)

        def save_trigger(_):
            self.save_trigger()

        # Make changes to trigger warning color
        self.trigger.description = "Save to GA"
        self.trigger.on_click(save_trigger)
        self.trigger.button_style = "success"

        def update_name(_):
            self.update_name()

        self.data.observe(update_name, names="value")
        self.update_name()

        self._widget = VBox(
            [
                self.project_panel,
                HBox(
                    [
                        VBox([self.widget]),
                        VBox(
                            [
                                self.sigma,
                                self.threshold,
                                self.line_length,
                                self.line_gap,
                                self.window_size,
                                self.compute,
                                self.export_as,
                                self.trigger_panel,
                            ],
                            layout=Layout(width="50%"),
                        ),
                        out,
                    ]
                ),
            ]
        )

    @property
    def compute(self):
        """ToggleButton"""
        return self._compute

    @property
    def export_as(self):
        """Text"""
        return self._export_as

    @property
    def line_length(self):
        """IntSlider"""
        return self._line_length

    @property
    def line_gap(self):
        """IntSlider"""
        return self._line_gap

    @property
    def sigma(self):
        """FloatSlider"""
        return self._sigma

    @property
    def threshold(self):
        """IntSlider"""
        return self._threshold

    @property
    def window_size(self):
        """IntSlider"""
        return self._window_size

    @property
    def widget(self):
        """
        :obj:`ipywidgets.VBox`: Pre-defined application layout
        """
        return self._widget

    def save_trigger(self):
        entity, _ = self.get_selected_entities()
        if getattr(self.trigger, "vertices", None) is not None:

            curves = [
                child
                for child in self.ga_group.children
                if child.name == self.export_as.value
            ]
            if any(curves):
                curve = curves[0]

                curve._children = []
                curve.vertices = self.trigger.vertices
                curve.cells = np.vstack(self.trigger.cells).astype("uint32")

                # Remove directly on geoh5
                project_handle = H5Writer.fetch_h5_handle(self.h5file, entity)
                base = list(project_handle.keys())[0]
                obj_handle = project_handle[base]["Objects"]
                for key in obj_handle[H5Writer.uuid_str(curve.uid)]["Data"].keys():
                    del project_handle[base]["Data"][key]
                del obj_handle[H5Writer.uuid_str(curve.uid)]

                H5Writer.save_entity(curve)

            else:
                curve = Curve.create(
                    self.workspace,
                    name=self.export_as.value,
                    vertices=self.trigger.vertices,
                    cells=self.trigger.cells,
                    parent=self.ga_group,
                )

            if self.live_link.value:
                self.live_link_output(curve)

    def update_name(self):
        if self.data.value is not None:
            self.export_as.value = self.data.value
        else:
            self.export_as.value = "Edges"

    def compute_trigger(self, compute):
        if compute:

            grid, data = self.get_selected_entities()

            x = grid.centroids[:, 0].reshape(grid.shape, order="F")
            y = grid.centroids[:, 1].reshape(grid.shape, order="F")
            z = grid.centroids[:, 2].reshape(grid.shape, order="F")
            grid_data = data.values.reshape(grid.shape, order="F")
            indices = self.indices
            ind_x, ind_y = (
                np.any(indices, axis=1),
                np.any(indices, axis=0),
            )
            x = x[ind_x, :][:, ind_y]
            y = y[ind_x, :][:, ind_y]
            z = z[ind_x, :][:, ind_y]
            grid_data = grid_data[ind_x, :][:, ind_y]
            grid_data -= grid_data.min()
            grid_data /= grid_data.max()

            if np.any(grid_data):
                # Find edges
                edges = canny(grid_data, sigma=self.sigma.value, use_quantiles=True)
                shape = edges.shape
                # Cycle through tiles of square size
                max_l = np.min([self.window_size.value, shape[0], shape[1]])
                half = np.floor(max_l / 2)
                overlap = 1.25

                n_cell_y = (shape[0] - 2 * half) * overlap / max_l
                n_cell_x = (shape[1] - 2 * half) * overlap / max_l

                if n_cell_x > 0:
                    cnt_x = np.linspace(
                        half, shape[1] - half, 2 + int(np.round(n_cell_x)), dtype=int
                    ).tolist()
                    half_x = half
                else:
                    cnt_x = [np.ceil(shape[1] / 2)]
                    half_x = np.ceil(shape[1] / 2)

                if n_cell_y > 0:
                    cnt_y = np.linspace(
                        half, shape[0] - half, 2 + int(np.round(n_cell_y)), dtype=int
                    ).tolist()
                    half_y = half
                else:
                    cnt_y = [np.ceil(shape[0] / 2)]
                    half_y = np.ceil(shape[0] / 2)

                coords = []
                for cx in cnt_x:
                    for cy in cnt_y:

                        i_min, i_max = int(cy - half_y), int(cy + half_y)
                        j_min, j_max = int(cx - half_x), int(cx + half_x)
                        lines = probabilistic_hough_line(
                            edges[i_min:i_max, j_min:j_max],
                            line_length=self.line_length.value,
                            threshold=self.threshold.value,
                            line_gap=self.line_gap.value,
                            seed=0,
                        )

                        if np.any(lines):
                            coord = np.vstack(lines)
                            coords.append(
                                np.c_[
                                    x[i_min:i_max, j_min:j_max][
                                        coord[:, 1], coord[:, 0]
                                    ],
                                    y[i_min:i_max, j_min:j_max][
                                        coord[:, 1], coord[:, 0]
                                    ],
                                    z[i_min:i_max, j_min:j_max][
                                        coord[:, 1], coord[:, 0]
                                    ],
                                ]
                            )
                if coords:
                    coord = np.vstack(coords)
                    self.objects.lines = coord
                    self.plot_store_lines()
                else:
                    self.objects.lines = None

            self.compute.value = False

    def plot_store_lines(self):

        xy = self.objects.lines
        indices_1 = filter_xy(
            xy[1::2, 0],
            xy[1::2, 1],
            self.resolution.value,
            window={
                "center": [self.center_x.value, self.center_y.value,],
                "size": [self.width.value, self.height.value,],
                "azimuth": self.azimuth.value,
            },
        )
        indices_2 = filter_xy(
            xy[::2, 0],
            xy[::2, 1],
            self.resolution.value,
            window={
                "center": [self.center_x.value, self.center_y.value,],
                "size": [self.width.value, self.height.value,],
                "azimuth": self.azimuth.value,
            },
        )

        indices = np.kron(
            np.any(np.c_[indices_1, indices_2], axis=1), np.ones(2),
        ).astype(bool)

        xy = self.objects.lines[indices, :2]
        self.collections = [
            collections.LineCollection(
                np.reshape(xy, (-1, 2, 2)), colors="k", linewidths=2
            )
        ]
        self.refresh.value = False
        self.refresh.value = True  # Trigger refresh

        if np.any(xy):
            vertices = np.vstack(self.objects.lines[indices, :])
            cells = np.arange(vertices.shape[0]).astype("uint32").reshape((-1, 2))
            if np.any(cells):
                self.trigger.vertices = vertices
                self.trigger.cells = cells
        else:
            self.trigger.vertices = None
            self.trigger.cells = None
