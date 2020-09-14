import re
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from geoh5py.objects import Curve, Grid2D, Points, Surface
from ipywidgets import (
    FloatSlider,
    FloatText,
    VBox,
    HBox,
    ToggleButton,
    Label,
    Layout,
)
from geoapps.utils import rotate_xy
from geoapps.selection import ObjectDataSelection
from geoapps.plotting import plot_plan_data_selection


class PlotSelection2D(ObjectDataSelection):
    """
    Application for selecting data in 2D plan map view
    """

    defaults = {
        "h5file": "../../assets/FlinFlon.geoh5",
        "objects": "Gravity_Magnetics_drape60m",
        "data": "Airborne_TMI",
    }

    def __init__(self, **kwargs):

        self.collections = []

        self._azimuth = FloatSlider(
            min=-90,
            max=90,
            value=0,
            steps=5,
            description="Azimuth",
            continuous_update=False,
        )
        self._center_x = FloatSlider(
            min=-100, max=100, steps=10, description="Easting", continuous_update=False,
        )
        self._center_y = FloatSlider(
            min=-100,
            max=100,
            steps=10,
            description="Northing",
            continuous_update=False,
            orientation="vertical",
        )
        self._contours = widgets.Text(
            value="", description="Contours", disabled=False, continuous_update=False,
        )
        self._data_count = Label("Data Count: 0", tooltip="Keep <1500 for speed")
        self._resolution = FloatText(description="Grid Resolution (m)",)
        self._width = FloatSlider(
            min=0,
            max=100,
            steps=10,
            value=1000,
            description="Width",
            continuous_update=False,
        )
        self._height = FloatSlider(
            min=0,
            max=100,
            steps=10,
            value=1000,
            description="Height",
            continuous_update=False,
            orientation="vertical",
        )
        self._zoom_extent = ToggleButton(
            value=True,
            description="Zoom on selection",
            tooltip="Keep plot extent on selection",
            icon="check",
        )

        def set_bounding_box(_):
            self.set_bounding_box()

        self.highlight_selection = None

        def plot_selection(
            data_name,
            resolution,
            center_x,
            center_y,
            width,
            height,
            azimuth,
            zoom_extent,
            contours,
            refresh,
        ):

            self.plot_selection(
                data_name,
                resolution,
                center_x,
                center_y,
                width,
                height,
                azimuth,
                zoom_extent,
                contours,
                refresh,
            )

        super().__init__(**self.apply_defaults(**kwargs))

        self.window_plot = widgets.interactive_output(
            plot_selection,
            {
                "data_name": self.data,
                "resolution": self.resolution,
                "center_x": self.center_x,
                "center_y": self.center_y,
                "width": self.width,
                "height": self.height,
                "azimuth": self.azimuth,
                "zoom_extent": self.zoom_extent,
                "contours": self.contours,
                "refresh": self.refresh,
            },
        )

        self.plot_widget = VBox(
            [
                VBox([self.resolution, self.data_count,]),
                HBox(
                    [
                        self.center_y,
                        self.height,
                        VBox(
                            [
                                self.width,
                                self.center_x,
                                self.window_plot,
                                self.azimuth,
                                self.zoom_extent,
                            ]
                        ),
                    ],
                    layout=Layout(align_items="center"),
                ),
            ]
        )
        self._widget = VBox([self.widget, self.plot_widget])
        self.figure = None
        self.axis = None
        self.indices = None

        self.objects.observe(set_bounding_box, names="value")
        self.set_bounding_box()

    @property
    def azimuth(self):
        """
        :obj:`ipywidgets.FloatSlider`: Rotation angle of the selection box.
        """
        return self._azimuth

    @property
    def center_x(self):
        """
        :obj:`ipywidgets.FloatSlider`: Easting position of the selection box.
        """
        return self._center_x

    @property
    def center_y(self):
        """
        :obj:`ipywidgets.FloatSlider`: Northing position of the selection box.
        """
        return self._center_y

    @property
    def contours(self):
        """
        :obj:`ipywidgets.widgets.Text` String defining sets of contours.
        Contours can be defined over an interval `50:200:10` and/or at a fix value `215`.
        Any combination of the above can be used:
        50:200:10, 215 => Contours between values 50 and 200 every 10, with a contour at 215.
        """
        return self._contours

    @property
    def data_count(self):
        """
        :obj:`ipywidgets.Label`: Data counter included in the selection box.
        """
        return self._data_count

    @property
    def height(self):
        """
        :obj:`ipywidgets.FloatSlider`: Height (m) of the selection box
        """
        return self._height

    @property
    def resolution(self):
        """
        :obj:`ipywidgets.FloatText`: Minimum data separation (m)
        """
        return self._resolution

    @property
    def widget(self):
        """
        :obj:`ipywidgets.VBox`: Application layout
        """
        return self._widget

    @property
    def width(self):
        """
        :obj:`ipywidgets.FloatSlider`: Width (m) of the selection box
        """
        return self._width

    @property
    def zoom_extent(self):
        """
        :obj:`ipywidgets.ToggleButton`: Set plotting limits to the selection box
        """
        return self._zoom_extent

    def plot_selection(
        self,
        data_name,
        resolution,
        center_x,
        center_y,
        width,
        height,
        azimuth,
        zoom_extent,
        contours,
        refresh,
    ):
        if not refresh:
            return

        # Parse the contours string
        if contours != "":
            vals = re.split(",", contours)
            cntrs = []
            for val in vals:
                if ":" in val:
                    param = np.asarray(re.split(":", val), dtype="int")
                    if len(param) == 2:
                        cntrs += [np.arange(param[0], param[1])]
                    else:
                        cntrs += [np.arange(param[0], param[1], param[2])]
                else:
                    cntrs += [np.float(val)]
            contours = np.unique(np.sort(np.hstack(cntrs)))
        else:
            contours = None

        entity, _ = self.get_selected_entities()
        data_obj = None
        if entity.get_data(self.data.value):
            data_obj = entity.get_data(self.data.value)[0]

        if isinstance(entity, (Grid2D, Surface, Points, Curve)):

            self.figure = plt.figure(figsize=(10, 10))
            self.axis = plt.subplot()
            corners = np.r_[
                np.c_[-1.0, -1.0],
                np.c_[-1.0, 1.0],
                np.c_[1.0, 1.0],
                np.c_[1.0, -1.0],
                np.c_[-1.0, -1.0],
            ]
            corners[:, 0] *= width / 2
            corners[:, 1] *= height / 2
            corners = rotate_xy(corners, [0, 0], -azimuth)
            self.axis.plot(corners[:, 0] + center_x, corners[:, 1] + center_y, "k")
            self.axis, _, ind_filter, _, contour_set = plot_plan_data_selection(
                entity,
                data_obj,
                **{
                    "axis": self.axis,
                    "resolution": resolution,
                    "window": {
                        "center": [center_x, center_y],
                        "size": [width, height],
                        "azimuth": azimuth,
                    },
                    "zoom_extent": zoom_extent,
                    "resize": True,
                    "contours": contours,
                    "highlight_selection": self.highlight_selection,
                    "collections": self.collections,
                },
            )

            self.indices = ind_filter
            self.contours.contour_set = contour_set
            self.data_count.value = f"Data Count: {ind_filter.sum()}"

    def set_bounding_box(self):
        # Fetch vertices in the project
        lim_x = [1e8, -1e8]
        lim_y = [1e8, -1e8]

        obj, _ = self.get_selected_entities()
        if isinstance(obj, Grid2D):
            lim_x[0], lim_x[1] = obj.centroids[:, 0].min(), obj.centroids[:, 0].max()
            lim_y[0], lim_y[1] = obj.centroids[:, 1].min(), obj.centroids[:, 1].max()
        elif isinstance(obj, (Points, Curve, Surface)):
            lim_x[0], lim_x[1] = obj.vertices[:, 0].min(), obj.vertices[:, 0].max()
            lim_y[0], lim_y[1] = obj.vertices[:, 1].min(), obj.vertices[:, 1].max()
        else:
            return

        self.refresh.value = False
        self.center_x.min = -np.inf
        self.center_x.max = lim_x[1]
        self.center_x.value = np.mean(lim_x)
        self.center_x.min = lim_x[0]

        self.center_y.min = -np.inf
        self.center_y.max = lim_y[1]
        self.center_y.value = np.mean(lim_y)
        self.center_y.min = lim_y[0]

        self.width.max = lim_x[1] - lim_x[0]
        self.width.value = self.width.max / 2.0
        self.width.min = 0

        self.height.max = lim_y[1] - lim_y[0]
        self.height.min = 0
        self.height.value = self.height.max / 2.0
        self.refresh.value = True
