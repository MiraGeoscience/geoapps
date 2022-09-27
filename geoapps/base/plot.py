#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from uuid import UUID

import numpy as np
from geoh5py.objects import Curve, Grid2D, Points, Surface

from geoapps.base.selection import ObjectDataSelection
from geoapps.shared_utils.utils import get_contours, rotate_xyz
from geoapps.utils import warn_module_not_found
from geoapps.utils.plotting import plot_plan_data_selection

with warn_module_not_found():
    from matplotlib import pyplot as plt

with warn_module_not_found():
    import ipywidgets as widgets
    from ipywidgets import (
        FloatSlider,
        FloatText,
        HBox,
        Label,
        Layout,
        ToggleButton,
        VBox,
    )

app_initializer = {
    "geoh5": "../../assets/FlinFlon.geoh5",
    "objects": UUID("{538a7eb1-2218-4bec-98cc-0a759aa0ef4f}"),
    "data": UUID("{44822654-b6ae-45b0-8886-2d845f80f422}"),
}


class PlotSelection2D(ObjectDataSelection):
    """
    Application for selecting data in 2D plan map view
    """

    plot_result = True

    def __init__(self, **kwargs):

        self.defaults.update(**app_initializer)
        self.defaults.update(**kwargs)
        self.axis = None
        self.indices = None
        self.highlight_selection = None
        self.collections = []
        self._window_azimuth = FloatSlider(
            min=-90,
            max=90,
            value=0,
            step=5,
            description="Azimuth",
            continuous_update=False,
        )
        self._window_center_x = FloatSlider(
            min=-100,
            max=100,
            step=10,
            description="Easting",
            continuous_update=False,
        )
        self._window_center_y = FloatSlider(
            min=-100,
            max=100,
            step=10,
            description="Northing",
            continuous_update=False,
            orientation="vertical",
        )
        self._colorbar = widgets.Checkbox(description="Colorbar")
        self._interval_min = FloatText(
            description="Contour min:",
        )
        self._interval_max = FloatText(
            description="Contour max:",
        )
        self._interval_spacing = FloatText(
            description="Contour spacing:",
        )
        self._fixed_contours = widgets.Text(
            value="",
            description="Fixed contours",
            disabled=False,
            continuous_update=False,
        )
        self._data_count = Label("Data Count: 0")
        self._resolution = FloatText(
            description="Grid Resolution (m)", style={"description_width": "initial"}
        )
        self._window_width = FloatSlider(
            min=0,
            max=100,
            step=10,
            value=1000,
            description="Width",
            continuous_update=False,
        )
        self._window_height = FloatSlider(
            min=0,
            max=100,
            step=10,
            value=1000,
            description="Height",
            continuous_update=False,
            orientation="vertical",
        )
        self._zoom_extent = ToggleButton(
            value=True,
            description="Zoom on selection",
            icon="check",
        )
        self.objects.observe(self.set_bounding_box, names="value")
        self.figure = None
        super().__init__(**self.defaults)

        self.window_plot = widgets.interactive_output(
            self.plot_selection,
            {
                "data_name": self.data,
                "resolution": self.resolution,
                "center_x": self.window_center_x,
                "center_y": self.window_center_y,
                "width": self.window_width,
                "height": self.window_height,
                "azimuth": self.window_azimuth,
                "zoom_extent": self.zoom_extent,
                "interval_min": self.interval_min,
                "interval_max": self.interval_max,
                "interval_spacing": self.interval_spacing,
                "fixed_contours": self.fixed_contours,
                "refresh": self.refresh,
                "colorbar": self.colorbar,
            },
        )
        self.window_selection = VBox(
            [
                VBox([self.resolution, self.data_count]),
                HBox(
                    [
                        self.window_center_y,
                        self.window_height,
                        VBox(
                            [
                                self.window_width,
                                self.window_center_x,
                                self.window_plot,
                                self.window_azimuth,
                                HBox([self.zoom_extent, self.colorbar]),
                            ]
                        ),
                    ],
                    layout=Layout(align_items="center"),
                ),
            ]
        )

    @property
    def main(self):
        if self._main is None:
            self._main = VBox(
                [self.project_panel, self.data_panel, self.window_selection]
            )

        return self._main

    @property
    def window_azimuth(self):
        """
        :obj:`ipywidgets.FloatSlider`: Rotation angle of the selection box.
        """
        return self._window_azimuth

    @property
    def window_center_x(self):
        """
        :obj:`ipywidgets.FloatSlider`: Easting position of the selection box.
        """
        return self._window_center_x

    @property
    def window_center_y(self):
        """
        :obj:`ipywidgets.FloatSlider`: Northing position of the selection box.
        """
        return self._window_center_y

    @property
    def colorbar(self):
        """
        :obj:`ipywidgets.widgets.Checkbox` Display the colorbar.
        """
        return self._colorbar

    @property
    def interval_min(self):
        """
        ipywidgets.FloatText(): Minimum value for contours.
        """
        return self._interval_min

    @property
    def interval_max(self):
        """
        ipywidgets.FloatText(): Maximum value for contours.
        """
        return self._interval_max

    @property
    def interval_spacing(self):
        """
        ipywidgets.FloatText(): Step size for contours.
        """
        return self._interval_spacing

    @property
    def fixed_contours(self):
        """
        :obj:`ipywidgets.Text`: String defining sets of fixed_contours.
        """
        return self._fixed_contours

    @property
    def data_count(self):
        """
        :obj:`ipywidgets.Label`: Data counter included in the selection box.
        """
        return self._data_count

    @property
    def window_height(self):
        """
        :obj:`ipywidgets.FloatSlider`: Height (m) of the selection box
        """
        return self._window_height

    @property
    def resolution(self):
        """
        :obj:`ipywidgets.FloatText`: Minimum data separation (m)
        """
        return self._resolution

    @property
    def window_width(self):
        """
        :obj:`ipywidgets.FloatSlider`: Width (m) of the selection box
        """
        return self._window_width

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
        interval_min,
        interval_max,
        interval_spacing,
        fixed_contours,
        refresh,
        colorbar,
    ):
        if not refresh or not self.plot_result:
            return

        # Parse the contours string
        contours = get_contours(
            interval_min, interval_max, interval_spacing, fixed_contours
        )

        entity, _ = self.get_selected_entities()
        if entity is None:
            return
        data_obj = None

        if hasattr(self, "plotting_data"):
            data_channel = self.plotting_data  # pylint: disable=E1101
        else:
            if self.select_multiple and data_name:
                data_channel = data_name[0]
            else:
                data_channel = data_name

        if isinstance(data_channel, str) and (data_channel in "XYZ"):
            data_obj = data_channel
        elif self.workspace.get_entity(data_channel):
            data_obj = self.workspace.get_entity(data_channel)[0]

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
            corners = rotate_xyz(corners, [0, 0], -azimuth)
            self.axis.plot(corners[:, 0] + center_x, corners[:, 1] + center_y, "k")
            self.axis, _, ind_filter, _, _ = plot_plan_data_selection(
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
                    "colorbar": colorbar,
                },
            )
            plt.show()
            self.indices = ind_filter
            self.data_count.value = f"Data Count: {ind_filter.sum()}"

    def set_bounding_box(self, _):
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

        width = lim_x[1] - lim_x[0]
        height = lim_y[1] - lim_y[0]

        self.refresh.value = False
        self.window_center_x.min = -1e8
        self.window_center_x.max = lim_x[1] + width * 0.1
        self.window_center_x.value = np.mean(lim_x)
        self.window_center_x.min = lim_x[0] - width * 0.1

        self.window_center_y.min = -1e8
        self.window_center_y.max = lim_y[1] + height * 0.1
        self.window_center_y.value = np.mean(lim_y)
        self.window_center_y.min = lim_y[0] - height * 0.1

        self.window_width.max = width * 1.2
        self.window_width.value = self.window_width.max
        self.window_width.min = 0

        self.window_height.max = height * 1.2
        self.window_height.min = 0
        self.window_height.value = self.window_height.max
        self.refresh.value = True
