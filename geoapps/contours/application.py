#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


from time import time

import numpy as np
from geoh5py.groups import ContainerGroup
from geoh5py.objects import Curve, Points, Surface
from geoh5py.ui_json.utils import monitored_directory_copy
from ipywidgets import Checkbox, HBox, Label, Layout, Text, VBox, interactive_output
from matplotlib.pyplot import axes
from scipy.interpolate import LinearNDInterpolator

from geoapps import PlotSelection2D
from geoapps.utils.formatters import string_name
from geoapps.utils.plotting import plot_plan_data_selection
from geoapps.utils.utils import input_string_2_float


class ContourValues(PlotSelection2D):
    """
    Application for 2D contouring of spatial data.
    """

    defaults = {
        "h5file": "../../assets/FlinFlon.geoh5",
        "objects": "{538a7eb1-2218-4bec-98cc-0a759aa0ef4f}",
        "data": "{44822654-b6ae-45b0-8886-2d845f80f422}",
        "contours": "-400:2000:100,-240",
        "resolution": 50,
        "ga_group_name": "Contours",
    }

    def __init__(self, **kwargs):
        self.defaults.update(**kwargs)
        self._contours = Text(
            value="", description="Contours", disabled=False, continuous_update=False
        )
        self._export_as = Text(value="Contours")
        self._z_value = Checkbox(
            value=False, indent=False, description="Assign Z from values"
        )
        self.data.observe(self.update_name, names="value")
        super().__init__(**self.defaults)

        self.selection = interactive_output(
            self.compute_plot,
            {
                "contour_values": self.contours,
            },
        )

        self.trigger.on_click(self.trigger_click)
        self.trigger.description = "Export"
        self.trigger.button_style = "danger"

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
    def z_value(self):
        """
        :obj:`ipywidgets.Checkbox`: Assign z-coordinate based on contour values
        """
        return self._z_value

    @property
    def main(self):
        """
        :obj:`ipywidgets.VBox`: A box containing all widgets forming the application.
        """
        if self._main is None:
            self._main = VBox(
                [
                    self.project_panel,
                    HBox(
                        [
                            VBox(
                                [
                                    Label("Input options:"),
                                    self.data_panel,
                                    self.contours,
                                    self.window_selection,
                                ]
                            ),
                            VBox(
                                [
                                    Label("Save as:"),
                                    self.export_as,
                                    self.z_value,
                                    self.output_panel,
                                ],
                                layout=Layout(width="50%"),
                            ),
                        ]
                    ),
                    self.selection,
                ]
            )
        return self._main

    def compute_plot(self, contour_values):
        """
        Get current selection and trigger update
        """
        entity, data = self.get_selected_entities()
        if data is None:
            return
        if contour_values is not None:
            self.contours.value = contour_values

    def update_contours(self):
        """
        Assign
        """
        if self.data.value is not None:
            self.export_as.value = (
                self.data.uid_name_map[self.data.value] + "_" + self.contours.value
            )

    def update_name(self, _):
        if self.data.value is not None:
            self.export_as.value = self.data.uid_name_map[self.data.value]
        else:
            self.export_as.value = "Contours"

    def trigger_click(self, _):
        entity, data = self.get_selected_entities()

        _, _, _, _, contour_set = plot_plan_data_selection(
            entity,
            data[0],
            **{
                "axis": axes(),
                "resolution": self.resolution.value,
                "window": {
                    "center": [self.window_center_x.value, self.window_center_y.value],
                    "size": [self.window_width.value, self.window_height.value],
                    "azimuth": self.window_azimuth.value,
                },
                "contours": input_string_2_float(self.contours.value),
            },
        )

        if contour_set is not None:
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

            temp_geoh5 = f"{entity.name}_{data[0].name}_{time():.3f}.geoh5"
            with self.get_output_workspace(
                self.export_directory.selected_path, temp_geoh5
            ) as workspace:
                curve = Curve.create(
                    workspace,
                    name=string_name(self.export_as.value),
                    vertices=vertices,
                    cells=np.vstack(cells).astype("uint32"),
                )
                out_entity = curve
                if len(self.ga_group_name.value) > 0:
                    out_entity = ContainerGroup.create(
                        workspace, name=string_name(self.ga_group_name.value)
                    )
                    curve.parent = out_entity

                curve.add_data({self.contours.value: {"values": np.hstack(values)}})

            if self.live_link.value:
                monitored_directory_copy(
                    self.export_directory.selected_path, out_entity
                )
