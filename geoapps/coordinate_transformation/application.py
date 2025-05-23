# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024-2025 Mira Geoscience Ltd.                                '
#                                                                              '
#  This file is part of geoapps.                                               '
#                                                                              '
#  geoapps is distributed under the terms and conditions of the MIT License    '
#  (see LICENSE file at the root of this source code package).                 '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from __future__ import annotations

import re
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy
from geoh5py.data import FloatData
from geoh5py.groups import ContainerGroup
from geoh5py.objects import Curve, Grid2D, Points, Surface
from geoh5py.ui_json.utils import monitored_directory_copy

from geoapps import assets_path
from geoapps.base.selection import ObjectDataSelection
from geoapps.utils import warn_module_not_found
from geoapps.utils.plotting import plot_plan_data_selection

from ..base.application import BaseApplication


with warn_module_not_found():
    from ipywidgets import HBox, Layout, SelectMultiple, Text, Textarea, VBox

with warn_module_not_found():
    from fiona.transform import transform

from uuid import UUID


with warn_module_not_found():
    from osgeo import gdal, osr

    from geoapps.utils.io import export_grid_2_geotiff

    from .utils import geotiff_2_grid


app_initializer = {
    "ga_group_name": "CoordinateTransformation",
    "geoh5": str(assets_path() / "FlinFlon.geoh5"),
    "objects": [
        UUID("{538a7eb1-2218-4bec-98cc-0a759aa0ef4f}"),
        UUID("{bb208abb-dc1f-4820-9ea9-b8883e5ff2c6}"),
    ],
    "code_in": "EPSG:26914",
    "code_out": "EPSG:4326",
}


class CoordinateTransformation(ObjectDataSelection):
    """Re-project entities between known coordinate systems."""

    def __init__(self, **kwargs):
        self.defaults.update(**app_initializer)
        self.defaults.update(**kwargs)
        self._code_in = Text(description="Input Projection:", continuous_update=False)
        self._code_out = Text(description="Output Projection:", continuous_update=False)
        self._wkt_in = Textarea(description="<=> WKT", layout=Layout(width="50%"))
        self._wkt_out = Textarea(description="<=> WKT", layout=Layout(width="50%"))
        self.defaults.update(**kwargs)
        self.code_out.observe(self.set_wkt_out, names="value")
        self.code_in.observe(self.set_wkt_in, names="value")
        self.wkt_in.observe(self.set_authority_in, names="value")
        self.wkt_out.observe(self.set_authority_out, names="value")

        super().__init__(**self.defaults)

        self.trigger.on_click(self.trigger_click)
        self.input_projection = HBox([self.code_in, self.wkt_in])
        self.output_projection = HBox([self.code_out, self.wkt_out])

    @property
    def main(self):
        if self._main is None:
            self._main = VBox(
                [
                    self.project_panel,
                    self.objects,
                    self.input_projection,
                    self.output_projection,
                    self.output_panel,
                ]
            )

        return self._main

    def trigger_click(self, _):
        """
        Run the coordinate transformation
        """

        if self.wkt_in.value != "" and self.wkt_out.value != "":
            if self.plot_result:
                self._figure = plt.figure(figsize=(12, 8))
                ax1 = plt.subplot(1, 2, 1)
                ax2 = plt.subplot(1, 2, 2)

            temp_geoh5 = f"CoordinateTransformation_{time():.0f}.geoh5"
            ws, self.live_link.value = BaseApplication.get_output_workspace(
                self.live_link.value, self.export_directory.selected_path, temp_geoh5
            )
            with ws as workspace:
                out_entity = ContainerGroup.create(
                    workspace, name=self.ga_group_name.value
                )

                for uid in self.objects.value:
                    obj = self.workspace.get_entity(uid)[0]
                    input_copy = obj.copy(parent=ws)

                    if isinstance(input_copy, Grid2D):
                        count = 0
                        for child in input_copy.children:
                            temp_file = (
                                child.name
                                + f"_{self.code_in.value.replace(':', '_')}.tif"
                            )
                            temp_file_out = child.name + ".tif"

                            if isinstance(child, FloatData):
                                export_grid_2_geotiff(
                                    child,
                                    temp_file,
                                    wkt_code=self.wkt_in.value,
                                    data_type="float",
                                )
                                grid = gdal.Open(temp_file)
                                gdal.Warp(
                                    temp_file_out,
                                    grid,
                                    dstSRS=self.wkt_out.value,
                                )

                                if count == 0:
                                    new_obj = geotiff_2_grid(
                                        workspace,
                                        temp_file_out,
                                        grid_name=input_copy.name
                                        + self.code_out.value.replace(":", "_"),
                                        parent=out_entity,
                                    )

                                else:
                                    _ = geotiff_2_grid(
                                        workspace, temp_file_out, grid=new_obj
                                    )

                                del grid
                                Path(temp_file).unlink(missing_ok=True)
                                Path(temp_file_out).unlink(missing_ok=True)
                                count += 1
                    else:
                        if not hasattr(input_copy, "vertices"):
                            print(
                                f"Skipping {input_copy.name}. Entity does not have vertices"
                            )
                            continue

                        x, y = (
                            input_copy.vertices[:, 0].tolist(),
                            input_copy.vertices[:, 1].tolist(),
                        )

                        if self.code_in.value == "EPSG:4326":
                            x, y = y, x

                        x2, y2 = transform(
                            self.wkt_in.value,
                            self.wkt_out.value,
                            x,
                            y,
                        )

                        new_obj = input_copy.copy(
                            parent=out_entity,
                            vertices=numpy.c_[x2, y2, input_copy.vertices[:, 2]],
                            name=(
                                input_copy.name + self.code_out.value.replace(":", "_")
                            ),
                        )

                    if self.plot_result:
                        plot_plan_data_selection(
                            input_copy, input_copy.children[0], axis=ax1
                        )
                        if '"Longitude",EAST' in self.wkt_in.value:
                            ax1.set_xlabel("Longitude")
                            ax1.set_ylabel("Latitude")

                        plot_plan_data_selection(new_obj, new_obj.children[0], axis=ax2)
                        if '"Longitude",EAST' in self.wkt_out.value:
                            ax2.set_xlabel("Longitude")
                            ax2.set_ylabel("Latitude")

            if self.live_link.value:
                monitored_directory_copy(
                    self.export_directory.selected_path, out_entity
                )

    @property
    def object_types(self):
        if getattr(self, "_object_types", None) is None:
            self._object_types = (Curve, Grid2D, Points, Surface)
        return self._object_types

    @property
    def objects(self):
        if getattr(self, "_objects", None) is None:
            self._objects = SelectMultiple(description="Object:")
        return self._objects

    @property
    def code_in(self):
        """Input EPSG or ESRI code."""
        return self._code_in

    @property
    def code_out(self):
        """
        Output EPSG or ESRI code.
        """
        return self._code_out

    @property
    def wkt_in(self):
        """Input Well-Known-Text (WKT) string."""
        return self._wkt_in

    @property
    def wkt_out(self):
        """Output Well-Known-Text (WKT) string."""
        return self._wkt_out

    def set_wkt_in(self, _):
        dataset_SRS = osr.SpatialReference()
        dataset_SRS.SetFromUserInput(self.code_in.value.upper())

        self.wkt_in.unobserve_all("value")
        self.wkt_in.value = dataset_SRS.ExportToWkt()
        self.wkt_in.observe(self.set_authority_in, names="value")

    def set_authority_in(self, _):
        self.code_in.unobserve_all("value")

        code = re.findall(r'AUTHORITY\["(\D+","\d+)"\]', self.wkt_in.value)
        if code:
            self.code_in.value = code[-1].replace('","', ":")
        else:
            self.code_in.value = ""
        self.code_in.observe(self.set_wkt_in, names="value")

    def set_wkt_out(self, _):
        dataset_SRS = osr.SpatialReference()
        dataset_SRS.SetFromUserInput(self.code_out.value.upper())

        self.wkt_out.unobserve_all("value")
        self.wkt_out.value = dataset_SRS.ExportToWkt()
        self.wkt_out.observe(self.set_authority_out, names="value")

    def set_authority_out(self, _):
        self.code_out.unobserve_all("value")
        code = re.findall(r'AUTHORITY\["(\D+","\d+)"\]', self.wkt_out.value)
        if code:
            self.code_out.value = code[-1].replace('","', ":")
        else:
            self.code_out.value = ""
        self.code_out.observe(self.set_wkt_out, names="value")
