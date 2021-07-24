#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import os
import re

import matplotlib.pyplot as plt
import numpy
from fiona.transform import transform
from geoh5py.data import FloatData
from geoh5py.objects import Curve, Grid2D, Points, Surface
from geoh5py.workspace import Workspace
from ipywidgets import HBox, Layout, SelectMultiple, Text, Textarea, VBox
from osgeo import gdal, osr

from geoapps.plotting import plot_plan_data_selection
from geoapps.selection import ObjectDataSelection
from geoapps.utils.utils import export_grid_2_geotiff, geotiff_2_grid


class CoordinateTransformation(ObjectDataSelection):
    """"""

    defaults = {
        "ga_group_name": "CoordinateTransformation",
        "h5file": "../../assets/FlinFlon.geoh5",
        "objects": [
            "{538a7eb1-2218-4bec-98cc-0a759aa0ef4f}",
            "{bb208abb-dc1f-4820-9ea9-b8883e5ff2c6}",
        ],
        "code_in": "EPSG:26914",
        "code_out": "EPSG:4326",
    }

    def __init__(self, **kwargs):

        self.defaults = self.update_defaults(**kwargs)

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
                self.figure = plt.figure(figsize=(12, 8))
                ax1 = plt.subplot(1, 2, 1)
                ax2 = plt.subplot(1, 2, 2)

            for name in self.objects.value:
                obj = self.workspace.get_entity(name)[0]
                temp_work = Workspace(self.workspace.name + "temp")
                count = 0
                if isinstance(obj, Grid2D):
                    for child in obj.children:
                        temp_file = (
                            child.name + f"_{self.code_in.value.replace(':', '_')}.tif"
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
                                    temp_work,
                                    temp_file_out,
                                    grid_name=obj.name
                                    + self.code_out.value.replace(":", "_"),
                                )
                            else:
                                _ = geotiff_2_grid(
                                    temp_work, temp_file_out, grid=new_obj
                                )

                            del grid
                            if os.path.exists(temp_file):
                                os.remove(temp_file)

                            if os.path.exists(temp_file_out):
                                os.remove(temp_file_out)

                            count += 1

                    new_obj.copy(parent=self.ga_group)
                    os.remove(temp_work.h5file)

                else:
                    if not hasattr(obj, "vertices"):
                        print(f"Skipping {name}. Entity does not have vertices")
                        continue

                    x, y = obj.vertices[:, 0].tolist(), obj.vertices[:, 1].tolist()

                    if self.code_in.value == "EPSG:4326":
                        x, y = y, x

                    x2, y2 = transform(
                        self.wkt_in.value,
                        self.wkt_out.value,
                        x,
                        y,
                    )

                    new_obj = obj.copy(parent=self.ga_group, copy_children=True)
                    new_obj.vertices = numpy.c_[x2, y2, obj.vertices[:, 2]]
                    new_obj.name = new_obj.name + self.code_out.value.replace(":", "_")

                if self.plot_result:
                    plot_plan_data_selection(obj, obj.children[0], axis=ax1)
                    if '"Longitude",EAST' in self.wkt_in.value:
                        ax1.set_xlabel("Longitude")
                        ax1.set_ylabel("Latitude")

                    plot_plan_data_selection(new_obj, new_obj.children[0], axis=ax2)
                    if '"Longitude",EAST' in self.wkt_out.value:
                        ax2.set_xlabel("Longitude")
                        ax2.set_ylabel("Latitude")

            if self.live_link.value:
                self.live_link_output(
                    self.export_directory.selected_path, self.ga_group
                )
            self.workspace.finalize()

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
        if getattr(self, "_code_in", None) is None:
            self._code_in = Text(
                description="Input Projection:", continuous_update=False
            )
        return self._code_in

    @property
    def code_out(self):
        if getattr(self, "_code_out", None) is None:
            self._code_out = Text(
                description="Output Projection:", continuous_update=False
            )
        return self._code_out

    @property
    def wkt_in(self):
        if getattr(self, "_wkt_in", None) is None:
            self._wkt_in = Textarea(description="<=> WKT", layout=Layout(width="50%"))
        return self._wkt_in

    @property
    def wkt_out(self):
        if getattr(self, "_wkt_out", None) is None:
            self._wkt_out = Textarea(description="<=> WKT", layout=Layout(width="50%"))
        return self._wkt_out

    @property
    def workspace(self):
        """
        Target geoh5py workspace
        """
        if (
            getattr(self, "_workspace", None) is None
            and getattr(self, "_h5file", None) is not None
        ):
            self.workspace = Workspace(self.h5file)
        return self._workspace

    @workspace.setter
    def workspace(self, workspace):
        assert isinstance(workspace, Workspace), f"Workspace must of class {Workspace}"
        self._workspace = workspace
        self._h5file = workspace.h5file
        self.update_objects_list()

    def set_wkt_in(self, _):
        datasetSRS = osr.SpatialReference()
        datasetSRS.SetFromUserInput(self.code_in.value.upper())

        self.wkt_in.unobserve_all("value")
        self.wkt_in.value = datasetSRS.ExportToWkt()
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
        datasetSRS = osr.SpatialReference()
        datasetSRS.SetFromUserInput(self.code_out.value.upper())

        self.wkt_out.unobserve_all("value")
        self.wkt_out.value = datasetSRS.ExportToWkt()
        self.wkt_out.observe(self.set_authority_out, names="value")

    def set_authority_out(self, _):
        self.code_out.unobserve_all("value")
        code = re.findall(r'AUTHORITY\["(\D+","\d+)"\]', self.wkt_out.value)
        if code:
            self.code_out.value = code[-1].replace('","', ":")
        else:
            self.code_out.value = ""
        self.code_out.observe(self.set_wkt_out, names="value")
