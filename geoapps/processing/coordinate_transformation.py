import re
import os
import osr
from ipywidgets import SelectMultiple, VBox, HBox, Text, Textarea
import numpy
from geoh5py.data import FloatData
from geoh5py.objects import Curve, Grid2D, Points, Surface
from geoh5py.workspace import Workspace
from geoapps.base import BaseApplication

from fiona.transform import transform
import gdal
from geoapps.utils import (
    export_grid_2_geotiff,
    geotiff_2_grid,
)


class CoordinateTransformation(BaseApplication):
    """

    """

    defaults = {
        "ga_group_name": "CoordinateTransformation",
        "h5file": "../../assets/FlinFlon.geoh5",
        "objects": ["Gravity_Magnetics_drape60m"],
        "code_in": "EPSG:26914",
        "code_out": "EPSG:26913",
    }

    def __init__(self, **kwargs):

        kwargs = self.apply_defaults(**kwargs)

        self.code_out.observe(self.set_wkt_out, names="value")
        self.code_in.observe(self.set_wkt_in, names="value")
        self.wkt_in.observe(self.set_authority_in, names="value")
        self.wkt_out.observe(self.set_authority_out, names="value")

        super().__init__(**kwargs)

        self._widget = VBox(
            [
                self.project_panel,
                self.objects,
                HBox([self.code_in, self.wkt_in]),
                HBox([self.code_out, self.wkt_out]),
                self.trigger_panel,
            ]
        )

        def trigger_click(_):
            self.trigger_click()

        self.trigger.on_click(trigger_click)

    def trigger_click(self):
        """
        Run the coordinate transformation
        """
        if self.code_in.value != 0 and self.code_out.value != 0:
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
                                temp_file_out, grid, dstSRS=self.wkt_out.value,
                            )

                            if count == 0:
                                grid2d = geotiff_2_grid(
                                    temp_work,
                                    temp_file_out,
                                    grid_name=obj.name
                                    + self.code_out.value.replace(":", "_"),
                                )
                            else:
                                _ = geotiff_2_grid(
                                    temp_work, temp_file_out, grid_object=grid2d
                                )

                            del grid
                            os.remove(temp_file)
                            os.remove(temp_file_out)
                            count += 1

                    grid2d.copy(parent=self.ga_group)
                    os.remove(temp_work.h5file)

                else:
                    if not hasattr(obj, "vertices"):
                        print(f"Skipping {name}. Entity does not have vertices")
                        continue

                    x, y = obj.vertices[:, 0].tolist(), obj.vertices[:, 1].tolist()

                    if self.code_in.value == 4326:
                        x, y = y, x

                    x2, y2 = transform(self.wkt_in.value, self.wkt_out.value, x, y,)

                    if self.code_out.value == 4326:
                        x2, y2 = y2, x2

                    new_obj = obj.copy(parent=self.ga_group, copy_children=True)
                    new_obj.vertices = numpy.c_[x2, y2, obj.vertices[:, 2]]
                    new_obj.name = new_obj.name + self.code_out.value.replace(":", "_")

            if self.live_link.value:
                self.live_link_output(self.ga_group)
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
            self._code_in = Text(description="In code:", continuous_update=False)
        return self._code_in

    @property
    def code_out(self):
        if getattr(self, "_code_out", None) is None:
            self._code_out = Text(description="Output:", continuous_update=False)
        return self._code_out

    @property
    def wkt_in(self):
        if getattr(self, "_wkt_in", None) is None:
            self._wkt_in = Textarea(description="wkt")
        return self._wkt_in

    @property
    def wkt_out(self):
        if getattr(self, "_wkt_out", None) is None:
            self._wkt_out = Textarea(description="wkt")
        return self._wkt_out

    @property
    def widget(self):
        """
        :obj:`ipywidgets.VBox`: Pre-defined application layout
        """
        return self._widget

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

    def update_objects_list(self):
        if getattr(self, "_workspace", None) is not None:

            self.objects.options = [""] + [
                obj.name
                for obj in self._workspace.all_objects()
                if isinstance(obj, self.object_types)
            ]

    def set_wkt_in(self, _):
        datasetSRS = osr.SpatialReference()
        datasetSRS.SetFromUserInput(self.code_in.value.upper())

        self.wkt_in.unobserve_all("value")
        self.wkt_in.value = datasetSRS.ExportToWkt()
        self.wkt_in.observe(self.set_authority_in, names="value")

    def set_authority_in(self, _):
        self.epsg_in.unobserve_all("value")
        self.epsg_in.value = re.findall(
            r'AUTHORITY\["(\D+","\d+)"\]', self.wkt_in.value
        )[-1].replace('","', ":")
        self.epsg_in.observe(self.set_wkt, names="value")

    def set_wkt_out(self, _):
        datasetSRS = osr.SpatialReference()
        datasetSRS.SetFromUserInput(self.code_out.value.upper())

        self.wkt_out.unobserve_all("value")
        self.wkt_out.value = datasetSRS.ExportToWkt()
        self.wkt_out.observe(self.set_authority_out, names="value")

    def set_authority_out(self, _):
        self.epsg_out.unobserve_all("value")
        self.epsg_out.value = re.findall(
            r'AUTHORITY\["(\D+","\d+)"\]', self.wkt_out.value
        )[-1].replace('","', ":")
        self.epsg_out.observe(self.set_wkt, names="value")
