import re
import os
import time

from ipywidgets import SelectMultiple, VBox, IntText, ToggleButton
import matplotlib.pyplot as plt
import numpy
from geoh5py.data import FloatData
from geoh5py.groups import ContainerGroup
from geoh5py.objects import Curve, Grid2D, Points, Surface
from geoh5py.workspace import Workspace

from geoapps.utils import format_labels
from geoapps.base import BaseApplication

from fiona.transform import transform
import gdal
from geoapps.plotting import plot_plan_data_selection
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
        "epsg_in": 26914,
        "epsg_out": 26913,
    }

    def __init__(self, **kwargs):

        kwargs = self.apply_defaults(**kwargs)
        super().__init__(**kwargs)

        self._widget = VBox(
            [
                self.project_panel,
                self.objects,
                self.epsg_in,
                self.epsg_out,
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
        if self.epsg_in.value != 0 and self.epsg_out.value != 0:
            for name in self.objects.value:
                obj = self.workspace.get_entity(name)[0]
                temp_work = Workspace(self.workspace.name + "temp")
                count = 0
                if isinstance(obj, Grid2D):
                    for child in obj.children:
                        temp_file = child.name + f"_{self.epsg_in.value}.tif"
                        temp_file_out = child.name + ".tif"

                        if isinstance(child, FloatData):

                            export_grid_2_geotiff(
                                child,
                                temp_file,
                                f"{self.epsg_in.value}",
                                data_type="float",
                            )
                            grid = gdal.Open(temp_file)
                            gdal.Warp(
                                temp_file_out,
                                grid,
                                dstSRS=f"EPSG:{self.epsg_out.value}",
                            )

                            if count == 0:
                                grid2d = geotiff_2_grid(
                                    temp_work,
                                    temp_file_out,
                                    grid_name=obj.name + f"_EPSG{self.epsg_out.value}",
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

                    if self.epsg_in.value == 4326:
                        x, y = y, x

                    x2, y2 = transform(
                        f"EPSG:{self.epsg_in.value}",
                        f"EPSG:{self.epsg_out.value}",
                        x,
                        y,
                    )

                    if self.epsg_out.value == 4326:
                        x2, y2 = y2, x2

                    new_obj = obj.copy(parent=self.ga_group, copy_children=True)
                    new_obj.vertices = numpy.c_[x2, y2, obj.vertices[:, 2]]
                    new_obj.name = new_obj.name + f"_EPSG{self.epsg_out.value}"

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
    def epsg_in(self):
        if getattr(self, "_epsg_in", None) is None:
            self._epsg_in = IntText(
                description="EPSG # in:", disabled=False, continuous_update=False
            )
        return self._epsg_in

    @property
    def epsg_out(self):
        if getattr(self, "_epsg_out", None) is None:
            self._epsg_out = IntText(
                description="EPSG # out:", disabled=False, continuous_update=False
            )
        return self._epsg_out

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
