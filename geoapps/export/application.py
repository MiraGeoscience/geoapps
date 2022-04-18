#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import re
import warnings
from os import path

import discretize
import matplotlib.pyplot as plt
import numpy as np
from geoh5py.objects import BlockModel, Curve, Octree
from ipywidgets import Dropdown, FloatText, Layout, RadioButtons, Text, Textarea
from ipywidgets.widgets import HBox, VBox

try:
    from osgeo import osr
except ModuleNotFoundError:
    warnings.warn(
        "Modules 'gdal' is missing from the environment. "
        "Consider installing with: 'conda install -c conda-forge fiona gdal'"
    )

from geoapps.base.selection import ObjectDataSelection
from geoapps.utils.plotting import plot_plan_data_selection
from geoapps.utils.utils import (
    export_curve_2_shapefile,
    export_grid_2_geotiff,
    object_2_dataframe,
    octree_2_treemesh,
)


class Export(ObjectDataSelection):
    """
    General widget to export geoh5 objects to different file formats.
    Currently supported:
    shapefiles: *.shp
    """

    defaults = {
        "h5file": "../../assets/FlinFlon.geoh5",
        "objects": "{538a7eb1-2218-4bec-98cc-0a759aa0ef4f}",
        "data": [
            "{44822654-b6ae-45b0-8886-2d845f80f422}",
            "{53e59b2b-c2ae-4b77-923b-23e06d874e62}",
        ],
        "epsg_code": "EPSG:26914",
        "file_type": "geotiff",
        "data_type": "RGB",
    }

    _select_multiple = True

    def __init__(self, **kwargs):
        self.defaults.update(**kwargs)
        self._file_type = Dropdown(
            options=["ESRI shapefile", "csv", "geotiff", "UBC format"],
            value="csv",
            description="Export type",
        )
        self._data_type = RadioButtons(
            options=[
                "float",
                "RGB",
            ],
            description="Type:",
        )
        self._no_data_value = FloatText(
            description="No-Data-Value",
            value=-99999,
        )
        self._epsg_code = Text(description="Projection:", continuous_update=False)
        self._export_as = Text(description="Save as:", continuous_update=False)
        self._wkt_code = Textarea(
            description="WKT:", continuous_update=False, layout=Layout(width="75%")
        )
        self.epsg_code.observe(self.set_wkt, names="value")
        self.wkt_code.observe(self.set_authority_code, names="value")
        self.type_widget = VBox([self.file_type])
        self.projection_panel = VBox([self.epsg_code, self.wkt_code])
        self.file_type.observe(self.update_options)
        self.objects.observe(self.update_name, names="value")

        super().__init__(**self.defaults)

        self.trigger.description = "Export"
        self.trigger.on_click(self.trigger_click)

    @property
    def wkt_code(self):
        if getattr(self, "_wkt_code", None) is None:
            self._wkt_code = Textarea(description="wkt")
        return self._wkt_code

    @property
    def file_type(self):
        """
        ipywidgets.Dropdown()
        """
        if getattr(self, "_file_type", None) is None:
            self._file_type = Dropdown(
                options=["ESRI shapefile", "csv", "geotiff", "UBC format"],
                value="csv",
                description="Export type",
            )
        return self._file_type

    @property
    def data_type(self):
        """
        ipywidgets.RadioButtons()
        """
        if getattr(self, "_data_type", None) is None:
            self._data_type = RadioButtons(
                options=[
                    "float",
                    "RGB",
                ],
                description="Type:",
            )

        return self._data_type

    @property
    def no_data_value(self):
        """
        ipywidgets.FloatText()
        """
        if getattr(self, "_no_data_value", None) is None:
            self._no_data_value = FloatText(
                description="no-data-value",
                value=-99999,
            )
        return self._no_data_value

    @property
    def epsg_code(self):
        """
        ipywidgets.Text()
        """
        if getattr(self, "_epsg_code", None) is None:
            self._epsg_code = Text(description="EPSG code:", disabled=False)
        return self._epsg_code

    @property
    def export_as(self):
        """
        ipywidgets.Text()
        """
        if getattr(self, "_export_as", None) is None:
            self._export_as = Text(
                value=[
                    key
                    for key, value in self.objects.options
                    if value == self.objects.value
                ][0],
                description="Save as:",
                disabled=False,
            )
        return self._export_as

    @property
    def main(self):
        if self._main is None:
            self._main = VBox(
                [
                    self.project_panel,
                    HBox([self.data_panel, self.no_data_value]),
                    self.type_widget,
                    self.no_data_value,
                    self.export_as,
                    self.trigger,
                    self.export_directory,
                ]
            )
        return self._main

    def trigger_click(self, _):
        if self.workspace.get_entity(self.objects.value):
            entity = self.workspace.get_entity(self.objects.value)[0]
        else:
            return

        if self.data.value:

            data_values = {}

            for key in self.data.value:
                if self.workspace.get_entity(key):
                    data_values[key] = self.workspace.get_entity(key)[0].values.copy()
                    data_values[key][
                        (data_values[key] > 1e-38) * (data_values[key] < 2e-38)
                    ] = self.no_data_value.value
        else:
            data_values = {}

        if self.file_type.value == "csv":
            dataframe = object_2_dataframe(entity, fields=list(data_values.keys()))
            dataframe.to_csv(
                f"{path.join(self.export_directory.selected_path, self.export_as.value)}"
                + ".csv",
                index=False,
            )

        elif self.file_type.value == "ESRI shapefile":

            assert isinstance(
                entity, Curve
            ), f"Only Curve objects are support for type {self.file_type.value}"

            if self.data.value:
                for key in self.data.value:
                    out_name = re.sub(
                        "[^0-9a-zA-Z]+",
                        "_",
                        self.export_as.value + "_" + self.data.uid_name_map[key],
                    )
                    export_curve_2_shapefile(
                        entity,
                        attribute=key,
                        file_name=path.join(
                            self.export_directory.selected_path, out_name
                        ),
                        wkt_code=self.wkt_code.value,
                    )
                    print(
                        f"Object saved to {path.join(self.export_directory.selected_path, out_name) + '.shp'}"
                    )
            else:
                out_name = re.sub("[^0-9a-zA-Z]+", "_", self.export_as.value)
                export_curve_2_shapefile(
                    entity,
                    file_name=path.join(self.export_directory.selected_path, out_name),
                    wkt_code=self.wkt_code.value,
                )
                print(
                    f"Object saved to {path.join(self.export_directory.selected_path, out_name) + '.shp'}"
                )

        elif self.file_type.value == "geotiff":
            for key in self.data.value:
                name = (
                    path.join(self.export_directory.selected_path, self.export_as.value)
                    + "_"
                    + self.data.uid_name_map[key]
                    + ".tif"
                )
                if self.workspace.get_entity(key):
                    export_grid_2_geotiff(
                        self.workspace.get_entity(key)[0],
                        name,
                        wkt_code=self.wkt_code.value,
                        data_type=self.data_type.value,
                    )

                    if self.data_type.value == "RGB":
                        fig, ax = plt.figure(), plt.subplot()

                        if not self.plot_result:
                            plt.gca().set_visible(False)

                        ax, im, _, _, _ = plot_plan_data_selection(
                            entity, self.workspace.get_entity(key)[0], axis=ax
                        )
                        plt.colorbar(im, fraction=0.02)
                        plt.savefig(
                            path.join(
                                self.export_directory.selected_path,
                                self.export_as.value,
                            )
                            + "_"
                            + self.data.uid_name_map[key]
                            + "_Colorbar.png",
                            dpi=300,
                            bbox_inches="tight",
                        )
                        if not self.plot_result:
                            plt.close(fig)

                    print(f"Object saved to {name}")

        elif self.file_type.value == "UBC format":

            assert isinstance(
                entity, (Octree, BlockModel)
            ), "Export available for BlockModel or octree only"
            if isinstance(entity, Octree):
                mesh = octree_2_treemesh(entity)

                models = {}
                for key, item in data_values.items():
                    ind = np.argsort(mesh._ubc_order)
                    models[
                        path.join(
                            self.export_directory.selected_path, self.export_as.value
                        )
                        + "_"
                        + self.data.uid_name_map[key]
                        + ".mod"
                    ] = item[ind]
                name = (
                    path.join(self.export_directory.selected_path, self.export_as.value)
                    + ".msh"
                )
                mesh.writeUBC(
                    name,
                    models=models,
                )
                print(f"Mesh saved to {name}")
                print(f"Models saved to {list(models.keys())}")

            else:

                mesh = discretize.TensorMesh(
                    [
                        np.abs(entity.u_cells),
                        np.abs(entity.v_cells),
                        np.abs(entity.z_cells[::-1]),
                    ]
                )

                # Move the origin to the bottom SW corner
                mesh.x0 = [
                    entity.origin["x"] + entity.u_cells[entity.u_cells < 0].sum(),
                    entity.origin["y"] + entity.v_cells[entity.v_cells < 0].sum(),
                    entity.origin["z"] + entity.z_cells[entity.z_cells < 0].sum(),
                ]
                name = (
                    path.join(self.export_directory.selected_path, self.export_as.value)
                    + ".msh"
                )
                mesh.writeUBC(name)
                print(f"Mesh saved to {name}")

                if any(data_values):
                    for key, item in data_values.items():

                        if mesh.x0[2] == entity.origin["z"]:
                            values = item.copy()
                            values = values.reshape(
                                (mesh.nCz, mesh.nCx, mesh.nCy), order="F"
                            )[::-1, :, :]
                            values = values.reshape((-1, 1), order="F")
                        else:
                            values = item

                        name = (
                            path.join(
                                self.export_directory.selected_path,
                                self.data.uid_name_map[key],
                            )
                            + ".mod"
                        )
                        np.savetxt(name, values)
                        print(f"Model saved to {name}")

    def set_wkt(self, _):
        datasetSRS = osr.SpatialReference()
        datasetSRS.SetFromUserInput(self.epsg_code.value.upper())

        self.wkt_code.unobserve_all("value")
        self.wkt_code.value = datasetSRS.ExportToWkt()
        self.wkt_code.observe(self.set_authority_code, names="value")

    def set_authority_code(self, _):
        self.epsg_code.unobserve_all("value")
        code = re.findall(r'AUTHORITY\["(\D+","\d+)"\]', self.wkt_code.value)
        if code:
            self.epsg_code.value = code[-1].replace('","', ":")
        else:
            self.epsg_code.value = ""
        self.epsg_code.observe(self.set_wkt, names="value")

    def update_name(self, _):
        if self._workspace.get_entity(self.objects.value):
            self.export_as.value = self._workspace.get_entity(self.objects.value)[
                0
            ].name

    def update_options(self, _):
        if self.file_type.value in ["ESRI shapefile"]:
            self.type_widget.children = [self.file_type, self.projection_panel]
        elif self.file_type.value in ["geotiff"]:
            self.type_widget.children = [
                self.file_type,
                self.projection_panel,
                self.data_type,
            ]
        else:
            self.type_widget.children = [self.file_type]
