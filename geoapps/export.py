from os import path, mkdir
import re
import osr
import discretize
from ipywidgets import Dropdown, Text, FloatText, RadioButtons, Textarea, Layout
import matplotlib.pyplot as plt
import numpy as np
from geoh5py.objects import BlockModel, Curve, Octree
from geoh5py.workspace import Workspace
from ipywidgets.widgets import HBox, VBox
from geoapps.plotting import plot_plan_data_selection
from geoapps.selection import ObjectDataSelection
from geoapps.utils import (
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
        "select_multiple": True,
        "h5file": "../../assets/FlinFlon.geoh5",
        "objects": "Gravity_Magnetics_drape60m",
        "data": ["Airborne_Gxx"],
        "epsg_code": "EPSG:26914",
        "file_type": "geotiff",
        "data_type": "RGB",
    }

    def __init__(self, **kwargs):
        kwargs = self.apply_defaults(**kwargs)
        self._file_type = Dropdown(
            options=["ESRI shapefile", "csv", "geotiff", "UBC format"],
            value="csv",
            description="Export type",
        )
        self._data_type = RadioButtons(options=["float", "RGB",], description="Type:")
        self._no_data_value = FloatText(
            description="No-Data-Value",
            value=-99999,
            style={"description_width": "initial"},
        )
        self._epsg_code = Text(
            description="Projection:", indent=False, continuous_update=False
        )
        self._export_as = Text(
            description="Save as:", indent=False, continuous_update=False
        )
        self._wkt_code = Textarea(
            description="WKT:", continuous_update=False, layout=Layout(width="75%")
        )
        self.epsg_code.observe(self.set_wkt, names="value")
        self.wkt_code.observe(self.set_authority_code, names="value")

        self.type_widget = VBox([self.file_type])

        def update_options(_):
            self.update_options()

        self.file_type.observe(update_options)

        def update_name(_):
            self.export_as.value = self.objects.value

        self.objects.observe(update_name, names="value")
        super().__init__(**kwargs)
        self.trigger.description = "Export"

        def save_selection(_):
            self.save_selection()

        self.trigger.on_click(save_selection)

        self._widget = VBox(
            [
                self.project_panel,
                self.widget,
                self.type_widget,
                self.no_data_value,
                self.export_as,
                self.trigger,
                self.export_directory,
            ]
        )

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
                options=["float", "RGB",], description="Type:"
            )

        return self._data_type

    @property
    def no_data_value(self):
        """
        ipywidgets.FloatText()
        """
        if getattr(self, "_no_data_value", None) is None:
            self._no_data_value = FloatText(description="no-data-value", value=-99999,)
        return self._no_data_value

    @property
    def epsg_code(self):
        """
        ipywidgets.Text()
        """
        if getattr(self, "_epsg_code", None) is None:
            self._epsg_code = Text(
                description="EPSG code:", indent=False, disabled=False
            )
        return self._epsg_code

    @property
    def export_as(self):
        """
        ipywidgets.Text()
        """
        if getattr(self, "_export_as", None) is None:
            self._export_as = Text(
                value=self.objects.value,
                description="Save as:",
                indent=False,
                disabled=False,
            )
        return self._export_as

    @property
    def workspace(self):
        """
        geoh5py.workspace.Workspace
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

        # Refresh the list of objects
        self.update_objects_list()

        export_path = path.abspath(path.dirname(self.h5file))
        if not path.exists(export_path):
            mkdir(export_path)

        self.export_directory._set_form_values(export_path, "")
        self.export_directory._apply_selection()

    def save_selection(self):
        if self.workspace.get_entity(self.objects.value):
            entity = self.workspace.get_entity(self.objects.value)[0]
        else:
            return

        if self.data.value:

            data_values = {}

            for key in self.data.value:
                if entity.get_data(key):
                    data_values[key] = entity.get_data(key)[0].values.copy()
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
                        "[^0-9a-zA-Z]+", "_", self.export_as.value + "_" + key
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
                    + key
                    + ".tif"
                )
                if entity.get_data(key):
                    export_grid_2_geotiff(
                        entity.get_data(key)[0],
                        name,
                        wkt_code=self.wkt_code.value,
                        data_type=self.data_type.value,
                    )

                    if self.data_type.value == "RGB":
                        fig, ax = plt.figure(), plt.subplot()
                        plt.gca().set_visible(False)
                        ax, im, _, _, _ = plot_plan_data_selection(
                            entity, entity.get_data(key)[0], ax=ax
                        )
                        plt.colorbar(im, fraction=0.02)
                        plt.savefig(
                            path.join(
                                self.export_directory.selected_path,
                                self.export_as.value,
                            )
                            + "_"
                            + key
                            + "_Colorbar.png",
                            dpi=300,
                            bbox_inches="tight",
                        )

                    print(f"Object saved to {name}")

        elif self.file_type.value == "UBC format":

            assert isinstance(
                entity, (Octree, BlockModel)
            ), "Export available for BlockModel or Octree only"
            if isinstance(entity, Octree):
                mesh = octree_2_treemesh(entity)

                models = {}
                for key, item in data_values.items():
                    ind = np.argsort(mesh._ubc_order)

                    data_obj = entity.get_data(key)[0]
                    models[
                        path.join(
                            self.export_directory.selected_path, self.export_as.value
                        )
                        + "_"
                        + key
                        + ".mod"
                    ] = item[ind]
                mesh.writeUBC(
                    path.join(self.export_directory.selected_path, self.export_as.value)
                    + ".msh",
                    models=models,
                )

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

                mesh.writeUBC(
                    path.join(self.export_directory.selected_path, self.export_as.value)
                    + ".msh"
                )

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

                        np.savetxt(
                            path.join(self.export_directory.selected_path, key)
                            + ".mod",
                            values,
                        )

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

    def update_options(self):

        if self.file_type.value in ["ESRI shapefile"]:
            self.type_widget.children = [
                self.file_type,
                VBox([self.epsg_code, self.wkt_code]),
            ]
        elif self.file_type.value in ["geotiff"]:
            self.type_widget.children = [
                self.file_type,
                VBox([VBox([self.epsg_code, self.wkt_code])]),
                self.data_type,
            ]
        else:
            self.type_widget.children = [self.file_type]
