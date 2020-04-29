import numpy as np
import os

import ipywidgets as widgets
from ipywidgets.widgets import VBox, HBox

from .geoh5py.workspace import Workspace
from .geoh5py.objects import Curve
from .utils import export_curve_2_shapefile, export_grid_2_geotiff
from .selection import object_data_selection_widget


def export_widget(h5file):
    """
    General widget to export geoh5 objects to different file formats.
    Currently supported:
    shapefiles: *.shp
    """
    workspace = Workspace(h5file)

    dsep = os.path.sep
    out_dir = dsep.join(
        os.path.dirname(os.path.abspath(h5file)).split(dsep)
    ) + os.path.sep

    def save_selection(_):
        if export.value:

            export.value = False  # Reset button

            entity = workspace.get_entity(objects.value)[0]


            if entity.get_data(data.value):
                data_values = entity.get_data(data.value)[0].values
            else:
                data_values = []

            values = []
            if file_type.value == 'xyz':
                if getattr(entity, 'vertices', None) is not None:
                    values = entity.vertices
                elif getattr(entity, 'centroids', None) is not None:
                    values = entity.centroids

                if len(values) > 0:
                    if any(data_values):
                        values = np.c_[values, data_values]

                    np.savetxt(f"{out_dir + export_as.value}.xyz", values)
                    print(f"Object saved to {out_dir + export_as.value}.xyz")
                else:
                    print("Could not find spatial data to export")

            elif file_type.value == 'ESRI shapefile':

                assert isinstance(entity, Curve), f"Only Curve objects are support for type {file_type.value}"
                export_curve_2_shapefile(
                    entity, attribute=data.value,
                    file_name=out_dir + export_as.value,
                    epsg=epsg_code.value
                )
                print(f"Object saved to {out_dir + export_as.value}.shp")

            elif file_type.value == 'geotiff' and entity.get_data(data.value):
                export_grid_2_geotiff(
                    entity.get_data(data.value)[0],
                    out_dir + export_as.value, epsg_code.value,
                )
                print(f"Object saved to {out_dir + export_as.value}.tif")


    def update_options(_):

        if file_type.value in ['ESRI shapefile', "geotiff"]:
            type_widget.children = [file_type, epsg_code]
        else:
            type_widget.children = [file_type]

    file_type = widgets.Dropdown(
        options=['ESRI shapefile', 'xyz', "geotiff"],
        value='xyz',
        description='Export type',
    )

    epsg_code = widgets.Text(
        description="EPSG code:",
        indent=False,
        disabled=False
    )

    type_widget = HBox([file_type])

    file_type.observe(update_options)

    objects, data = object_data_selection_widget(h5file)

    def update_name(_):
        export_as.value = objects.value.replace(":", "_")

    objects.observe(update_name, names='value')
    export = widgets.ToggleButton(
        value=False,
        description='Export',
        button_style='danger',
        tooltip='Description',
        icon='check'
    )

    export.observe(save_selection, names='value')

    export_as = widgets.Text(
        value=objects.value,
        description="Save as:",
        indent=False,
        disabled=False
    )
    return VBox([objects, data, type_widget, export_as, export])
