import os
import re
import discretize
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from geoh5py.objects import BlockModel, Curve, Octree
from geoh5py.workspace import Workspace
from ipywidgets.widgets import HBox, VBox
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree

from geoapps.plotting import plot_plan_data_selection
from geoapps.selection import ObjectDataSelection
from geoapps.inversion import TopographyOptions
from geoapps.utils import (
    export_curve_2_shapefile,
    export_grid_2_geotiff,
    object_2_dataframe,
    octree_2_treemesh,
)


def export_widget(h5file):
    """
    General widget to export geoh5 objects to different file formats.
    Currently supported:
    shapefiles: *.shp
    """
    workspace = Workspace(h5file)

    dsep = os.path.sep
    out_dir = (
        dsep.join(os.path.dirname(os.path.abspath(h5file)).split(dsep)) + os.path.sep
    )

    def save_selection(_):
        if export.value:

            export.value = False  # Reset button

            entity = workspace.get_entity(selection.objects.value)[0]

            if selection.data.value:

                data_values = {}

                for key in selection.data.value:
                    if entity.get_data(key):
                        data_values[key] = entity.get_data(key)[0].values.copy()
                        data_values[key][
                            (data_values[key] > 1e-38) * (data_values[key] < 2e-38)
                        ] = no_data_value.value
            else:
                data_values = {}

            if file_type.value == "csv":
                dataframe = object_2_dataframe(entity, fields=list(data_values.keys()))
                dataframe.to_csv(f"{out_dir + export_as.value}" + ".csv", index=False)

            elif file_type.value == "ESRI shapefile":

                assert isinstance(
                    entity, Curve
                ), f"Only Curve objects are support for type {file_type.value}"

                if selection.data.value:
                    for key in selection.data.value:
                        out_name = re.sub(
                            "[^0-9a-zA-Z]+", "_", export_as.value + "_" + key
                        )
                        export_curve_2_shapefile(
                            entity,
                            attribute=key,
                            file_name=out_dir + out_name,
                            epsg=epsg_code.value,
                        )
                        print(f"Object saved to {out_dir + out_name + '.shp'}")
                else:
                    out_name = re.sub("[^0-9a-zA-Z]+", "_", export_as.value)
                    export_curve_2_shapefile(
                        entity, file_name=out_dir + out_name, epsg=epsg_code.value,
                    )
                    print(f"Object saved to {out_dir + out_name + '.shp'}")

            elif file_type.value == "geotiff":

                for key in selection.data.value:
                    name = out_dir + export_as.value + "_" + key + ".tif"
                    if entity.get_data(key):

                        export_grid_2_geotiff(
                            entity.get_data(key)[0],
                            name,
                            epsg_code.value,
                            data_type=data_type.value,
                        )

                        if data_type.value == "RGB":
                            fig, ax = plt.figure(), plt.subplot()
                            plt.gca().set_visible(False)
                            (ax, im), _, _, _ = plot_plan_data_selection(
                                entity, entity.get_data(key)[0], ax=ax
                            )
                            plt.colorbar(im, fraction=0.02)
                            plt.savefig(
                                out_dir + export_as.value + "_" + key + "_Colorbar.png",
                                dpi=300,
                                bbox_inches="tight",
                            )

                        print(f"Object saved to {name}")

            elif file_type.value == "UBC format":

                assert isinstance(
                    entity, (Octree, BlockModel)
                ), "Export available for BlockModel or Octree only"
                if isinstance(entity, Octree):
                    mesh = octree_2_treemesh(entity)

                    models = {}
                    for key, item in data_values.items():
                        ind = np.argsort(mesh._ubc_order)

                        data_obj = entity.get_data(key)[0]
                        models[out_dir + data_obj.name + "_" + key + ".mod"] = item[ind]
                    mesh.writeUBC(out_dir + export_as.value + ".msh", models=models)

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

                    mesh.writeUBC(out_dir + export_as.value + ".msh")

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

                            np.savetxt(out_dir + key + ".mod", values)

    def update_options(_):

        if file_type.value in ["ESRI shapefile"]:
            type_widget.children = [file_type, epsg_code]
        elif file_type.value in ["geotiff"]:
            type_widget.children = [file_type, VBox([epsg_code, data_type])]
        else:
            type_widget.children = [file_type]

    file_type = widgets.Dropdown(
        options=["ESRI shapefile", "csv", "geotiff", "UBC format"],
        value="csv",
        description="Export type",
    )

    data_type = widgets.RadioButtons(options=["float", "RGB",], description="Type:")
    no_data_value = widgets.FloatText(description="no-data-value", value=-99999,)

    epsg_code = widgets.Text(description="EPSG code:", indent=False, disabled=False)

    type_widget = HBox([file_type])

    file_type.observe(update_options)

    selection = ObjectDataSelection(h5file=h5file, select_multiple=True)

    def update_name(_):
        export_as.value = selection.objects.value.replace(":", "_")

    selection.objects.observe(update_name, names="value")
    export = widgets.ToggleButton(
        value=False,
        description="Export",
        button_style="danger",
        tooltip="Description",
        icon="check",
    )

    export.observe(save_selection, names="value")

    export_as = widgets.Text(
        value=selection.objects.value,
        description="Save as:",
        indent=False,
        disabled=False,
    )
    return VBox(
        [HBox([selection.widget, no_data_value]), type_widget, export_as, export]
    )
