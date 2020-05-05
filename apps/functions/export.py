import numpy as np
import os

import ipywidgets as widgets
from ipywidgets.widgets import VBox, HBox

import discretize
from .geoh5py.workspace import Workspace
from .geoh5py.objects import Curve, BlockModel, Octree
from .utils import export_curve_2_shapefile, export_grid_2_geotiff, octree_2_treemesh
from .selection import object_data_selection_widget
from scipy.spatial import cKDTree


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
                    out_dir + export_as.value + ".tif", epsg_code.value,
                )
                print(f"Object saved to {out_dir + export_as.value}.tif")

            elif file_type.value == 'UBC format':

                assert isinstance(entity, (Octree, BlockModel)), "Export available for BlockModel or Octree only"
                if isinstance(entity, Octree):
                    mesh = octree_2_treemesh(entity)

                    models = {}
                    if entity.get_data(data.value):
                        ind = np.argsort(mesh._ubc_order)
                        data_obj = entity.get_data(data.value)[0]
                        models[out_dir + data_obj.name + ".mod"] = data_obj.values[ind]
                    mesh.writeUBC(out_dir + export_as.value + ".msh", models=models)

                else:
                    mesh = discretize.TensorMesh(
                        [entity.u_cells, entity.v_cells, entity.z_cells]
                    )
                    mesh.x0 = [entity.origin['x'], entity.origin['y'], entity.origin['z']]

                    mesh.writeUBC(out_dir + export_as.value + ".msh")

                    if entity.get_data(data.value):

                        data_obj = entity.get_data(data.value)[0]

                        values = data_obj.values

                        np.savetxt(out_dir + data_obj.name + ".mod", values)
                        # model = np.reshape(values, (mesh.nCx, mesh.nCy, mesh.nCz), order='F')
                        # model = model[:, :, ::-1]
                        # # model = np.transpose(model, (1, 2, 0))
                        #
                        # mesh.writeModelUBC(out_dir + data_obj.name + ".mod", model.ravel())




    def update_options(_):

        if file_type.value in ['ESRI shapefile', "geotiff"]:
            type_widget.children = [file_type, epsg_code]
        else:
            type_widget.children = [file_type]

    file_type = widgets.Dropdown(
        options=['ESRI shapefile', 'xyz', "geotiff", "UBC format"],
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


def mesh_to_mesh_interpolation(h5file):
    def out_update(_):

        if out_mode.value == 'To Object:':

            out_panel.children = [out_mode, mesh_dropdown]

        else:
            out_panel.children = [out_mode, VBox([new_grid, ref_dropdown, core_cell_size, depth_core])]

    def interpolate_call(_):

        if interpolate.value:

            mesh_in = workspace.get_entity(objects.value)[0]
            model_in = workspace.get_entity(data.value)[0]

            if hasattr(mesh_in, "centroids"):
                xyz = mesh_in.centroids
            elif hasattr(mesh_in, "vertices"):
                xyz = mesh_in.vertices

            if out_mode.value == 'To Object:':

                mesh_out = workspace.get_entity(mesh_dropdown.value)[0]

                if hasattr(mesh_out, "centroids"):
                    xyz_out = mesh_out.centroids
                elif hasattr(mesh_out, "vertices"):
                    xyz_out = mesh_out.vertices

            else:

                ref_in = workspace.get_entity(ref_dropdown.value)[0]

                if hasattr(ref_in, "centroids"):
                    xyz_ref = ref_in.centroids
                elif hasattr(ref_in, "vertices"):
                    xyz_ref = ref_in.vertices

                # Find extent of grid
                h = np.asarray(core_cell_size.value.split(",")).astype(float).tolist()

                xmin, xmax = xyz_ref[:, 0].min() - h[0] / 2., xyz_ref[:, 0].max() + h[0] / 2.
                nodal_x = np.arange(xmin, xmax, h[0]) - xmin

                ymin, ymax = xyz_ref[:, 1].min() - h[1] / 2., xyz_ref[:, 1].max() + h[1] / 2.
                nodal_y = np.arange(ymin, ymax, h[1]) - ymin

                zmin, zmax = xyz_ref[:, 2].max() - depth_core.value, xyz_ref[:, 2].max() + h[2] / 2.
                nodal_z = np.arange(zmax, zmin, -h[2]) - zmax

                mesh_out = BlockModel.create(
                    workspace,
                    origin=[xmin, ymin, zmax],
                    u_cell_delimiters=nodal_x,
                    v_cell_delimiters=nodal_y,
                    z_cell_delimiters=nodal_z,
                    name=new_grid.value,
                )

                xyz_out = mesh_out.centroids

            tree = cKDTree(xyz)
            # Find nearest cells
            rad, ind = tree.query(xyz_out)
            mesh_out.add_data({model_in.name + "_interp": {"values": model_in.values[ind]}})

            interpolate.value = False

    workspace = Workspace(h5file)

    names = list(workspace.list_objects_name.values())

    objects, data = object_data_selection_widget(h5file)

    out_mode = widgets.RadioButtons(
        options=['To Object:', 'Re-grid Tensor'],
        value='Re-grid Tensor',
        disabled=False
    )

    out_mode.observe(out_update)
    mesh_dropdown = widgets.Dropdown(
        options=names,
        value=names[0],
    )

    ref_dropdown = widgets.Dropdown(
        options=names,
        value=names[0],
        description="XY Extent from:",
        style={'description_width': 'initial'}
    )

    new_grid = widgets.Text(
        value="InterpGrid",
        description='New grid name:',
        disabled=False,
        style={'description_width': 'initial'}
    )

    core_cell_size = widgets.Text(
        value='25, 25, 25',
        description='Smallest cells',
        disabled=False,
        style={'description_width': 'initial'}
    )

    depth_core = widgets.FloatText(
        value=500,
        description='Core depth (m)',
        disabled=False,
        style={'description_width': 'initial'}
    )

    interpolate = widgets.ToggleButton(
        value=False,
        description='Interpolate',
        icon='check'
    )

    interpolate.observe(interpolate_call)

    out_panel = VBox([out_mode, VBox([new_grid, ref_dropdown, core_cell_size, depth_core])])

    return VBox([objects, data, out_panel, interpolate])