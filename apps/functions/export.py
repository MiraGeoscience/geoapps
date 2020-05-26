import numpy as np
import os

import ipywidgets as widgets
from ipywidgets.widgets import VBox, HBox

import discretize
from .geoh5py.workspace import Workspace
from .geoh5py.objects import Curve, BlockModel, Octree
from .utils import (
    export_curve_2_shapefile, export_grid_2_geotiff,
    octree_2_treemesh, object_2_dataframe
)
from .selection import object_data_selection_widget
from scipy.spatial import cKDTree
from scipy.interpolate import LinearNDInterpolator


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

            if data.value:

                data_values = {}

                for key in data.value:
                    if entity.get_data(key):
                        data_values[key] = entity.get_data(key)[0].values.copy()
                        data_values[key][
                            (data_values[key] > 1e-38) * (data_values[key] < 2e-38)
                        ] = no_data_value.value
            else:
                data_values = {}

            if file_type.value == 'csv':
                dataframe = object_2_dataframe(entity, fields=list(data_values.keys()))
                dataframe.to_csv(f"{out_dir + export_as.value}" + ".csv", index=False)

            elif file_type.value == 'ESRI shapefile':

                assert isinstance(entity, Curve), f"Only Curve objects are support for type {file_type.value}"

                for key, item in data_values.items():
                    export_curve_2_shapefile(
                        entity, attribute=item,
                        file_name=out_dir + export_as.value + "_" + key,
                        epsg=epsg_code.value
                    )
                    print(f"Object saved to {out_dir + export_as.value}.shp")

            elif file_type.value == 'geotiff':

                for key in data.value:
                    name = out_dir + export_as.value + "_" + key + ".tif"
                    if entity.get_data(key):

                        export_grid_2_geotiff(
                            entity.get_data(key)[0],
                            name,
                            epsg_code.value,
                        )
                        print(f"Object saved to {name}")


            elif file_type.value == 'UBC format':

                assert isinstance(entity, (Octree, BlockModel)), "Export available for BlockModel or Octree only"
                if isinstance(entity, Octree):
                    mesh = octree_2_treemesh(entity)

                    models = {}
                    for key, item in data_values.items():
                        ind = np.argsort(mesh._ubc_order)
                        data_obj = entity.get_data(data.value)[0]
                        models[out_dir + data_obj.name + "_" + key + ".mod"] = item[ind]
                    mesh.writeUBC(out_dir + export_as.value + ".msh", models=models)

                else:
                    mesh = discretize.TensorMesh(
                        [
                            entity.u_cells,
                            entity.v_cells,
                            entity.z_cells
                        ]
                    )

                    # Move the origin to the bottom SW corner
                    mesh.x0 = [
                        entity.origin['x'] + entity.u_cells[entity.u_cells < 0].sum(),
                        entity.origin['y'] + entity.v_cells[entity.v_cells < 0].sum(),
                        entity.origin['z'] + entity.z_cells[entity.z_cells < 0].sum()
                    ]

                    mesh.writeUBC(out_dir + export_as.value + ".msh")

                    if any(data_values):
                        for key, item in data_values.items():

                            if mesh.x0[2] == entity.origin['z']:
                                values = item.copy()
                                values = values.reshape(
                                    (mesh.nCz, mesh.nCx, mesh.nCy), order='F'
                                )[::-1, :, :]
                                values = values.reshape((-1, 1), order='F')
                            else:
                                values = item

                            np.savetxt(
                                out_dir + key + ".mod",
                                values
                            )




    def update_options(_):

        if file_type.value in ['ESRI shapefile', "geotiff"]:
            type_widget.children = [file_type, epsg_code]
        else:
            type_widget.children = [file_type]

    file_type = widgets.Dropdown(
        options=['ESRI shapefile', 'csv', "geotiff", "UBC format"],
        value='csv',
        description='Export type',
    )

    no_data_value = widgets.FloatText(
        description="no-data-value",
        value=-99999,
        style={'description_width': 'initial'}
    )

    epsg_code = widgets.Text(
        description="EPSG code:",
        indent=False,
        disabled=False
    )

    type_widget = HBox([file_type])

    file_type.observe(update_options)

    objects, data = object_data_selection_widget(
        h5file, select_multiple=True
    )

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
    return VBox([objects, HBox([data, no_data_value]), type_widget, export_as, export])


def object_to_object_interpolation(h5file):
    def out_update(_):

        if out_mode.value == 'To Object:':

            out_panel.children = [out_mode, mesh_dropdown]

        else:
            out_panel.children = [
                out_mode, new_grid_panel
            ]

    def interpolate_call(_):

        if interpolate.value:

            mesh_in = workspace.get_entity(objects.value)[0]

            if hasattr(mesh_in, "centroids"):
                xyz = mesh_in.centroids.copy()
            elif hasattr(mesh_in, "vertices"):
                xyz = mesh_in.vertices.copy()

            # Create a tree for the input mesh
            tree = cKDTree(xyz)

            if out_mode.value == 'To Object:':

                mesh_out = workspace.get_entity(mesh_dropdown.value)[0]

                if hasattr(mesh_out, "centroids"):
                    xyz_out = mesh_out.centroids.copy()
                elif hasattr(mesh_out, "vertices"):
                    xyz_out = mesh_out.vertices.copy()

            else:

                ref_in = workspace.get_entity(ref_dropdown.value)[0]

                if hasattr(ref_in, "centroids"):
                    xyz_ref = ref_in.centroids
                elif hasattr(ref_in, "vertices"):
                    xyz_ref = ref_in.vertices


                # Find extent of grid
                h = np.asarray(core_cell_size.value.split(",")).astype(float).tolist()

                pads = np.asarray(padding_distance.value.split(",")).astype(float).tolist()

                # Use discretize to build a tensor mesh
                mesh = discretize.utils.meshutils.mesh_builder_xyz(
                    xyz_ref, h,
                    padding_distance=[
                        [pads[0], pads[1]],
                        [pads[2], pads[3]],
                        [pads[4], pads[5]]
                    ],
                    depth_core=depth_core.value,
                    expansion_factor=expansion_fact.value,
                )

                mesh_out = BlockModel.create(
                    workspace,
                    origin=[mesh.x0[0], mesh.x0[1], mesh.x0[2]],
                    u_cell_delimiters=mesh.vectorNx - mesh.x0[0],
                    v_cell_delimiters=mesh.vectorNy - mesh.x0[1],
                    z_cell_delimiters=(mesh.vectorNz - mesh.x0[2]),
                    name=new_grid.value,
                )

                # Try to recenter on nearest
                # Find nearest cells
                rad, ind = tree.query(mesh_out.centroids)
                ind_nn = np.argmin(rad)

                d_xyz = mesh_out.centroids[ind_nn, :] - xyz[ind[ind_nn], :]

                mesh_out.origin = np.r_[mesh_out.origin.tolist()] - d_xyz

                xyz_out = mesh_out.centroids.copy()

            values = {}
            for field in data.value:
                model_in = mesh_in.get_data(field)[0]
                values[field] = model_in.values.copy()
                if space.value == 'Log':
                    values[field] = np.log(values[field])

            values_interp = {}
            if method.value == 'Linear':

                for key, value in values.items():
                    F = LinearNDInterpolator(xyz, value)
                    values_interp[key] = F(xyz_out)

            elif method.value == 'Inverse Distance':

                ang = np.deg2rad((450. - np.asarray(skew_angle.value)) % 360.)

                R = np.r_[
                    np.c_[np.cos(ang), np.sin(ang)],
                    np.c_[-np.sin(ang), np.cos(ang)]
                ]

                center = np.mean(xyz, axis=0).reshape((3, 1))
                xyz -= np.kron(center, np.ones(xyz.shape[0])).T
                xyz[:, :2] = np.dot(R, xyz[:, :2].T).T
                xyz[:, 1] *= skew_factor.value

                tree = cKDTree(xyz)

                xyz_out -= np.kron(center, np.ones(xyz_out.shape[0])).T
                xyz_out[:, :2] = np.dot(R, xyz_out[:, :2].T).T
                xyz_out[:, 1] *= skew_factor.value

                # Find nearest cells
                rad, ind = tree.query(xyz_out, 8)

                for key, value in values.items():
                    values_interp[key] = np.zeros(xyz_out.shape[0])
                    weight = np.zeros(xyz_out.shape[0])

                    for ii in range(8):
                        values_interp[key] += value[ind[:, ii]]/(rad[:, ii] + 1e-1)
                        weight += 1./(rad[:, ii] + 1e-1)

                    values_interp[key] /= weight

            else:
                # Find nearest cells
                for key, value in values.items():
                    rad, ind = tree.query(xyz_out)
                    values_interp[key] = value[ind]

            for key in values_interp.keys():
                if space.value == 'Log':
                    values_interp[key] = np.exp(values_interp[key])

                values_interp[key][np.isnan(values_interp[key])] = -99999

                if method.value == 'Inverse Distance':
                    values_interp[key][rad[:, 0] > max_distance.value] = -99999
                    if max_depth.value is not None:
                        values_interp[key][
                            np.abs(xyz_out[:, 2] - xyz[ind[:, 0], 2]) > max_depth.value
                        ] = -99999

                else:
                    values_interp[key][rad > max_distance.value] = -99999

                    if max_depth.value is not None:
                        values_interp[key][
                            np.abs(xyz_out[:, 2] - xyz[ind, 2]) > max_depth.value
                        ] = -99999


            if topography.value is not None:
                topo = workspace.get_entity(topography.value)[0].vertices
                xyz_out = mesh_out.centroids.copy()
                F = LinearNDInterpolator(topo[:, :2], topo[:, 2])
                z_interp = F(xyz_out[:, :2])

                ind_nan = np.isnan(z_interp)
                if any(ind_nan):
                    tree = cKDTree(topo[:, :2])
                    _, ind = tree.query(xyz_out[ind_nan, :2])
                    z_interp[ind_nan] = topo[ind, 2]

                for key in values_interp.keys():
                    values_interp[key][xyz_out[:, 2] > z_interp] = -99999

            if xy_extent.value is not None:

                xy_ref = workspace.get_entity(xy_extent.value)[0]

                if hasattr(xy_ref, "centroids"):
                    xy_ref = xy_ref.centroids
                elif hasattr(xy_ref, "vertices"):
                    xy_ref = xy_ref.vertices

                tree = cKDTree(xy_ref[:, :2])
                rad, _ = tree.query(xyz_out[:, :2])
                for key in values_interp.keys():
                    values_interp[key][rad > max_distance.value] = -99999

            for key in values_interp.keys():
                mesh_out.add_data({key + "_interp": {"values": values_interp[key]}})

            interpolate.value = False

            workspace.finalize()

    workspace = Workspace(h5file)

    names = list(workspace.list_objects_name.values())

    objects, data = object_data_selection_widget(h5file, select_multiple=True)

    out_mode = widgets.RadioButtons(
        options=['To Object:', 'Create 3D Grid'],
        value='To Object:',
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

    padding_distance = widgets.Text(
        value="0, 0, 0, 0, 0, 0",
        description='Pad Distance (W, E, N, S, D, U)',
        disabled=False,
        style={'description_width': 'initial'}
    )

    expansion_fact = widgets.FloatText(
        value=1.05,
        description='Expansion factor',
        disabled=False,
        style={'description_width': 'initial'}
    )

    space = widgets.RadioButtons(
        options=['Linear', 'Log'],
        value='Linear',
        disabled=False
    )

    def method_update(_):

        if method.value == 'Inverse Distance':
            method_panel.children = [method, method_skew]
        else:
            method_panel.children = [method]

    method = widgets.RadioButtons(
        options=['Nearest', 'Linear', 'Inverse Distance'],
        value='Nearest',
        disabled=False
    )

    max_distance = widgets.FloatText(
        value=0,
        description='Maximum distance XY (m)',
        style={'description_width': 'initial'}
    )

    max_depth = widgets.FloatText(
        value=0,
        description='Maximum distance Z (m)',
        style={'description_width': 'initial'}
    )

    skew_angle = widgets.FloatText(
        value=0,
        description='Azimuth (d.dd)',
        disabled=False,
        style={'description_width': 'initial'}
    )

    skew_factor = widgets.FloatText(
        value=1,
        description='Factor',
        disabled=False,
        style={'description_width': 'initial'}
    )

    method_skew = VBox([widgets.Label('Skew interpolation'), skew_angle, skew_factor])
    method_panel = HBox([method])

    method.observe(method_update)
    interpolate = widgets.ToggleButton(
        value=False,
        description='Interpolate',
        icon='check'
    )

    interpolate.observe(interpolate_call)

    topography = widgets.Dropdown(
        options=[None] + names,
        description="Cut with topo",
        style={'description_width': 'initial'}
    )

    xy_extent = widgets.Dropdown(
        options=[None] + names,
        description="Trim xy extent with:",
        style={'description_width': 'initial'}
    )

    new_grid_panel = VBox([
            new_grid, ref_dropdown,
            core_cell_size, depth_core, padding_distance, expansion_fact
        ])

    out_panel = HBox([
        out_mode, mesh_dropdown
    ])

    return VBox([
        HBox([
            VBox([widgets.Label("Input"), objects, data]),
            VBox([widgets.Label("Output"), out_panel])
            ]),
        VBox([
            widgets.Label("Interpolation Parameters"),
            HBox([
            VBox([widgets.Label("Space"), space]),
            VBox([widgets.Label("Method"), method_panel])]),
            max_distance, max_depth, topography, xy_extent,
            ]),
        interpolate
    ])