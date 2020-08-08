import re
import os
import time
import json

import ipywidgets as widgets
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib import collections
import numpy
from geoh5py.data import FloatData
from geoh5py.groups import ContainerGroup
from geoh5py.objects import Curve, Grid2D, Points, Surface
from geoh5py.workspace import Workspace
from ipywidgets.widgets import HBox, Label, Layout, VBox
from scipy.interpolate import LinearNDInterpolator, interp1d
from scipy.spatial import cKDTree, Delaunay
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line

from geoapps.functions.plotting import format_labels
from geoapps.functions.inversion import TopographyOptions
from geoapps.functions.selection import (
    object_data_selection_widget,
    plot_plan_data_selection,
    LineOptions,
)
from geoapps.functions.utils import (
    filter_xy,
    export_grid_2_geotiff,
    geotiff_2_grid,
    rotate_xy,
    find_value,
    signal_processing_1d,
)


def calculator(h5file):
    w_s = Workspace(h5file)
    objects, data = object_data_selection_widget(h5file, select_multiple=False)
    _, store = object_data_selection_widget(
        h5file, objects=objects, select_multiple=False
    )
    store.description = "Assign result to: "
    store.style = {"description_width": "initial"}
    use = widgets.ToggleButton(description=">> Add >>")
    add = widgets.ToggleButton(
        description=">> Create >>", style={"description_width": "initial"}
    )
    compute = widgets.ToggleButton(description="Compute: ", button_style="success")
    channel = widgets.Text("NewChannel", description="Name: ")
    equation = widgets.Textarea(layout=Layout(width="75%"))

    var = {}

    def evaluate(var):
        vals = eval(equation.value)
        obj = w_s.get_entity(objects.value)[0]
        obj.get_data(store.value)[0].values = vals
        print(vals)
        w_s.finalize()

    def click_add(_):
        if add.value:
            obj = w_s.get_entity(objects.value)[0]

            if getattr(obj, "vertices", None) is not None:
                new_data = obj.add_data(
                    {channel.value: {"values": numpy.zeros(obj.n_vertices)}}
                )
            else:
                new_data = obj.add_data(
                    {channel.value: {"values": numpy.zeros(obj.n_cells)}}
                )

            data.options = obj.get_data_list()
            store.options = [new_data.name]
            store.value = new_data.name

            add.value = False

    def click_use(_):
        if use.value:
            name = objects.value + "." + data.value
            if name not in var.keys():
                obj = w_s.get_entity(objects.value)[0]
                var[name] = obj.get_data(data.value)[0].values

            equation.value = equation.value + "var['" + name + "']"
            use.value = False

    def click_compute(_):
        if compute.value:
            evaluate(var)
            compute.value = False

    use.observe(click_use)
    add.observe(click_add)
    compute.observe(click_compute)

    return VBox(
        [
            objects,
            HBox([use, data]),
            VBox(
                [equation, HBox([add, channel]), store, compute],
                layout=Layout(width="100%"),
            ),
        ]
    )


def cdi_curve_2_surface(h5file):
    """
    Application for the conversion of conductivity/depth curves to
    a pseudo 3D conductivity model on surface.
    """

    workspace = Workspace(h5file)

    objects, models = object_data_selection_widget(h5file, add_groups=True)
    models.description = "Model fields: "

    _, line_channel = object_data_selection_widget(
        h5file, add_groups=True, objects=objects, find_value=["line"]
    )
    line_channel.description = "Line field:"

    topo_options = TopographyOptions(h5file)

    _, elevations = object_data_selection_widget(
        h5file, add_groups=True, objects=objects
    )
    elevations.description = "Elevations:"

    def z_option_change(_):
        if z_option.value == "depth":
            elevations.description = "Depth:"
            depth_panel.children = [
                z_option,
                widgets.VBox(
                    [elevations, widgets.Label("Topography"), topo_options.widget,]
                ),
            ]
        else:
            elevations.description = "Elevation:"
            depth_panel.children = [z_option, elevations]

    z_option = widgets.RadioButtons(
        options=["elevation", "depth"],
        description="Layers reference:",
        style={"description_width": "initial"},
    )

    z_option.observe(z_option_change)

    max_depth = widgets.FloatText(
        value=400, description="Max depth (m):", style={"description_width": "initial"}
    )
    max_distance = widgets.FloatText(
        value=50,
        description="Max distance (m):",
        style={"description_width": "initial"},
    )

    tolerance = widgets.FloatText(
        value=1, description="Tolerance (m):", style={"description_width": "initial"}
    )
    depth_panel = widgets.HBox([z_option, elevations])

    out_name = widgets.Text("CDI_", description="Name: ")

    def convert_trigger(_):

        if convert.value:
            if workspace.get_entity(objects.value):
                curve = workspace.get_entity(objects.value)[0]
                convert.value = False
            else:
                convert.button_style = "warning"
                convert.value = False
                return

            lines_id = curve.get_data(line_channel.value)[0].values
            lines = numpy.unique(lines_id).tolist()

            if z_option.value == "depth":
                if topo_options.options_button.value == "Object":

                    topo_obj = workspace.get_entity(topo_options.objects.value)[0]

                    if hasattr(topo_obj, "centroids"):
                        vertices = topo_obj.centroids.copy()
                    else:
                        vertices = topo_obj.vertices.copy()

                    topo_xy = vertices[:, :2]

                    if topo_options.value.value == "Vertices":
                        topo_z = vertices[:, 2]
                    else:
                        topo_z = topo_obj.get_data(topo_options.value.value)[0].values

                else:
                    topo_xy = curve.vertices[:, :2].copy()

                    if topo_options.options_button.value == "Constant":
                        topo_z = (
                            numpy.ones_like(curve.vertices[:, 2])
                            * topo_options.constant.value
                        )
                    else:
                        topo_z = (
                            numpy.ones_like(curve.vertices[:, 2])
                            + topo_options.offset.value
                        )

                surf = Delaunay(topo_xy)
                topo = LinearNDInterpolator(surf, topo_z)
                tree_topo = cKDTree(topo_xy)

            model_vertices = []
            model_cells = []
            model_count = 0
            locations = curve.vertices
            model = []
            line_ids = []
            for line in lines:

                line_ind = numpy.where(lines_id == line)[0]

                n_sounding = len(line_ind)
                if n_sounding < 2:
                    continue

                xyz = locations[line_ind, :]

                # Create a 2D mesh to store the results
                if numpy.std(xyz[:, 1]) > numpy.std(xyz[:, 0]):
                    order = numpy.argsort(xyz[:, 1])
                else:
                    order = numpy.argsort(xyz[:, 0])

                X, Y, Z, M = [], [], [], []
                # Stack the z-coordinates and model
                nZ = 0
                for ind, (z_prop, m_prop) in enumerate(
                    zip(
                        curve.get_property_group(elevations.value).properties,
                        curve.get_property_group(models.value).properties,
                    )
                ):
                    nZ += 1
                    z_vals = workspace.get_entity(z_prop)[0].values[line_ind]
                    m_vals = workspace.get_entity(m_prop)[0].values[line_ind]

                    keep = (
                        (z_vals > 1e-38)
                        * (z_vals < 2e-38)
                        * (m_vals > 1e-38)
                        * (m_vals < 2e-38)
                    ) == False
                    keep[numpy.isnan(z_vals)] = False
                    keep[numpy.isnan(m_vals)] = False

                    X.append(xyz[:, 0][order][keep])
                    Y.append(xyz[:, 1][order][keep])

                    if z_option.value == "depth":
                        z_topo = topo(xyz[:, 0][order][keep], xyz[:, 1][order][keep])

                        nan_z = numpy.isnan(z_topo)
                        if numpy.any(nan_z):
                            _, ii = tree_topo.query(xyz[:, :2][order][keep][nan_z])
                            z_topo[nan_z] = topo_z[ii]

                        Z.append(z_topo + z_vals[order][keep])

                    else:
                        Z.append(z_vals[order][keep])

                    M.append(m_vals[order][keep])

                    if ind == 0:
                        x_loc = xyz[:, 0][order][keep]
                        y_loc = xyz[:, 1][order][keep]
                        z_loc = Z[0]

                X = numpy.hstack(X)
                Y = numpy.hstack(Y)
                Z = numpy.hstack(Z)
                model.append(numpy.ravel(numpy.hstack(M)))
                line_ids.append(numpy.ones_like(Z.ravel()) * line)
                if numpy.std(y_loc) > numpy.std(x_loc):
                    tri2D = Delaunay(numpy.c_[numpy.ravel(Y), numpy.ravel(Z)])
                    dist = numpy.ravel(Y)
                    topo_top = interp1d(y_loc, z_loc)
                else:
                    tri2D = Delaunay(numpy.c_[numpy.ravel(X), numpy.ravel(Z)])
                    dist = numpy.ravel(X)
                    topo_top = interp1d(x_loc, z_loc)

                    # Remove triangles beyond surface edges
                indx = numpy.ones(tri2D.simplices.shape[0], dtype=bool)
                for ii in range(3):
                    x = numpy.mean(
                        numpy.c_[
                            dist[tri2D.simplices[:, ii]],
                            dist[tri2D.simplices[:, ii - 1]],
                        ],
                        axis=1,
                    )
                    z = numpy.mean(
                        numpy.c_[
                            Z[tri2D.simplices[:, ii]], Z[tri2D.simplices[:, ii - 1]]
                        ],
                        axis=1,
                    )

                    length = numpy.linalg.norm(
                        tri2D.points[tri2D.simplices[:, ii], :]
                        - tri2D.points[tri2D.simplices[:, ii - 1], :],
                        axis=1,
                    )

                    indx *= (
                        (z <= (topo_top(x) + tolerance.value))
                        * (z >= (topo_top(x) - max_depth.value - tolerance.value))
                        * (length < max_distance.value)
                    )

                # Remove the simplices too long
                tri2D.simplices = tri2D.simplices[indx, :]
                tri2D.vertices = tri2D.vertices[indx, :]

                temp = numpy.arange(int(nZ * n_sounding)).reshape(
                    (nZ, n_sounding), order="F"
                )
                model_vertices.append(
                    numpy.c_[numpy.ravel(X), numpy.ravel(Y), numpy.ravel(Z)]
                )
                model_cells.append(tri2D.simplices + model_count)

                model_count += tri2D.points.shape[0]

            surface = Surface.create(
                workspace,
                name=out_name.value,
                vertices=numpy.vstack(model_vertices),
                cells=numpy.vstack(model_cells),
            )

            surface.add_data(
                {
                    models.value: {"values": numpy.hstack(model)},
                    "Line": {"values": numpy.hstack(line_ids)},
                }
            )

    convert = widgets.ToggleButton(description="Convert >>", button_style="success")
    convert.observe(convert_trigger)
    widget = widgets.VBox(
        [
            widgets.VBox(
                [
                    widgets.VBox(
                        [
                            objects,
                            line_channel,
                            models,
                            depth_panel,
                            widgets.Label("Triangulation"),
                            max_depth,
                            max_distance,
                            tolerance,
                        ]
                    ),
                ]
            ),
            widgets.Label("Output"),
            widgets.HBox([convert, out_name,]),
        ]
    )

    return widget


def contour_values_widget(h5file, **kwargs):
    """
    Application for 2D contouring of spatial data.
    """

    workspace = Workspace(h5file)

    def compute_plot(entity_name, data_name, contour_vals):

        entity = workspace.get_entity(entity_name)[0]

        if entity.get_data(data_name):
            data = entity.get_data(data_name)[0]
        else:
            return

        if data.entity_type.color_map is not None:
            new_cmap = data.entity_type.color_map.values
            map_vals = new_cmap["Value"].copy()
            cmap = colors.ListedColormap(
                numpy.c_[
                    new_cmap["Red"] / 255,
                    new_cmap["Green"] / 255,
                    new_cmap["Blue"] / 255,
                ]
            )
            color_norm = colors.BoundaryNorm(map_vals, cmap.N)

        else:
            cmap = None
            color_norm = None

        if contour_vals != "":
            vals = re.split(",", contour_vals)
            cntrs = []
            for val in vals:
                if ":" in val:
                    param = numpy.asarray(re.split(":", val), dtype="int")

                    if len(param) == 2:
                        cntrs += [numpy.arange(param[0], param[1])]
                    else:

                        cntrs += [numpy.arange(param[0], param[2], param[1])]

                else:
                    cntrs += [numpy.float(val)]
            contour_vals = numpy.unique(numpy.sort(numpy.hstack(cntrs)))
        else:
            contour_vals = None

        plt.figure(figsize=(10, 10))
        axs = plt.subplot()
        contour_sets = None
        if isinstance(entity, Grid2D):
            xx = entity.centroids[:, 0].reshape(entity.shape, order="F")
            yy = entity.centroids[:, 1].reshape(entity.shape, order="F")
            if len(data.values) == entity.n_cells:
                grid_data = data.values.reshape(xx.shape, order="F")

                axs.pcolormesh(
                    xx, yy, grid_data, cmap=cmap, norm=color_norm, shading="auto"
                )
                format_labels(xx, yy, axs)
                if contour_vals is not None:
                    contour_sets = axs.contour(
                        xx,
                        yy,
                        grid_data,
                        len(contour_vals),
                        levels=contour_vals,
                        colors="k",
                        linewidths=0.5,
                    )

        elif isinstance(entity, (Points, Curve, Surface)):

            if len(data.values) == entity.n_vertices:
                xx = entity.vertices[:, 0]
                yy = entity.vertices[:, 1]
                axs.scatter(xx, yy, 5, data.values, cmap=cmap)
                if contour_vals is not None:
                    contour_sets = axs.tricontour(
                        xx,
                        yy,
                        data.values,
                        levels=contour_vals,
                        linewidths=0.5,
                        colors="k",
                    )
                format_labels(xx, yy, axs)

        else:
            contours.contours = None

        contours.contours = contour_sets

    def save_selection(_):
        if export.value:

            entity = workspace.get_entity(objects.value)[0]
            data_name = data.value

            # TODO
            #  Create temporary workspace and write to trigger LIVE LINK
            # temp_geoh5 = os.path.join(os.path.dirname(
            #     os.path.abspath(workspace.h5file)), "Temp", "temp.geoh5")
            # ws_out = Workspace(temp_geoh5)

            if contours.contours is not None:

                vertices, cells, values = [], [], []
                count = 0
                for segs, level in zip(
                    contours.contours.allsegs, contours.contours.levels
                ):
                    for poly in segs:
                        n_v = len(poly)
                        vertices.append(poly)
                        cells.append(
                            numpy.c_[
                                numpy.arange(count, count + n_v - 1),
                                numpy.arange(count + 1, count + n_v),
                            ]
                        )
                        values.append(numpy.ones(n_v) * level)

                        count += n_v
                if vertices:
                    vertices = numpy.vstack(vertices)

                    if z_value.value:
                        vertices = numpy.c_[vertices, numpy.hstack(values)]
                    else:

                        if isinstance(entity, (Points, Curve, Surface)):
                            z_interp = LinearNDInterpolator(
                                entity.vertices[:, :2], entity.vertices[:, 2]
                            )
                            vertices = numpy.c_[vertices, z_interp(vertices)]
                        else:
                            vertices = numpy.c_[
                                vertices,
                                numpy.ones(vertices.shape[0]) * entity.origin["z"],
                            ]

                    curve = Curve.create(
                        entity.workspace,
                        name=export_as.value,
                        vertices=vertices,
                        cells=numpy.vstack(cells).astype("uint32"),
                    )
                    curve.add_data({contours.value: {"values": numpy.hstack(values)}})

                    # objects.options = list(entity.workspace.list_objects_name.values())
                    # objects.value = entity.name
                    # data.options = entity.get_data_list()
                    # data.value = data_name

                export.value = False

    if "contours" in kwargs.keys():
        contours = kwargs["contours"]
    else:
        contours = ""

    contours = widgets.Text(
        value=contours, description="Contours", disabled=False, continuous_update=False
    )

    def updateContours(_):
        if data.value is not None:
            export_as.value = data.value + "_" + contours.value

    contours.observe(updateContours, names="value")
    contours.contours = None

    objects, data = object_data_selection_widget(h5file)

    if "objects" in kwargs.keys() and kwargs["objects"] in objects.options:
        objects.value = kwargs["objects"]

    if "data" in kwargs.keys() and kwargs["data"] in data.options:
        data.value = kwargs["data"]

    export = widgets.ToggleButton(
        value=False,
        description="Export to GA",
        button_style="danger",
        tooltip="Description",
        icon="check",
    )

    export.observe(save_selection, names="value")

    export_as = widgets.Text(indent=False,)
    updateContours("")

    def updateName(_):
        export_as.value = data.value + "_" + contours.value

    data.observe(updateName, names="value")

    z_value = widgets.Checkbox(
        value=False, indent=False, description="Assign Z from values"
    )

    out = widgets.interactive_output(
        compute_plot,
        {"entity_name": objects, "data_name": data, "contour_vals": contours},
    )

    contours.value = contours.value
    return widgets.VBox(
        [
            widgets.HBox(
                [
                    VBox([Label("Input options:"), objects, data, contours]),
                    VBox(
                        [Label("Output options:"), export_as, z_value, export],
                        layout=Layout(width="50%"),
                    ),
                ]
            ),
            out,
        ]
    )


def coordinate_transformation_widget(
    h5file, plot=False, epsg_in=None, epsg_out=None, object_names=[]
):
    """

    """
    try:
        import os
        import gdal
        import fiona
        from fiona.transform import transform
    except ModuleNotFoundError as err:
        print(err, "Trying to install through geopandas, hang tight...")
        import os

        os.system("conda install -c conda-forge geopandas=0.7.0")
        from fiona.transform import transform
        import gdal

    workspace = Workspace(h5file)

    def listObjects(obj_names, epsg_in, epsg_out, export, plot_it):

        out_list = []
        if epsg_in != 0 and epsg_out != 0 and (plot_it or export):
            inProj = f"EPSG:{int(epsg_in)}"
            outProj = f"EPSG:{int(epsg_out)}"

            if inProj == "EPSG:4326":
                labels_in = ["Lon", "Lat"]
                tick_format_in = "%.3f"
            else:
                labels_in = ["Easting", "Northing"]
                tick_format_in = "%i"

            if outProj == "EPSG:4326":
                labels_out = ["Lon", "Lat"]
                tick_format_out = "%.3f"
            else:
                labels_out = ["Easting", "Northing"]
                tick_format_out = "%i"

            if plot_it:
                fig = plt.figure(figsize=(10, 10))
                ax1 = plt.subplot(1, 2, 1)
                ax2 = plt.subplot(1, 2, 2)
                X1, Y1, X2, Y2 = [], [], [], []

            if export:
                group = ContainerGroup.create(
                    workspace, name=f"Projection epsg:{int(epsg_out)}"
                )

            for name in obj_names:
                obj = workspace.get_entity(name)[0]

                temp_work = Workspace(workspace.name + "temp")

                count = 0

                if isinstance(obj, Grid2D):
                    # Get children data
                    if obj.rotation != 0:
                        print(
                            f"{name} object ignored. Re-projection only available for non-rotated Grid2D"
                        )
                        continue

                    for child in obj.children:

                        temp_file = child.name + ".tif"
                        temp_file_in = child.name + "_" + str(int(epsg_out)) + ".tif"

                        if isinstance(child, FloatData):

                            export_grid_2_geotiff(
                                child, temp_file, epsg_in, dataType="float"
                            )

                            grid = gdal.Open(temp_file)
                            gdal.Warp(
                                temp_file_in, grid, dstSRS="EPSG:" + str(int(epsg_out))
                            )

                            if count == 0:
                                grid2d = geotiff_2_grid(
                                    temp_work, temp_file_in, grid_name=obj.name
                                )

                            else:
                                _ = geotiff_2_grid(
                                    temp_work, temp_file_in, grid_object=grid2d
                                )

                            if plot_it:
                                plot_plan_data_selection(obj, child, ax=ax1)
                                X1.append(obj.centroids[:, 0])
                                Y1.append(obj.centroids[:, 1])
                                plot_plan_data_selection(
                                    grid2d, grid2d.children[-1], ax=ax2
                                )
                                X2.append(grid2d.centroids[:, 0])
                                Y2.append(grid2d.centroids[:, 1])

                            del grid
                            os.remove(temp_file)
                            os.remove(temp_file_in)
                            count += 1

                    if export:
                        grid2d.copy(parent=group)

                    os.remove(temp_work.h5file)

                else:
                    if not hasattr(obj, "vertices"):
                        print(f"Skipping {name}. Entity does not have vertices")
                        continue

                    x, y = obj.vertices[:, 0].tolist(), obj.vertices[:, 1].tolist()

                    if epsg_in == "4326":
                        x, y = y, x

                    x2, y2 = transform(inProj, outProj, x, y)

                    if epsg_in == "4326":
                        x2, y2 = y2, x2

                    if export:
                        new_obj = obj.copy(parent=group, copy_children=True)

                        new_obj.vertices = numpy.c_[x2, y2, obj.vertices[:, 2]]
                        out_list.append(new_obj)

                    if plot_it:
                        ax1.scatter(x, y, 5)
                        ax2.scatter(x2, y2, 5)
                        X1.append(x), Y1.append(y), X2.append(x2), Y2.append(y2)

            workspace.finalize()
            if plot_it and X1:
                format_labels(
                    numpy.hstack(X1),
                    numpy.hstack(Y1),
                    ax1,
                    labels=labels_in,
                    tick_format=tick_format_in,
                )

            if plot_it and X2:
                format_labels(
                    numpy.hstack(X2),
                    numpy.hstack(Y2),
                    ax2,
                    labels=labels_out,
                    tick_format=tick_format_out,
                )

        return out_list

    names = []
    for obj in workspace._objects.values():
        if isinstance(obj.__call__(), (Curve, Points, Surface, Grid2D)):
            names.append(obj.__call__().name)

    def saveIt(_):
        if export.value:
            export.value = False
            print("Export completed!")

    object_names = [name for name in object_names if name in names]
    objects = widgets.SelectMultiple(
        options=names, value=object_names, description="Object:",
    )

    export = widgets.ToggleButton(
        value=False,
        description="Export to GA",
        button_style="danger",
        tooltip="Description",
        icon="check",
    )

    export.observe(saveIt)

    plot_it = widgets.ToggleButton(
        value=False,
        description="Plot",
        button_style="",
        tooltip="Description",
        icon="check",
    )

    epsg_in = widgets.FloatText(value=epsg_in, description="EPSG # in:", disabled=False)

    epsg_out = widgets.FloatText(
        value=epsg_out, description="EPSG # out:", disabled=False
    )

    out = widgets.interactive(
        listObjects,
        obj_names=objects,
        epsg_in=epsg_in,
        epsg_out=epsg_out,
        export=export,
        plot_it=plot_it,
    )

    return out


def edge_detection_widget(
    h5file, sigma=1.0, threshold=3, line_length=4.0, line_gap=2, resolution=100
):
    """
    Widget for Grid2D objects for the automated detection of line features.
    The application relies on the Canny and Hough trandforms from the
    Scikit-Image library.

    :param grid: Grid2D object
    :param data: Children data object for the provided grid

    Optional
    --------

    :param sigma [Canny]: standard deviation of the Gaussian filter
    :param threshold [Hough]: Value threshold
    :param line_length [Hough]: Minimum accepted pixel length of detected lines
    :param line_gap [Hough]: Maximum gap between pixels to still form a line.
    """

    workspace = Workspace(h5file)

    objects, data_obj = object_data_selection_widget(h5file)

    center_x = widgets.FloatSlider(
        min=-100, max=100, steps=10, description="Easting", continuous_update=False,
    )
    center_y = widgets.FloatSlider(
        min=-100,
        max=100,
        steps=10,
        description="Northing",
        continuous_update=False,
        orientation="vertical",
    )
    azimuth = widgets.FloatSlider(
        min=-90,
        max=90,
        value=0,
        steps=5,
        description="Orientation",
        continuous_update=False,
    )
    width_x = widgets.FloatSlider(
        min=0,
        max=100,
        steps=10,
        value=1000,
        description="Width",
        continuous_update=False,
    )

    width_y = widgets.FloatSlider(
        min=0,
        max=100,
        steps=10,
        value=1000,
        description="Height",
        continuous_update=False,
        orientation="vertical",
    )

    resolution = widgets.FloatText(
        value=resolution,
        description="Grid Resolution (m)",
        style={"description_width": "initial"},
    )

    window_size = widgets.IntText(
        value=64,
        description="Window size (pixels)",
        style={"description_width": "initial"},
    )

    data_count = Label("Data Count: 0", tooltip="Keep <1500 for speed")
    zoom_extent = widgets.ToggleButton(
        value=True,
        description="Zoom on selection",
        tooltip="Keep plot extent on selection",
        icon="check",
    )
    export = widgets.ToggleButton(
        value=False,
        description="Export to GA",
        button_style="danger",
        tooltip="Description",
        icon="check",
    )
    compute = widgets.ToggleButton(
        value=False,
        description="Compute",
        button_style="warning",
        tooltip="Description",
        icon="check",
    )
    sigma = widgets.FloatSlider(
        min=0.0,
        max=10,
        step=0.1,
        value=sigma,
        continuous_update=False,
        description="Sigma",
        style={"description_width": "initial"},
    )

    line_length = widgets.IntSlider(
        min=1.0,
        max=100.0,
        step=1.0,
        value=line_length,
        continuous_update=False,
        description="Line Length",
        style={"description_width": "initial"},
    )

    line_gap = widgets.IntSlider(
        min=1.0,
        max=100.0,
        step=1.0,
        value=line_gap,
        continuous_update=False,
        description="Line Gap",
        style={"description_width": "initial"},
    )

    threshold = widgets.IntSlider(
        min=1.0,
        max=100.0,
        step=1.0,
        value=threshold,
        continuous_update=False,
        description="Threshold",
        style={"description_width": "initial"},
    )

    export_as = widgets.Text(
        value=(
            f"S:{sigma.value}"
            + f" T:{threshold.value}"
            + f" LL={line_length.value}"
            + f" LG={line_gap.value}"
        ),
        description="Save as:",
        disabled=False,
    )

    def set_bounding_box(_, update_values=False):
        # Fetch vertices in the project
        lim_x = [1e8, -1e8]
        lim_y = [1e8, -1e8]

        obj = workspace.get_entity(objects.value)[0]
        if isinstance(obj, Grid2D):

            objects.grid = obj

            lim_x[0], lim_x[1] = obj.centroids[:, 0].min(), obj.centroids[:, 0].max()
            lim_y[0], lim_y[1] = obj.centroids[:, 1].min(), obj.centroids[:, 1].max()

            center_x.max = lim_x[1]
            center_x.value = numpy.mean(lim_x)
            center_x.min = lim_x[0]

            center_y.max = lim_y[1]
            center_y.value = numpy.mean(lim_y)
            center_y.min = lim_y[0]

            width_x.max = lim_x[1] - lim_x[0]
            width_x.value = width_x.max / 2.0
            width_x.min = 100

            width_y.max = lim_y[1] - lim_y[0]
            width_y.value = width_y.max / 2.0
            width_y.min = 100

            if update_values:
                center_x.value = numpy.mean(lim_x)
                width_x.value = (lim_x[1] - lim_x[0]) / 2

                center_y.value = numpy.mean(lim_y)
                width_y.value = (lim_y[1] - lim_y[0]) / 2

    objects.observe(set_bounding_box)
    set_bounding_box("", update_values=True)

    def saveIt(_):
        if export.value and getattr(export, "vertices", None) is not None:
            Curve.create(
                objects.grid.workspace,
                name=export_as.value,
                vertices=export.vertices,
                cells=export.cells,
            )

            export.value = False
            print(f"Lines {export_as.value} exported to: {workspace.h5file}")

    export.observe(saveIt)

    def compute_trigger(_):
        if compute.value and getattr(objects, "grid", None) is not None:
            x = objects.grid.centroids[:, 0].reshape(objects.grid.shape, order="F")
            y = objects.grid.centroids[:, 1].reshape(objects.grid.shape, order="F")

            grid_data = objects.grid.get_data(data_obj.value)[0].values
            grid_data = grid_data.reshape(objects.grid.shape, order="F")

            filter = filter_xy(x, y, resolution.value)

            ind_x, ind_y = (
                numpy.any(filter, axis=1),
                numpy.any(filter, axis=0),
            )

            grid_data = grid_data[ind_x, :][:, ind_y]
            grid_data -= grid_data.min()
            grid_data /= grid_data.max()

            if numpy.any(grid_data):

                x_locs = x[ind_x, 0]
                y_locs = y[0, ind_y]
                X = x[ind_x, :][:, ind_y]
                Y = y[ind_x, :][:, ind_y]

                # Replace no-data with inverse distance interpolation
                # ndv = (grid_data > 1e-38) * (grid_data < 2e-38)
                # active = ndv == False
                # tree = cKDTree(numpy.c_[X[active], Y[active]])
                # rad, ind = tree.query(numpy.c_[X[ndv], Y[ndv]], k=5)
                #
                # nC = int(ndv.sum())
                # m, w = numpy.zeros(nC), numpy.zeros(nC)
                # for ii in range(5):
                #     m += grid_data[active][ind[:, ii]] / rad[:, ii]
                #     w += 1. / rad[:, ii]
                #
                # grid_data[ndv] = m / w

                # Find edges
                edges = canny(grid_data, sigma=sigma.value, use_quantiles=True)

                # Cycle through tiles of square size
                max_l = window_size.value

                cnt_x = numpy.arange(0, edges.shape[0], max_l / 1.5, dtype=int)
                cnt_y = numpy.arange(0, edges.shape[1], max_l / 1.5, dtype=int)

                coords = []

                for ii in cnt_x.tolist():
                    for jj in cnt_y.tolist():

                        i_min, i_max = (
                            numpy.min([ii, edges.shape[0] - max_l]),
                            numpy.min([ii + max_l, edges.shape[0]]),
                        )
                        j_min, j_max = (
                            numpy.min([jj, edges.shape[1] - max_l]),
                            numpy.min([jj + max_l, edges.shape[1]]),
                        )

                        # tile = active[i_min:i_max, j_min:j_max]
                        lines = probabilistic_hough_line(
                            edges[i_min:i_max, j_min:j_max],
                            line_length=line_length.value,
                            threshold=threshold.value,
                            line_gap=line_gap.value,
                            seed=0,
                        )

                        if numpy.any(lines):
                            coord = numpy.vstack(lines)

                            # # Remove lines in no-data region
                            # indices_1 = tile[coord[1::2, 1], coord[1::2, 0]]
                            # indices_2 = tile[coord[0::2, 1], coord[0::2, 0]]
                            #
                            # indices = numpy.kron(
                            #     numpy.all(numpy.c_[indices_1, indices_2], axis=1), numpy.ones(2),
                            # ).astype(bool)

                            coords.append(
                                numpy.c_[
                                    x_locs[i_min:i_max][coord[:, 1]],
                                    y_locs[j_min:j_max][coord[:, 0]],
                                ]
                            )

                if coords:
                    coord = numpy.vstack(coords)
                    objects.lines = numpy.unique(coord, axis=1)

                    export.button_style = "success"

                else:
                    objects.lines = None

            compute.value = False

            compute.button_style = ""

    compute.observe(compute_trigger)

    def parameters_change(_):
        export_as.value = (
            f"S:{sigma.value}"
            + f" T:{threshold.value}"
            + f" LL:{line_length.value}"
            + f" LG:{line_gap.value}"
        )
        compute.button_style = "warning"
        export.button_style = "danger"

    data_obj.observe(parameters_change)
    resolution.observe(parameters_change)
    threshold.observe(parameters_change)
    sigma.observe(parameters_change)
    line_length.observe(parameters_change)
    line_gap.observe(parameters_change)

    def compute_plot(
        data_name,
        resolution,
        center_x,
        center_y,
        width_x,
        width_y,
        azimuth,
        zoom_extent,
        compute,
    ):

        if getattr(objects, "grid", None) is not None and objects.grid.get_data(
            data_name
        ):
            data_obj = objects.grid.get_data(data_name)[0]
            fig = plt.figure(figsize=(10, 10))
            ax1 = plt.subplot()

            corners = numpy.r_[
                numpy.c_[-1.0, -1.0],
                numpy.c_[-1.0, 1.0],
                numpy.c_[1.0, 1.0],
                numpy.c_[1.0, -1.0],
                numpy.c_[-1.0, -1.0],
            ]
            corners[:, 0] *= width_x / 2
            corners[:, 1] *= width_y / 2
            corners = rotate_xy(corners, [0, 0], -azimuth)
            ax1.plot(corners[:, 0] + center_x, corners[:, 1] + center_y, "k")

            _, _, ind_filter, _ = plot_plan_data_selection(
                objects.grid,
                data_obj,
                **{
                    "ax": ax1,
                    "resolution": resolution,
                    "window": {
                        "center": [center_x, center_y],
                        "size": [width_x, width_y],
                        "azimuth": azimuth,
                    },
                    "zoom_extent": zoom_extent,
                    "resize": True,
                },
            )
            data_count.value = f"Data Count: {ind_filter.sum()}"
        if getattr(objects, "lines", None) is not None:
            xy = objects.lines

            indices_1 = filter_xy(
                xy[1::2, 0],
                xy[1::2, 1],
                resolution,
                window={
                    "center": [center_x, center_y],
                    "size": [width_x, width_y],
                    "azimuth": azimuth,
                },
            )
            indices_2 = filter_xy(
                xy[::2, 0],
                xy[::2, 1],
                resolution,
                window={
                    "center": [center_x, center_y],
                    "size": [width_x, width_y],
                    "azimuth": azimuth,
                },
            )

            indices = numpy.kron(
                numpy.any(numpy.c_[indices_1, indices_2], axis=1), numpy.ones(2),
            ).astype(bool)

            xy = objects.lines[indices, :]

            ax1.add_collection(
                collections.LineCollection(
                    numpy.reshape(xy, (-1, 2, 2)), colors="k", linewidths=2
                )
            )

            if numpy.any(xy):
                vertices = numpy.vstack(xy)
                cells = (
                    numpy.arange(vertices.shape[0]).astype("uint32").reshape((-1, 2))
                )
                if numpy.any(cells):
                    export.vertices = numpy.c_[vertices, numpy.zeros(vertices.shape[0])]
                    export.cells = cells
                    export.button_style = "success"
            else:
                export.vertices = None
                export.cells = None

    plot_window = widgets.interactive_output(
        compute_plot,
        {
            "data_name": data_obj,
            "resolution": resolution,
            "center_x": center_x,
            "center_y": center_y,
            "width_x": width_x,
            "width_y": width_y,
            "azimuth": azimuth,
            "zoom_extent": zoom_extent,
            "compute": compute,
        },
    )

    out = VBox(
        [
            HBox(
                [
                    VBox(
                        [
                            objects,
                            data_obj,
                            resolution,
                            compute,
                            HBox([export, export_as]),
                        ]
                    ),
                    VBox(
                        [
                            sigma,
                            threshold,
                            line_length,
                            line_gap,
                            window_size,
                            data_count,
                        ]
                    ),
                ]
            ),
            HBox(
                [
                    center_y,
                    width_y,
                    VBox([width_x, center_x, azimuth, zoom_extent, plot_window]),
                ],
                layout=Layout(align_items="center"),
            ),
        ]
    )

    return out


class EMLineProfiling:
    groups = {
        "early": {
            "color": "blue",
            "channels": [],
            "gates": [],
            "defaults": ["early"],
            "inflx_up": [],
            "inflx_dwn": [],
            "peaks": [],
            "times": [],
            "values": [],
            "locations": [],
        },
        "early + middle": {
            "color": "cyan",
            "channels": [],
            "gates": [],
            "defaults": ["early", "middle"],
            "inflx_up": [],
            "inflx_dwn": [],
            "peaks": [],
            "times": [],
            "values": [],
            "locations": [],
        },
        "middle": {
            "color": "green",
            "channels": [],
            "gates": [],
            "defaults": ["middle"],
            "inflx_up": [],
            "inflx_dwn": [],
            "peaks": [],
            "times": [],
            "values": [],
            "locations": [],
        },
        "early + middle + late": {
            "color": "orange",
            "channels": [],
            "gates": [],
            "defaults": ["early", "middle", "late"],
            "inflx_up": [],
            "inflx_dwn": [],
            "peaks": [],
            "times": [],
            "values": [],
            "locations": [],
        },
        "middle + late": {
            "color": "yellow",
            "channels": [],
            "gates": [],
            "defaults": ["middle", "late"],
            "inflx_up": [],
            "inflx_dwn": [],
            "peaks": [],
            "times": [],
            "values": [],
            "locations": [],
        },
        "late": {
            "color": "red",
            "channels": [],
            "gates": [],
            "defaults": ["late"],
            "inflx_up": [],
            "inflx_dwn": [],
            "peaks": [],
            "times": [],
            "values": [],
            "locations": [],
        },
    }

    viz_param = '<IParameterList Version="1.0">\n <Colour>4278190335</Colour>\n <Transparency>0</Transparency>\n <Nodesize>9</Nodesize>\n <Nodesymbol>Sphere</Nodesymbol>\n <Scalenodesbydata>false</Scalenodesbydata>\n <Data></Data>\n <Scale>1</Scale>\n <Scalebyabsolutevalue>[NO STRING REPRESENTATION]</Scalebyabsolutevalue>\n <Orientation toggled="true">{\n    "DataGroup": "AzmDip",\n    "ManualWidth": true,\n    "Scale": false,\n    "ScaleLog": false,\n    "Size": 30,\n    "Symbol": "Tablet",\n    "Width": 30\n}\n</Orientation>\n</IParameterList>\n'

    def __init__(self, h5file):

        self.workspace = Workspace(h5file)

        dir_path = os.path.dirname(os.path.realpath(os.path.realpath(__file__)))
        with open(os.path.join(dir_path, "AEM_systems.json")) as aem_systems:
            self.em_system_specs = json.load(aem_systems)

        self.system = widgets.Dropdown(
            options=[
                key
                for key, specs in self.em_system_specs.items()
                if specs["type"] == "time"
            ],
            description="Time-Domain System:",
            style={"description_width": "initial"},
        )

        self._groups = self.groups
        self.group_list = widgets.SelectMultiple(description="")

        self.early = numpy.arange(8, 17).tolist()
        self.middle = numpy.arange(17, 28).tolist()
        self.late = numpy.arange(28, 40).tolist()

        self.data_objects, self.data_field = object_data_selection_widget(
            h5file, select_multiple=True
        )
        self.data_objects.description = "Survey"

        self.model_objects, self.model_field = object_data_selection_widget(h5file)
        self.model_objects.description = "1D Object:"
        self.model_field.description = "Model"

        _, self.model_line_field = object_data_selection_widget(
            h5file, objects=self.model_objects, find_value=["line"]
        )
        self.model_line_field.description = "Line field: "

        self.marker = {"left": "<", "right": ">"}

        def update_model_line_fields(_):
            self.model_line_field.options = self.model_field.options
            self.model_line_field.value = find_value(
                self.model_line_field.options, ["line"]
            )

        self.model_field.observe(update_model_line_fields, names="options")

        def get_survey(_):
            if self.workspace.get_entity(self.data_objects.value):
                self.survey = self.workspace.get_entity(self.data_objects.value)[0]
                self.data_field.options = (
                    [p_g.name for p_g in self.survey.property_groups]
                    + ["^-- Groups --^"]
                    + list(self.data_field.options)
                )

        self.data_objects.observe(get_survey, names="value")
        self.data_objects.value = self.data_objects.options[0]
        get_survey("")

        self.lines = LineOptions(h5file, self.data_objects, select_multiple=False)
        self.lines.value.description = "Line"

        self.channels = widgets.SelectMultiple(description="Channels")
        self.group_default_early = widgets.Text(description="Early", value="9-16")
        self.group_default_middle = widgets.Text(description="Middle", value="17-27")
        self.group_default_late = widgets.Text(description="Late", value="28-40")

        def reset_default_bounds(_):
            self.reset_default_bounds()

        self.data = {}
        self.data_channel_options = {}

        def get_data(_):
            data = []

            groups = [p_g.name for p_g in self.survey.property_groups]
            channels = list(self.data_field.value)

            for channel in self.data_field.value:
                if channel in groups:
                    for prop in self.survey.get_property_group(channel).properties:
                        name = self.workspace.get_entity(prop)[0].name
                        if prop not in channels:
                            channels.append(name)

            self.channels.options = channels
            for channel in channels:
                if self.survey.get_data(channel):
                    self.data[channel] = self.survey.get_data(channel)[0]

            # Generate default groups
            self.reset_default_bounds()

            for key, widget in self.data_channel_options.items():
                widget.children[0].options = channels
                widget.children[0].value = find_value(channels, [key])

        self.data_field.observe(get_data, names="value")
        get_data("")

        def get_surf_model(_):
            if self.workspace.get_entity(self.model_objects.value):
                self.surf_model = self.workspace.get_entity(self.model_objects.value)[0]

        self.model_objects.observe(get_surf_model, names="value")

        def get_model(_):
            if self.surf_model.get_data(self.model_field.value):
                self.surf_model = self.surf_model.get_data(self.model_field.value)[0]

        self.model_objects.observe(get_model, names="value")

        self.smoothing = widgets.IntSlider(
            min=0,
            max=64,
            value=0,
            description="Running mean",
            continuous_update=False,
            style={"description_width": "initial"},
        )

        self.residual = widgets.Checkbox(description="Use residual", value=False)

        self.threshold = widgets.FloatSlider(
            value=50,
            min=10,
            max=90,
            step=5,
            continuous_update=False,
            description="Decay threshold (%)",
        )

        self.center = widgets.FloatSlider(
            value=0.5,
            min=0,
            max=1.0,
            step=0.001,
            description="Center (%)",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            style={"description_width": "initial"},
        )

        self.auto_picker = widgets.ToggleButton(description="Pick target", value=True)

        self.focus = widgets.FloatSlider(
            value=1.0,
            min=0.025,
            max=1.0,
            step=0.005,
            description="Width (%)",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
        )

        self.zoom = widgets.VBox([self.center, self.focus])

        self.scale_button = widgets.RadioButtons(
            options=["linear", "symlog",],
            description="Vertical scaling",
            style={"description_width": "initial"},
        )

        def scale_update(_):
            if self.scale_button.value == "symlog":
                scale_panel.children = [
                    self.scale_button,
                    widgets.VBox([widgets.Label("Linear threshold"), self.scale_value]),
                ]
            else:
                scale_panel.children = [self.scale_button]

        self.scale_button.observe(scale_update)
        self.scale_value = widgets.FloatLogSlider(
            min=-18,
            max=10,
            step=0.5,
            base=10,
            value=1e-2,
            description="",
            continuous_update=False,
            style={"description_width": "initial"},
        )
        scale_panel = widgets.HBox([self.scale_button])

        def add_group(_):
            self.add_group()

        def channel_setter(caller):

            channel = caller["owner"]
            data_widget = self.data_channel_options[channel.header]
            data_widget.children[0].value = find_value(
                data_widget.children[0].options, [channel.header]
            )

        self.channel_selection = widgets.Dropdown(
            description="Time Gate",
            style={"description_width": "initial"},
            options=self.em_system_specs[self.system.value]["channels"].keys(),
        )

        def system_observer(_):

            system_specs = {}
            for key, time_gate in self.em_system_specs[self.system.value][
                "channels"
            ].items():
                system_specs[key] = f"{time_gate:.5e}"

            self.channel_selection.options = self.em_system_specs[self.system.value][
                "channels"
            ].keys()

            self.data_channel_options = {}
            for ind, (key, value) in enumerate(system_specs.items()):
                channel_selection = widgets.Dropdown(
                    description="Channel",
                    style={"description_width": "initial"},
                    options=self.channels.options,
                    value=find_value(self.channels.options, [key]),
                )
                channel_selection.header = key
                channel_selection.observe(channel_setter, names="value")

                channel_time = widgets.FloatText(description="Time (s)", value=value)

                self.data_channel_options[key] = widgets.VBox(
                    [channel_selection, channel_time]
                )

        self.system.observe(system_observer)
        system_observer("")

        def channel_panel_update(_):
            self.channel_panel.children = [
                self.channel_selection,
                self.data_channel_options[self.channel_selection.value],
            ]

        self.channel_selection.observe(channel_panel_update, names="value")
        self.channel_panel = widgets.VBox(
            [
                self.channel_selection,
                self.data_channel_options[self.channel_selection.value],
            ]
        )

        self.group_default_early.observe(reset_default_bounds)
        self.group_default_middle.observe(reset_default_bounds)
        self.group_default_late.observe(reset_default_bounds)

        self.group_add = widgets.ToggleButton(description="<< Add New Group <<")
        self.group_name = widgets.Text(description="Name")
        self.group_color = widgets.ColorPicker(
            concise=False, description="Color", value="blue", disabled=False
        )
        self.group_add.observe(add_group)

        def highlight_selection(_):

            self.highlight_selection()

        self.group_list.observe(highlight_selection, names="value")
        self.markers = widgets.ToggleButton(description="Show all markers")

        def export_click(_):
            self.export_click()

        self.export = widgets.ToggleButton(description="Export", button_style="success")
        self.export.observe(export_click)

        def plot_data_selection(
            ind,
            smoothing,
            residual,
            markers,
            scale,
            scale_value,
            center,
            focus,
            groups,
            pick_trigger,
            x_label,
            threshold,
        ):
            self.plot_data_selection(
                ind,
                smoothing,
                residual,
                markers,
                scale,
                scale_value,
                center,
                focus,
                groups,
                pick_trigger,
                x_label,
                threshold,
            )

        def plot_model_selection(ind, center, focus):
            self.plot_model_selection(ind, center, focus)

        self.x_label = widgets.ToggleButtons(
            options=["Distance", "Easting", "Northing"],
            value="Distance",
            description="X-axis label:",
        )
        plotting = widgets.interactive_output(
            plot_data_selection,
            {
                "ind": self.lines.lines,
                "smoothing": self.smoothing,
                "residual": self.residual,
                "markers": self.markers,
                "scale": self.scale_button,
                "scale_value": self.scale_value,
                "center": self.center,
                "focus": self.focus,
                "groups": self.group_list,
                "pick_trigger": self.auto_picker,
                "x_label": self.x_label,
                "threshold": self.threshold,
            },
        )

        section = widgets.interactive_output(
            plot_model_selection,
            {"ind": self.lines.lines, "center": self.center, "focus": self.focus,},
        )

        self.model_panel = widgets.VBox(
            [
                widgets.HBox(
                    [
                        self.model_objects,
                        widgets.VBox([self.model_field, self.model_line_field]),
                    ]
                ),
                section,
            ]
        )

        self.show_model = widgets.Checkbox(description="Show model", value=False)

        def show_model_trigger(_):
            self.show_model_trigger()

        self.show_model.observe(show_model_trigger)

        self.live_link = widgets.Checkbox(
            description="GA Pro - Live link", value=False, indent=False
        )

        def live_link_trigger(_):
            self.live_link_trigger()

        self.live_link.observe(live_link_trigger)

        self.live_link_path = widgets.Text(
            description="",
            value=os.path.join(os.path.dirname(h5file), "Temp"),
            disabled=True,
            style={"description_width": "initial"},
        )

        self.data_panel = widgets.VBox(
            [
                widgets.HBox(
                    [
                        widgets.VBox([self.data_objects, self.data_field]),
                        widgets.VBox([self.system, self.channel_panel]),
                    ]
                ),
                self.lines.widget,
                plotting,
                self.x_label,
                widgets.HBox(
                    [
                        widgets.VBox(
                            [
                                widgets.HBox(
                                    [
                                        widgets.VBox(
                                            [self.zoom, scale_panel],
                                            layout=widgets.Layout(width="50%"),
                                        ),
                                        widgets.VBox(
                                            [
                                                self.smoothing,
                                                self.residual,
                                                self.threshold,
                                            ],
                                            layout=widgets.Layout(width="50%"),
                                        ),
                                    ]
                                ),
                                widgets.HBox([self.markers, self.auto_picker]),
                                widgets.VBox(
                                    [
                                        widgets.Label("Groups"),
                                        widgets.HBox(
                                            [
                                                widgets.Label("Defaults"),
                                                self.group_default_early,
                                                self.group_default_middle,
                                                self.group_default_late,
                                            ]
                                        ),
                                        widgets.HBox(
                                            [
                                                self.group_list,
                                                widgets.VBox(
                                                    [
                                                        self.channels,
                                                        self.group_name,
                                                        self.group_color,
                                                        self.group_add,
                                                    ]
                                                ),
                                            ]
                                        ),
                                        widgets.HBox(
                                            [
                                                self.export,
                                                widgets.VBox(
                                                    [
                                                        self.live_link,
                                                        widgets.Label(
                                                            "Monitoring folder"
                                                        ),
                                                        self.live_link_path,
                                                    ]
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                            ]
                        )
                    ]
                ),
            ]
        )

        self._widget = widgets.VBox([self.data_panel, self.show_model])

    def set_default_groups(self, channels):
        """
        Assign TEM channel for given gate #
        """
        # Reset channels
        for group in self.groups.values():
            if len(group["defaults"]) > 0:
                group["channels"] = []
                group["inflx_up"] = []
                group["inflx_dwn"] = []
                group["peaks"] = []
                group["mad_tau"] = []
                group["times"] = []
                group["values"] = []
                group["locations"] = []

        for ind, channel in enumerate(channels):
            for group in self.groups.values():
                if ind in group["gates"]:
                    group["channels"].append(channel)

        self.group_list.options = self.groups.keys()
        self.group_list.value = []

    def add_group(self):
        """
        Add a group to the list of groups
        """
        if self.group_add.value:

            if self.group_name.value not in self.group_list.options:
                self.group_list.options = list(self.group_list.options) + [
                    self.group_name.value
                ]

            self.groups[self.group_name.value] = {
                "color": self.group_color.value,
                "channels": list(self.channels.value),
                "inflx_up": [],
                "inflx_dwn": [],
                "peaks": [],
                "mad_tau": [],
                "times": [],
                "values": [],
                "locations": [],
            }
            self.group_add.value = False

    def export_click(self):
        if self.export.value:
            for group in self.group_list.value:

                for (
                    ind,
                    (channel, locations, peaks, inflx_dwn, inflx_up, vals, times),
                ) in enumerate(
                    zip(
                        self.groups[group]["channels"],
                        self.groups[group]["locations"],
                        self.groups[group]["peaks"],
                        self.groups[group]["inflx_dwn"],
                        self.groups[group]["inflx_up"],
                        self.groups[group]["values"],
                        self.groups[group]["times"],
                    )
                ):

                    if ind == 0:
                        cox_x = self.lines.profile.interp_x(peaks[0])
                        cox_y = self.lines.profile.interp_y(peaks[0])
                        cox_z = self.lines.profile.interp_z(peaks[0])
                        cox = numpy.r_[cox_x, cox_y, cox_z]

                        # Compute average dip
                        left_ratio = numpy.abs(
                            (peaks[1] - inflx_up[1]) / (peaks[0] - inflx_up[0])
                        )
                        right_ratio = numpy.abs(
                            (peaks[1] - inflx_dwn[1]) / (peaks[0] - inflx_dwn[0])
                        )

                        if left_ratio > right_ratio:
                            ratio = right_ratio / left_ratio
                            azm = (
                                450.0
                                - numpy.rad2deg(
                                    numpy.arctan2(
                                        (
                                            self.lines.profile.interp_y(inflx_up[0])
                                            - cox_y
                                        ),
                                        (
                                            self.lines.profile.interp_x(inflx_up[0])
                                            - cox_x
                                        ),
                                    )
                                )
                            ) % 360.0
                        else:
                            ratio = left_ratio / right_ratio
                            azm = (
                                450.0
                                - numpy.rad2deg(
                                    numpy.arctan2(
                                        (
                                            self.lines.profile.interp_y(inflx_dwn[0])
                                            - cox_y
                                        ),
                                        (
                                            self.lines.profile.interp_x(inflx_dwn[0])
                                            - cox_x
                                        ),
                                    )
                                )
                            ) % 360.0

                        dip = numpy.rad2deg(numpy.arcsin(ratio))
                    tau = self.groups[group]["mad_tau"]

                if self.workspace.get_entity(group):
                    points = self.workspace.get_entity(group)[0]
                    azm_data = points.get_data("azimuth")[0]
                    azm_vals = azm_data.values.copy()
                    dip_data = points.get_data("dip")[0]
                    dip_vals = dip_data.values.copy()

                    tau_data = points.get_data("tau")[0]
                    tau_vals = tau_data.values.copy()

                    points.vertices = numpy.vstack(
                        [points.vertices, cox.reshape((1, 3))]
                    )
                    azm_data.values = numpy.hstack([azm_vals, azm])
                    dip_data.values = numpy.hstack([dip_vals, dip])
                    tau_data.values = numpy.hstack([tau_vals, tau])

                else:
                    # if self.workspace.get_entity(group)
                    # parent =
                    points = Points.create(
                        self.workspace, name=group, vertices=cox.reshape((1, 3))
                    )
                    points.add_data(
                        {
                            "azimuth": {"values": numpy.asarray(azm)},
                            "dip": {"values": numpy.asarray(dip)},
                            "tau": {"values": numpy.asarray(tau)},
                            # "Visual Parameters": {"values": self.viz_param}
                        }
                    )
                    group = points.find_or_create_property_group(
                        name="AzmDip", property_group_type="Dip direction & dip"
                    )
                    group.properties = [
                        points.get_data("azimuth")[0].uid,
                        points.get_data("dip")[0].uid,
                    ]

                if self.live_link.value:
                    if not os.path.exists(self.live_link_path.value):
                        os.mkdir(self.live_link_path.value)

                    temp_geoh5 = os.path.join(
                        self.live_link_path.value, f"temp{time.time():.3f}.geoh5"
                    )
                    ws_out = Workspace(temp_geoh5)
                    points.copy(parent=ws_out)

            self.export.value = False
            self.workspace.finalize()

    def highlight_selection(self):
        """
        Highlight the group choice
        """
        highlights = []
        for group in self.group_list.value:
            highlights += self.groups[group]["channels"]
            self.group_color.value = self.groups[group]["color"]
        self.channels.value = highlights

    def live_link_trigger(self):
        """
        Enable the monitoring folder
        """
        if self.live_link.value:
            self.live_link_path.disabled = False
        else:
            self.live_link_path.disabled = True

    def plot_data_selection(
        self,
        ind,
        smoothing,
        residual,
        markers,
        scale,
        scale_value,
        center,
        focus,
        groups,
        pick_trigger,
        x_label,
        threshold,
    ):

        fig = plt.figure(figsize=(12, 8))
        ax2 = plt.subplot()

        self.line_update()

        for group in self.groups.values():
            group["inflx_up"] = []
            group["inflx_dwn"] = []
            group["peaks"] = []
            group["mad_tau"]
            group["times"] = []
            group["values"] = []
            group["locations"] = []

        if (
            getattr(self, "survey", None) is None
            or getattr(self.lines, "profile", None) is None
        ):
            return

        center_x = center * self.lines.profile.locations_resampled[-1]

        if residual != self.lines.profile.residual:
            self.lines.profile.residual = residual
            self.line_update()

        if smoothing != self.lines.profile.smoothing:
            self.lines.profile.smoothing = smoothing
            self.line_update()

        lims = numpy.searchsorted(
            self.lines.profile.locations_resampled,
            [
                (center - focus / 2.0) * self.lines.profile.locations_resampled[-1],
                (center + focus / 2.0) * self.lines.profile.locations_resampled[-1],
            ],
        )

        sub_ind = numpy.arange(lims[0], lims[1])

        channels = []
        for group in self.group_list.value:
            channels += self.groups[group]["channels"]

        if len(channels) == 0:
            channels = self.channels.options

        times = {}
        for channel in self.data_channel_options.values():
            times[channel.children[0].value] = channel.children[1].value

        y_min, y_max = numpy.inf, -numpy.inf
        for channel, d in self.data.items():

            if channel not in times.keys():
                continue

            if channel not in channels:
                continue

            self.lines.profile.values = d.values[self.survey.line_indices].copy()
            locs, values = (
                self.lines.profile.locations_resampled[sub_ind],
                self.lines.profile.values_resampled[sub_ind],
            )

            y_min = numpy.min([values.min(), y_min])
            y_max = numpy.max([values.max(), y_max])

            ax2.plot(locs, values, color=[0.5, 0.5, 0.5, 1])

            if not residual:
                raw = self.lines.profile._values_resampled_raw[sub_ind]
                ax2.fill_between(
                    locs, values, raw, where=raw > values, color=[1, 0, 0, 0.5]
                )
                ax2.fill_between(
                    locs, values, raw, where=raw < values, color=[0, 0, 1, 0.5]
                )

            dx = self.lines.profile.derivative(order=1)[sub_ind]
            ddx = self.lines.profile.derivative(order=2)[sub_ind]

            peaks = numpy.where((numpy.diff(numpy.sign(dx)) != 0) * (ddx[1:] < 0))[0]
            lows = numpy.where((numpy.diff(numpy.sign(dx)) != 0) * (ddx[1:] > 0))[0]

            up_inflx = numpy.where((numpy.diff(numpy.sign(ddx)) != 0) * (dx[1:] > 0))[0]
            dwn_inflx = numpy.where((numpy.diff(numpy.sign(ddx)) != 0) * (dx[1:] < 0))[
                0
            ]

            if markers:
                ax2.scatter(locs[peaks], values[peaks], s=100, color="r", marker="v")
                ax2.scatter(locs[lows], values[lows], s=100, color="b", marker="^")
                ax2.scatter(locs[up_inflx], values[up_inflx], color="g")
                ax2.scatter(locs[dwn_inflx], values[dwn_inflx], color="m")

            if len(peaks) == 0 or len(lows) < 2:
                continue

            if self.auto_picker.value:

                # Check dipping direction
                x_ind = numpy.min(
                    [values.shape[0] - 2, numpy.searchsorted(locs, center_x)]
                )
                if values[x_ind] > values[x_ind + 1]:
                    cox = peaks[numpy.searchsorted(locs[peaks], center_x) - 1]
                else:
                    x_ind = numpy.min(
                        [peaks.shape[0] - 1, numpy.searchsorted(locs[peaks], center_x)]
                    )
                    cox = peaks[x_ind]

                start = lows[numpy.searchsorted(locs[lows], locs[cox]) - 1]
                end = lows[
                    numpy.min(
                        [numpy.searchsorted(locs[lows], locs[cox]), len(lows) - 1]
                    )
                ]

                bump_x = locs[start:end]
                bump_v = values[start:end]

                if len(bump_x) == 0:
                    continue

                inflx_up = numpy.searchsorted(
                    bump_x,
                    locs[up_inflx[numpy.searchsorted(locs[up_inflx], locs[cox]) - 1]],
                )
                inflx_up = numpy.max([0, inflx_up])

                inflx_dwn = numpy.searchsorted(
                    bump_x,
                    locs[dwn_inflx[numpy.searchsorted(locs[dwn_inflx], locs[cox])]],
                )
                inflx_dwn = numpy.min([bump_x.shape[0] - 1, inflx_dwn])

                peak = numpy.min(
                    [bump_x.shape[0] - 1, numpy.searchsorted(bump_x, locs[cox])]
                )

                for ii, group in enumerate(self.group_list.value):
                    if channel in self.groups[group]["channels"]:
                        self.groups[group]["inflx_up"].append(
                            numpy.r_[bump_x[inflx_up], bump_v[inflx_up]]
                        )
                        self.groups[group]["peaks"].append(
                            numpy.r_[bump_x[peak], bump_v[peak]]
                        )
                        self.groups[group]["times"].append(times[channel])
                        self.groups[group]["inflx_dwn"].append(
                            numpy.r_[bump_x[inflx_dwn], bump_v[inflx_dwn]]
                        )
                        self.groups[group]["locations"].append(bump_x)
                        self.groups[group]["values"].append(bump_v)

                        # Compute average dip
                        left_ratio = (bump_v[peak] - bump_v[inflx_up]) / (
                            bump_x[peak] - bump_x[inflx_up]
                        )
                        right_ratio = (bump_v[peak] - bump_v[inflx_dwn]) / (
                            bump_x[inflx_dwn] - bump_x[peak]
                        )

                        if left_ratio > right_ratio:
                            ratio = right_ratio / left_ratio
                            ori = "left"
                        else:
                            ratio = left_ratio / right_ratio
                            ori = "right"

                        dip = numpy.rad2deg(numpy.arcsin(ratio))

                        # Left
                        ax2.plot(
                            bump_x[:peak],
                            bump_v[:peak],
                            "--",
                            color=self.groups[group]["color"],
                        )
                        # Right
                        ax2.plot(
                            bump_x[peak:],
                            bump_v[peak:],
                            color=self.groups[group]["color"],
                        )
                        ax2.scatter(
                            self.groups[group]["peaks"][-1][0],
                            self.groups[group]["peaks"][-1][1],
                            s=100,
                            c=self.groups[group]["color"],
                            marker=self.marker[ori],
                        )
                        if ~numpy.isnan(dip):
                            ax2.text(
                                self.groups[group]["peaks"][-1][0],
                                self.groups[group]["peaks"][-1][1],
                                f"{dip:.0f}",
                                va="bottom",
                                ha="center",
                            )
                        ax2.scatter(
                            self.groups[group]["inflx_dwn"][-1][0],
                            self.groups[group]["inflx_dwn"][-1][1],
                            s=100,
                            c=self.groups[group]["color"],
                            marker="1",
                        )
                        ax2.scatter(
                            self.groups[group]["inflx_up"][-1][0],
                            self.groups[group]["inflx_up"][-1][1],
                            s=100,
                            c=self.groups[group]["color"],
                            marker="2",
                        )

        ax2.plot(
            [
                self.lines.profile.locations_resampled[0],
                self.lines.profile.locations_resampled[-1],
            ],
            [0, 0],
            "r",
        )
        ax2.plot([center_x, center_x], [0, y_min], "r--")
        ax2.scatter(center_x, y_min, s=20, c="r", marker="^")

        for group in self.groups.values():
            if group["peaks"]:
                peaks = numpy.vstack(group["peaks"])
                ratio = peaks[:, 1] / peaks[0, 1]
                ind = numpy.where(ratio >= (threshold / 100))[0][-1]
                #                 print(ind)
                ax2.plot(
                    peaks[: ind + 1, 0], peaks[: ind + 1, 1], "--", color=group["color"]
                )
        #                 ax2.plot([peaks[0, 0], peaks[0, 0]], [peaks[0, 1], peaks[-1, 1]], '--', color='k')
        #                 ax2.plot(
        #                     [group['inflx_up'][ind][0], group['inflx_dwn'][ind][0]]
        #                     [group['inflx_up'][ind][1], group['inflx_dwn'][ind][1]], '--', color=[0.5,0.5,0.5]
        #                 )

        if scale == "symlog":
            plt.yscale("symlog", linthreshy=scale_value)

        x_lims = [
            center_x - focus / 2.0 * self.lines.profile.locations_resampled[-1],
            center_x + focus / 2.0 * self.lines.profile.locations_resampled[-1],
        ]
        ax2.set_xlim(x_lims)
        ax2.set_title(f"Line: {ind}")
        ax2.set_ylabel("dBdT")

        if x_label == "Easting":

            ax2.text(
                center_x,
                0,
                f"{self.lines.profile.interp_x(center_x):.0f} m E",
                va="top",
                ha="center",
                bbox={"edgecolor": "r"},
            )
            xlbl = [
                f"{self.lines.profile.interp_x(label):.0f}"
                for label in ax2.get_xticks()
            ]
            ax2.set_xticklabels(xlbl)
            ax2.set_xlabel("Easting (m)")
        elif x_label == "Northing":
            ax2.text(
                center_x,
                0,
                f"{self.lines.profile.interp_y(center_x):.0f} m N",
                va="top",
                ha="center",
                bbox={"edgecolor": "r"},
            )
            xlbl = [
                f"{self.lines.profile.interp_y(label):.0f}"
                for label in ax2.get_xticks()
            ]
            ax2.set_xticklabels(xlbl)
            ax2.set_xlabel("Northing (m)")
        else:
            ax2.text(
                center_x,
                0,
                f"{center_x:.0f} m",
                va="top",
                ha="center",
                bbox={"edgecolor": "r"},
            )
            ax2.set_xlabel("Distance (m)")

        ax2.grid(True)

        pos2 = ax2.get_position()

        ax = [pos2.x0, pos2.y0, pos2.width, pos2.height].copy()

        if self.auto_picker.value:
            ax[0] += 0.25
            ax[1] -= 0.5
            ax[2] /= 3
            ax[3] /= 2
            ax4 = plt.axes(ax)
            for group in self.group_list.value:
                if len(self.groups[group]["peaks"]) == 0:
                    continue

                peaks = (
                    numpy.vstack(self.groups[group]["peaks"])
                    * self.em_system_specs[self.system.value]["normalization"]
                )

                tc = numpy.hstack(self.groups[group]["times"][: ind + 1])
                vals = numpy.log(peaks[: ind + 1, 1])

                if tc.shape[0] < 2:
                    continue
                # Compute linear trend
                A = numpy.c_[numpy.ones_like(tc), tc]
                a, c = numpy.linalg.solve(numpy.dot(A.T, A), numpy.dot(A.T, vals))
                d = numpy.r_[tc.min(), tc.max()]
                vv = d * c + a

                ratio = numpy.abs((vv[0] - vv[1]) / (d[0] - d[1]))
                #                 angl = numpy.arctan(ratio**-1.)

                self.groups[group]["mad_tau"] = ratio ** -1.0

                ax4.plot(
                    d,
                    numpy.exp(d * c + a),
                    "--",
                    linewidth=2,
                    color=self.groups[group]["color"],
                )
                ax4.text(
                    numpy.mean(d),
                    numpy.exp(numpy.mean(vv)),
                    f"{ratio ** -1.:.2e}",
                    color=self.groups[group]["color"],
                )
                #                 plt.yscale('symlog', linthreshy=scale_value)
                #                 ax4.set_aspect('equal')
                ax4.scatter(
                    numpy.hstack(self.groups[group]["times"]),
                    peaks[:, 1],
                    color=self.groups[group]["color"],
                    marker="^",
                )
                ax4.grid(True)

            plt.yscale("symlog")
            ax4.yaxis.set_label_position("right")
            ax4.yaxis.tick_right()
            ax4.set_ylabel("log(V)")
            ax4.set_xlabel("Time (sec)")
            ax4.set_title("Decay - MADTau")

    def plot_model_selection(self, ind, center, focus):

        fig = plt.figure(figsize=(12, 8))
        ax3 = plt.subplot()

        if (
            getattr(self, "survey", None) is None
            or getattr(self.lines, "profile", None) is None
        ):
            return

        center_x = center * self.lines.profile.locations_resampled[-1]

        x_lims = [
            center_x - focus / 2.0 * self.lines.profile.locations_resampled[-1],
            center_x + focus / 2.0 * self.lines.profile.locations_resampled[-1],
        ]

        if getattr(self.lines, "model_x", None) is not None:
            return

            cs = ax3.tricontourf(
                self.lines.model_x,
                self.lines.model_z,
                self.lines.model_cells.reshape((-1, 3)),
                numpy.log10(self.lines.model_values),
                levels=numpy.linspace(-3, 0.5, 25),
                vmin=-2,
                vmax=-0.75,
                cmap="rainbow",
            )
            ax3.tricontour(
                self.lines.model_x,
                self.lines.model_z,
                self.lines.model_cells.reshape((-1, 3)),
                numpy.log10(self.lines.model_values),
                levels=numpy.linspace(-3, 0.5, 25),
                colors="k",
                linestyles="solid",
                linewidths=0.5,
            )
            #         ax3.scatter(center_x, center_z, 100, c='r', marker='x')
            ax3.set_xlim(x_lims)
            ax3.set_aspect("equal")
            ax3.grid(True)

    def line_update(self):
        """
        Re-compute derivatives
        """

        if getattr(self, "survey", None) is None:
            return

        if (
            len(self.survey.get_data(self.lines.value.value)) == 0
            or self.lines.lines.value == ""
        ):
            return

        line_ind = numpy.where(
            numpy.asarray(self.survey.get_data(self.lines.value.value)[0].values)
            == self.lines.lines.value
        )[0]

        if len(line_ind) == 0:
            return

        self.survey.line_indices = line_ind
        xyz = self.survey.vertices[line_ind, :]

        if numpy.std(xyz[:, 1]) > numpy.std(xyz[:, 0]):
            start = numpy.argmin(xyz[:, 1])
        else:
            start = numpy.argmin(xyz[:, 0])

        self.lines.profile = signal_processing_1d(
            xyz, None, smoothing=self.smoothing.value, residual=self.residual.value
        )

        # Get the corresponding along line model
        origin = xyz[0, :2]

        if self.workspace.get_entity(self.model_objects.value):
            surf_model = self.workspace.get_entity(self.model_objects.value)[0]

        if surf_model.get_data("Line") and numpy.any(
            numpy.where(
                surf_model.get_data("Line")[0].values == self.lines.lines.value
            )[0]
        ):

            surf_id = surf_model.get_data("Line")[0].values
            #             surf_ind = numpy.where(
            #                 surf_id == self.lines.lines.value
            #             )[0]

            cell_ind = numpy.where(
                surf_id[surf_model.cells[:, 0]] == self.lines.lines.value
            )[0]

            cells = surf_model.cells[cell_ind, :]
            vert_ind, cell_ind = numpy.unique(cells, return_inverse=True)

            surf_verts = surf_model.vertices[vert_ind, :]
            self.lines.model_x = numpy.linalg.norm(
                numpy.c_[
                    xyz[start, 0] - surf_verts[:, 0], xyz[start, 1] - surf_verts[:, 1]
                ],
                axis=1,
            )
            self.lines.model_z = surf_model.vertices[vert_ind, 2]
            self.lines.model_cells = cell_ind
            self.lines.model_values = surf_model.get_data(self.model_field.value)[
                0
            ].values[vert_ind]
        else:
            self.lines.model_x = None

    def reset_default_bounds(self):

        try:
            first, last = numpy.asarray(
                self.group_default_early.value.split("-"), dtype="int"
            )
            self.early = numpy.arange(first, last + 1).tolist()
        except ValueError:
            return

        try:
            first, last = numpy.asarray(
                self.group_default_middle.value.split("-"), dtype="int"
            )
            self.middle = numpy.arange(first, last + 1).tolist()
        except ValueError:
            return

        try:
            first, last = numpy.asarray(
                self.group_default_late.value.split("-"), dtype="int"
            )
            self.last = numpy.arange(first, last + 1).tolist()
        except ValueError:
            return

        for group in self.groups.values():
            gates = []
            if len(group["defaults"]) > 0:
                for default in group["defaults"]:
                    gates += getattr(self, default)
                group["gates"] = gates

        self.set_default_groups(self.channels.options)

    def show_model_trigger(self):
        """
        Add the model widget
        """
        if self.show_model.value:
            self._widget.children = [self.data_panel, self.show_model, self.model_panel]
        else:
            self._widget.children = [self.data_panel, self.show_model]

    @property
    def widget(self):
        return self._widget
