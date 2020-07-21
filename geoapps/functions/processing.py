import re

import ipywidgets as widgets
import matplotlib.colors as colors
import matplotlib.pyplot as plt
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

from .plotting import format_labels, plot_plan_data_selection
from .inversion import TopographyOptions
from .selection import object_data_selection_widget, plot_plan_data_selection
from .utils import export_grid_2_geotiff, geotiff_2_grid, rotate_xy


def calculator(h5file):
    w_s = Workspace(h5file)

    objects, data = object_data_selection_widget(h5file, select_multiple=False)
    _, store = object_data_selection_widget(
        h5file, objects=objects, select_multiple=False
    )
    store.description = "Assign result to: "
    store.style = {"description_width": "initial"}

    # selection = Select(description="Math")
    use = widgets.ToggleButton(description=">> Add >>")
    add = widgets.ToggleButton(
        description=">> Create >>", style={"description_width": "initial"}
    )
    compute = widgets.ToggleButton(description="Compute: ", button_style="success")
    channel = widgets.Text("NewChannel", description="Name: ")
    equation = widgets.Textarea(layout=Layout(width="75%"))

    var = {}

    def evaluate(var):
        # try:
        vals = eval(equation.value)
        obj = w_s.get_entity(objects.value)[0]
        obj.get_data(store.value)[0].values = vals

        print(vals)
        w_s.finalize()
        #
        # except:
        #     print("Error. Check inputs")

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
            store.options = new_data.name
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
            values = new_cmap["Value"]
            values -= values.min()
            values /= values.max()

            cdict = {
                "red": numpy.c_[
                    values, new_cmap["Red"] / 255, new_cmap["Red"] / 255
                ].tolist(),
                "green": numpy.c_[
                    values, new_cmap["Green"] / 255, new_cmap["Green"] / 255
                ].tolist(),
                "blue": numpy.c_[
                    values, new_cmap["Blue"] / 255, new_cmap["Blue"] / 255
                ].tolist(),
            }
            cmap = colors.LinearSegmentedColormap(
                "custom_map", segmentdata=cdict, N=len(values)
            )

        else:
            cmap = None

        # Parse contour values
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

                axs.pcolormesh(xx, yy, grid_data, cmap=cmap)
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

    export_as = widgets.Text(
        value=data.value + "_" + contours.value, indent=False, disabled=False
    )

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

    def compute_plot(
        entity_name,
        data_name,
        resolution,
        center_x,
        center_y,
        width_x,
        width_y,
        azimuth,
        zoom_extent,
        sigma,
        threshold,
        line_length,
        line_gap,
        export_as,
        export,
    ):

        if workspace.get_entity(entity_name):
            obj = workspace.get_entity(entity_name)[0]

            assert isinstance(
                obj, Grid2D
            ), "This application is only designed for Grid2D objects"

            if obj.get_data(data_name):
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
                data_obj = obj.get_data(data_name)[0]
                _, ind_filter, _ = plot_plan_data_selection(
                    obj,
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
                    },
                )
                data_count.value = f"Data Count: {ind_filter.sum()}"

                ind_x, ind_y = (
                    numpy.any(ind_filter, axis=1),
                    numpy.any(ind_filter, axis=0),
                )

                grid_data = data_obj.values.reshape(ind_filter.shape, order="F")[
                    ind_x, :
                ]
                grid_data = grid_data[:, ind_y]

                X = obj.centroids[:, 0].reshape(ind_filter.shape, order="F")[ind_x, :]
                X = X[:, ind_y]

                Y = obj.centroids[:, 1].reshape(ind_filter.shape, order="F")[ind_x, :]
                Y = Y[:, ind_y]

                # Parameters controlling the edge detection
                edges = canny(grid_data.T, sigma=sigma, use_quantiles=True)

                # Parameters controlling the line detection
                lines = probabilistic_hough_line(
                    edges,
                    line_length=line_length,
                    threshold=threshold,
                    line_gap=line_gap,
                    seed=0,
                )

                if data_obj.entity_type.color_map is not None:
                    new_cmap = data_obj.entity_type.color_map.values
                    values = new_cmap["Value"]
                    values -= values.min()
                    values /= values.max()

                    cdict = {
                        "red": numpy.c_[
                            values, new_cmap["Red"] / 255, new_cmap["Red"] / 255
                        ].tolist(),
                        "green": numpy.c_[
                            values, new_cmap["Green"] / 255, new_cmap["Green"] / 255
                        ].tolist(),
                        "blue": numpy.c_[
                            values, new_cmap["Blue"] / 255, new_cmap["Blue"] / 255
                        ].tolist(),
                    }
                    cmap = colors.LinearSegmentedColormap(
                        "custom_map", segmentdata=cdict, N=len(values)
                    )

                else:
                    cmap = None

                xy = []
                cells = []
                count = 0
                for line in lines:
                    p0, p1 = line

                    points = numpy.r_[
                        numpy.c_[X[p0[0], 0], Y[0, p0[1]], 0],
                        numpy.c_[X[p1[0], 0], Y[0, p1[1]], 0],
                    ]
                    xy.append(points)

                    cells.append(numpy.c_[count, count + 1].astype("uint32"))

                    count += 2

                    plt.plot(points[:, 0], points[:, 1], "k--", linewidth=2)

                if export:
                    # Save the result to geoh5
                    curve = Curve.create(
                        obj.workspace,
                        name=export_as,
                        vertices=numpy.vstack(xy),
                        cells=numpy.vstack(cells),
                    )

                return lines

    objects, data_obj = object_data_selection_widget(h5file)

    # Fetch vertices in the project
    lim_x = [1e8, 0]
    lim_y = [1e8, 0]

    obj = workspace.get_entity(objects.value)[0]
    if obj.vertices is not None:
        lim_x[0], lim_x[1] = (
            numpy.min([lim_x[0], obj.vertices[:, 0].min()]),
            numpy.max([lim_x[1], obj.vertices[:, 0].max()]),
        )
        lim_y[0], lim_y[1] = (
            numpy.min([lim_y[0], obj.vertices[:, 1].min()]),
            numpy.max([lim_y[1], obj.vertices[:, 1].max()]),
        )
    elif hasattr(obj, "centroids"):
        lim_x[0], lim_x[1] = (
            numpy.min([lim_x[0], obj.centroids[:, 0].min()]),
            numpy.max([lim_x[1], obj.centroids[:, 0].max()]),
        )
        lim_y[0], lim_y[1] = (
            numpy.min([lim_y[0], obj.centroids[:, 1].min()]),
            numpy.max([lim_y[1], obj.centroids[:, 1].max()]),
        )

    center_x = widgets.FloatSlider(
        min=lim_x[0],
        max=lim_x[1],
        value=numpy.mean(lim_x),
        steps=10,
        description="Easting",
        continuous_update=False,
    )
    center_y = widgets.FloatSlider(
        min=lim_y[0],
        max=lim_y[1],
        value=numpy.mean(lim_y),
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
        max=lim_x[1] - lim_x[0],
        min=100,
        value=(lim_x[1] - lim_x[0]) / 2,
        steps=10,
        description="Width",
        continuous_update=False,
    )
    width_y = widgets.FloatSlider(
        max=lim_y[1] - lim_y[0],
        min=100,
        value=(lim_y[1] - lim_y[0]) / 2,
        steps=10,
        description="Height",
        continuous_update=False,
        orientation="vertical",
    )
    resolution = widgets.FloatText(
        value=resolution,
        description="Resolution (m)",
        style={"description_width": "initial"},
    )

    data_count = Label("Data Count: 0", tooltip="Keep <1500 for speed")

    zoom_extent = widgets.ToggleButton(
        value=True,
        description="Zoom on selection",
        tooltip="Keep plot extent on selection",
        icon="check",
    )

    # selection_panel = VBox([
    #     Label("Window & Downsample"),
    #     VBox([resolution, data_count,
    #           HBox([
    #               center_y, width_y,
    #               plot_window,
    #           ], layout=Layout(align_items='center')),
    #           VBox([width_x, center_x, azimuth, zoom_extent], layout=Layout(align_items='center'))
    #           ], layout=Layout(align_items='center'))
    # ])

    def saveIt(_):
        if export.value:
            export.value = False
            print(f"Lines {export_as.value} exported to: {workspace.h5file}")

    def saveItAs(_):
        export_as.value = (
            f"S:{sigma.value}"
            + f" T:{threshold.value}"
            + f" LL:{line_length.value}"
            + f" LG:{line_gap.value}"
        )

    export = widgets.ToggleButton(
        value=False,
        description="Export to GA",
        button_style="danger",
        tooltip="Description",
        icon="check",
    )

    export.observe(saveIt)

    sigma = widgets.FloatSlider(
        min=0.0,
        max=10,
        step=0.1,
        value=sigma,
        continuous_update=False,
        description="Sigma",
        style={"description_width": "initial"},
    )

    sigma.observe(saveItAs)

    line_length = widgets.IntSlider(
        min=1.0,
        max=10.0,
        step=1.0,
        value=line_length,
        continuous_update=False,
        description="Line Length",
        style={"description_width": "initial"},
    )
    line_length.observe(saveItAs)

    line_gap = widgets.IntSlider(
        min=1.0,
        max=10.0,
        step=1.0,
        value=line_gap,
        continuous_update=False,
        description="Line Gap",
        style={"description_width": "initial"},
    )
    line_gap.observe(saveItAs)

    threshold = widgets.IntSlider(
        min=1.0,
        max=10.0,
        step=1.0,
        value=threshold,
        continuous_update=False,
        description="Threshold",
        style={"description_width": "initial"},
    )
    threshold.observe(saveItAs)

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

    plot_window = widgets.interactive_output(
        compute_plot,
        {
            "entity_name": objects,
            "data_name": data_obj,
            "resolution": resolution,
            "center_x": center_x,
            "center_y": center_y,
            "width_x": width_x,
            "width_y": width_y,
            "azimuth": azimuth,
            "zoom_extent": zoom_extent,
            "sigma": sigma,
            "threshold": threshold,
            "line_length": line_length,
            "line_gap": line_gap,
            "export_as": export_as,
            "export": export,
        },
    )

    out = VBox(
        [
            objects,
            data_obj,
            VBox(
                [
                    resolution,
                    data_count,
                    HBox(
                        [center_y, width_y, plot_window,],
                        layout=Layout(align_items="center"),
                    ),
                    VBox(
                        [width_x, center_x, azimuth, zoom_extent],
                        layout=Layout(align_items="center"),
                    ),
                ],
                layout=Layout(align_items="center"),
            ),
            VBox([sigma, threshold, line_length, line_gap, export_as, export]),
        ]
    )

    return out
