import re
import os
import time

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
from geoapps.utils import format_labels
from geoapps.inversion import TopographyOptions
from geoapps.selection import (
    ObjectDataSelection,
    LineOptions,
)
from geoapps.plotting import plot_plan_data_selection
from geoapps.utils import (
    filter_xy,
    export_grid_2_geotiff,
    geotiff_2_grid,
    rotate_xy,
    find_value,
    signal_processing_1d,
)


def calculator(h5file):
    w_s = Workspace(h5file)
    selection = ObjectDataSelection(h5file=h5file, select_multiple=False)
    store = ObjectDataSelection(
        h5file=h5file, objects=selection.objects, select_multiple=False
    ).data
    store.description = "Assign result to: "
    use = widgets.ToggleButton(description=">> Add >>")
    add = widgets.ToggleButton(description=">> Create >>")
    compute = widgets.ToggleButton(description="Compute: ", button_style="success")
    channel = widgets.Text("NewChannel", description="Name: ")
    equation = widgets.Textarea(layout=Layout(width="75%"))

    var = {}

    def evaluate(var):
        vals = eval(equation.value)
        obj = w_s.get_entity(selection.objects.value)[0]
        obj.get_data(store.value)[0].values = vals
        print(vals)
        w_s.finalize()

    def click_add(_):
        if add.value:
            obj = w_s.get_entity(selection.objects.value)[0]

            if getattr(obj, "vertices", None) is not None:
                new_data = obj.add_data(
                    {channel.value: {"values": numpy.zeros(obj.n_vertices)}}
                )
            else:
                new_data = obj.add_data(
                    {channel.value: {"values": numpy.zeros(obj.n_cells)}}
                )

            selection.data.options = obj.get_data_list()
            store.options = [new_data.name]
            store.value = new_data.name

            add.value = False

    def click_use(_):
        if use.value:
            name = selection.objects.value + "." + selection.data.value
            if name not in var.keys():
                obj = w_s.get_entity(selection.objects.value)[0]
                var[name] = obj.get_data(selection.data.value)[0].values

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
            selection.objects,
            HBox([use, selection.data]),
            VBox(
                [equation, HBox([add, channel]), store, compute],
                layout=Layout(width="100%"),
            ),
        ]
    )


class CDICurve2Surface(ObjectDataSelection):
    """
    Application for the conversion of conductivity/depth curves to
    a pseudo 3D conductivity model on surface.
    """

    defaults = {
        "add_groups": True,
        "select_multiple": True,
        "h5file": "../../assets/FlinFlon.geoh5",
        "objects": "CDI_VTEM_model",
        "data": ["COND"],
        "max_distance": 100,
        "max_depth": 400,
        "elevations": {"data": "ELEV"},
        "lines": {"data": "Line"},
    }

    def __init__(self, **kwargs):
        kwargs = self.apply_defaults(**kwargs)
        self._lines = ObjectDataSelection(add_groups=True, find_value=["line"])
        self._topography = TopographyOptions()
        self._elevations = ObjectDataSelection(add_groups=True)
        self._z_option = widgets.RadioButtons(
            options=["elevation", "depth"], description="Layers reference:",
        )
        self._max_depth = widgets.FloatText(description="Max depth (m):",)
        self._max_distance = widgets.FloatText(
            value=50, description="Max distance (m):",
        )
        self._tolerance = widgets.FloatText(value=1, description="Tolerance (m):")
        self._save_name = widgets.Text("CDI_", description="Name: ")
        self._convert = widgets.ToggleButton(
            description="Convert >>", button_style="success"
        )

        super().__init__(**kwargs)

        if "lines" in kwargs.keys():
            self.lines.__populate__(**kwargs["lines"])
        if "topography" in kwargs.keys():
            self.topography.__populate__(**kwargs["topography"])
        if "elevations" in kwargs.keys():
            self.elevations.__populate__(**kwargs["elevations"])

        self.lines.data.description = "Line field:"
        self.elevations.data.description = "Elevations:"
        self.data.description = "Model fields: "

        def z_options_change(_):
            self.z_options_change()

        self.z_option.observe(z_options_change)
        self.depth_panel = widgets.HBox([self.z_option, self.elevations.data])

        def convert_trigger(_):
            self.convert_trigger()

        self.convert.observe(convert_trigger)
        self._widget = widgets.VBox(
            [
                self.project_panel,
                widgets.VBox(
                    [
                        widgets.VBox(
                            [
                                self.widget,
                                self.lines.data,
                                self.depth_panel,
                                widgets.Label("Triangulation"),
                                self.max_depth,
                                self.max_distance,
                                self.tolerance,
                            ]
                        ),
                    ]
                ),
                widgets.Label("Output"),
                widgets.HBox([self.convert, self.save_name,]),
            ]
        )

    def convert_trigger(self):

        if self.convert.value:
            if self.workspace.get_entity(self.objects.value):
                curve = self.workspace.get_entity(self.objects.value)[0]
                self.convert.value = False
            else:
                self.convert.button_style = "warning"
                self.convert.value = False
                return

            lines_id = curve.get_data(self.lines.data.value)[0].values
            lines = numpy.unique(lines_id).tolist()

            if self.z_option.value == "depth":
                if self.topography.options_button.value == "Object":

                    topo_obj = self.workspace.get_entity(self.topography.objects.value)[
                        0
                    ]

                    if hasattr(topo_obj, "centroids"):
                        vertices = topo_obj.centroids.copy()
                    else:
                        vertices = topo_obj.vertices.copy()

                    topo_xy = vertices[:, :2]

                    if self.topography.value.value == "Vertices":
                        topo_z = vertices[:, 2]
                    else:
                        topo_z = topo_obj.get_data(self.topography.value.value)[
                            0
                        ].values

                else:
                    topo_xy = curve.vertices[:, :2].copy()

                    if self.topography.options_button.value == "Constant":
                        topo_z = (
                            numpy.ones_like(curve.vertices[:, 2])
                            * self.topography.constant.value
                        )
                    else:
                        topo_z = (
                            numpy.ones_like(curve.vertices[:, 2])
                            + self.topography.offset.value
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
                for ind, z_prop in enumerate(
                    curve.get_property_group(self.elevations.data.value).properties,
                ):
                    nZ += 1
                    z_vals = self.workspace.get_entity(z_prop)[0].values[line_ind]

                    m_vals = []
                    for m in self.data.value:
                        prop = curve.get_property_group(m).properties[ind]
                        m_vals.append(
                            self.workspace.get_entity(prop)[0].values[line_ind]
                        )

                    m_vals = numpy.vstack(m_vals).T
                    keep = (
                        (z_vals > 1e-38)
                        * (z_vals < 2e-38)
                        * numpy.any((m_vals > 1e-38) * (m_vals < 2e-38), axis=1)
                    ) == False
                    keep[numpy.isnan(z_vals)] = False
                    keep[numpy.any(numpy.isnan(m_vals), axis=1)] = False

                    X.append(xyz[:, 0][order][keep])
                    Y.append(xyz[:, 1][order][keep])

                    if self.z_option.value == "depth":
                        z_topo = topo(xyz[:, 0][order][keep], xyz[:, 1][order][keep])

                        nan_z = numpy.isnan(z_topo)
                        if numpy.any(nan_z):
                            _, ii = tree_topo.query(xyz[:, :2][order][keep][nan_z])
                            z_topo[nan_z] = topo_z[ii]

                        Z.append(z_topo + z_vals[order][keep])

                    else:
                        Z.append(z_vals[order][keep])

                    M.append(m_vals[order, :][keep, :])

                    if ind == 0:

                        x_loc = xyz[:, 0][order][keep]
                        y_loc = xyz[:, 1][order][keep]
                        z_loc = Z[0]

                X = numpy.hstack(X)
                Y = numpy.hstack(Y)
                Z = numpy.hstack(Z)
                model.append(numpy.vstack(M))
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
                        (z <= (topo_top(x) + self.tolerance.value))
                        * (
                            z
                            >= (
                                topo_top(x)
                                - self.max_depth.value
                                - self.tolerance.value
                            )
                        )
                        * (length < self.max_distance.value)
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
                self.workspace,
                name=self.save_name.value,
                vertices=numpy.vstack(model_vertices),
                cells=numpy.vstack(model_cells),
            )

            surface.add_data(
                {"Line": {"values": numpy.hstack(line_ids)},}
            )

            models = numpy.vstack(model)
            for ind, field in enumerate(self.data.value):

                surface.add_data(
                    {field: {"values": models[:, ind]},}
                )

    def z_options_change(self):
        if self.z_option.value == "depth":
            self.elevations.data.description = "Depth:"
            self.depth_panel.children = [
                self.z_option,
                widgets.VBox(
                    [
                        self.elevations.data,
                        widgets.Label("Topography"),
                        self.topography.widget,
                    ]
                ),
            ]
        else:
            self.elevations.data.description = "Elevation:"
            self.depth_panel.children = [self.z_option, self.elevations.data]

    @property
    def lines(self):
        """
            Line field options
        """
        return self._lines

    @property
    def topography(self):
        """
            TopographyOptions()
        """
        return self._topography

    @property
    def elevations(self):
        """
            ObjectDataSelection()
        """
        return self._elevations

    @elevations.setter
    def elevations(self, value):
        assert isinstance(
            value, ObjectDataSelection
        ), f"elevations must be an object of type {ObjectDataSelection}"
        self._elevations = value

    @property
    def z_option(self):
        """
            widgets.RadioButtons()
        """
        return self._z_option

    @property
    def max_depth(self):
        """
            widgets.FloatText()
        """
        return self._max_depth

    @property
    def max_distance(self):
        """
            widgets.FloatText()
        """
        return self._max_distance

    @property
    def tolerance(self):
        """
            widgets.FloatText()
        """
        return self._tolerance

    @property
    def save_name(self):
        """
            widgets.Text()
        """
        return self._save_name

    @property
    def convert(self):
        """
            widgets.ToggleButton()
        """
        return self._convert

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

        # Refresh the list of objects
        self.update_objects_list()

        self.lines.workspace = workspace
        self.lines.objects = self.objects

        self.elevations.workspace = workspace
        self.elevations.objects = self.objects


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
                                child, temp_file, epsg_in, data_type="float"
                            )

                            grid = gdal.Open(temp_file)
                            gdal.Warp(
                                temp_file_in, grid, dstSRS="EPSG:" + str(int(epsg_out))
                            )
                            print("EPSG:" + str(int(epsg_out)))
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
