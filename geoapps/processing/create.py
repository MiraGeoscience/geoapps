import re
from ipywidgets import (
    FloatText,
    HBox,
    Label,
    RadioButtons,
    Text,
    ToggleButton,
    VBox,
    Layout,
    Checkbox,
    interactive_output,
)
import numpy as np
from geoh5py.objects import Surface, Curve, Points
from geoh5py.workspace import Workspace
from geoh5py.io import H5Writer
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree, Delaunay
from geoapps.selection import ObjectDataSelection, TopographyOptions
from geoapps.plotting import PlotSelection2D


class Surface2D(ObjectDataSelection):
    """
    Application for the conversion of conductivity/depth curves to
    a pseudo 3D conductivity model on surface.
    """

    defaults = {
        "add_groups": "only",
        "select_multiple": True,
        "object_types": Curve,
        "h5file": "../../assets/FlinFlon.geoh5",
        "objects": "CDI_VTEM_model",
        "data": ["COND"],
        "max_distance": 250,
        "elevations": {"data": "ELEV"},
        "lines": {"data": "Line"},
    }

    def __init__(self, **kwargs):
        kwargs = self.apply_defaults(**kwargs)
        self._lines = ObjectDataSelection(add_groups=True, find_value=["line"])
        self._topography = TopographyOptions()
        self._elevations = ObjectDataSelection(add_groups="only")
        self._z_option = RadioButtons(
            options=["elevation", "depth"],
            description="Vertical Reference:",
            style={"description_width": "initial"},
        )
        self._type = RadioButtons(
            options=["Sections", "Horizon"],
            value="Sections",
            description="Surface Type:",
            style={"description_width": "initial"},
        )
        self._max_distance = FloatText(description="Max Triangulation Distance (m):",)
        self._export_as = Text("CDI_", description="Surface:")
        self._convert = ToggleButton(description="Convert >>", button_style="success")

        super().__init__(**kwargs)

        self.ga_group_name.value = "CDI"
        if "lines" in kwargs.keys():
            self.lines.__populate__(**kwargs["lines"])
        if "topography" in kwargs.keys():
            self.topography.__populate__(**kwargs["topography"])
        if "elevations" in kwargs.keys():
            self.elevations.__populate__(**kwargs["elevations"])

        self.lines.data.description = "Line field:"
        self.elevations.data.description = "Elevations:"
        self.type.observe(self.type_change, names="value")
        self.data.observe(self.data_change, names="value")
        self.data_change(None)
        self.data.description = "Model fields: "

        def z_options_change(_):
            self.z_options_change()

        self.z_option.observe(z_options_change)
        self.depth_panel = HBox([self.z_option, self.elevations.data])
        self.data_panel = self.main
        self.trigger.on_click(self.compute_trigger)
        self._main = HBox(
            [
                VBox(
                    [
                        self.project_panel,
                        self.data_panel,
                        self.type,
                        self.depth_panel,
                        self.lines.data,
                        self.max_distance,
                        Label("Output"),
                        self.export_as,
                        self.output_panel,
                    ]
                )
            ]
        )

    def compute_trigger(self, _):

        if not self.workspace.get_entity(self.objects.value):
            return

        obj, data_list = self.get_selected_entities()
        _, elevations = self.elevations.get_selected_entities()

        if hasattr(obj, "centroids"):
            locations = obj.centroids
        else:
            locations = obj.vertices

        if self.z_option.value == "depth":
            if self.topography.options.value == "Object":

                topo_obj = self.workspace.get_entity(self.topography.objects.value)[0]

                if hasattr(topo_obj, "centroids"):
                    vertices = topo_obj.centroids.copy()
                else:
                    vertices = topo_obj.vertices.copy()

                topo_xy = vertices[:, :2]

                if topo_obj.get_data(self.topography.data.value):
                    topo_z = topo_obj.get_data(self.topography.data.value)[0].values
                else:
                    topo_z = vertices[:, 2]

            else:
                topo_xy = locations[:, :2].copy()

                if self.topography.options.value == "Constant":
                    topo_z = (
                        np.ones_like(locations[:, 2]) * self.topography.constant.value
                    )
                else:
                    topo_z = (
                        np.ones_like(locations[:, 2]) + self.topography.offset.value
                    )

            surf = Delaunay(topo_xy)
            topo = LinearNDInterpolator(surf, topo_z)
            tree_topo = cKDTree(topo_xy)

        if self.type.value == "Sections":
            lines_id = obj.get_data(self.lines.data.value)[0].values
            lines = np.unique(lines_id).tolist()
            model_vertices = []
            model_cells = []
            model_count = 0
            model = []
            line_ids = []
            for line in lines:

                line_ind = np.where(lines_id == line)[0]

                n_sounding = len(line_ind)
                if n_sounding < 2:
                    continue

                xyz = locations[line_ind, :]

                # Create a 2D mesh to store the results
                if np.std(xyz[:, 1]) > np.std(xyz[:, 0]):
                    order = np.argsort(xyz[:, 1])
                else:
                    order = np.argsort(xyz[:, 0])

                X, Y, Z, M, L = [], [], [], [], []
                # Stack the z-coordinates and model
                nZ = 0

                for ind, elev in enumerate(elevations):
                    # data = self.workspace.get_entity(z_prop)[0]
                    nZ += 1
                    z_vals = elev.values[line_ind]

                    m_vals = []
                    for m in self.data.value:
                        prop = obj.get_property_group(m).properties[ind]
                        m_vals.append(
                            self.workspace.get_entity(prop)[0].values[line_ind]
                        )

                    m_vals = np.vstack(m_vals).T
                    keep = (
                        (z_vals > 1e-38)
                        * (z_vals < 2e-38)
                        * np.any((m_vals > 1e-38) * (m_vals < 2e-38), axis=1)
                    ) == False
                    keep[np.isnan(z_vals)] = False
                    keep[np.any(np.isnan(m_vals), axis=1)] = False

                    X.append(xyz[:, 0][order][keep])
                    Y.append(xyz[:, 1][order][keep])

                    if self.z_option.value == "depth":
                        z_topo = topo(xyz[:, 0][order][keep], xyz[:, 1][order][keep])

                        nan_z = np.isnan(z_topo)
                        if np.any(nan_z):
                            _, ii = tree_topo.query(xyz[:, :2][order][keep][nan_z])
                            z_topo[nan_z] = topo_z[ii]

                        Z.append(z_topo + z_vals[order][keep])

                    else:
                        Z.append(z_vals[order][keep])

                    L.append(
                        np.ones_like(z_vals[order][keep])
                        * -int(re.findall(r"\d+", elev.name)[-1])
                    )
                    M.append(m_vals[order, :][keep, :])

                    if ind == 0:

                        x_loc = xyz[:, 0][order][keep]
                        y_loc = xyz[:, 1][order][keep]
                        z_loc = Z[0]

                X = np.hstack(X)
                Y = np.hstack(Y)
                Z = np.hstack(Z)
                L = np.hstack(L)

                model.append(np.vstack(M))
                line_ids.append(np.ones_like(Z.ravel()) * line)

                if np.std(y_loc) > np.std(x_loc):
                    tri2D = Delaunay(np.c_[np.ravel(Y), np.ravel(L)])
                else:
                    tri2D = Delaunay(np.c_[np.ravel(X), np.ravel(L)])

                    # Remove triangles beyond surface edges
                tri2D.points[:, 1] = np.ravel(Z)
                indx = np.ones(tri2D.simplices.shape[0], dtype=bool)
                for ii in range(3):
                    length = np.linalg.norm(
                        tri2D.points[tri2D.simplices[:, ii], :]
                        - tri2D.points[tri2D.simplices[:, ii - 1], :],
                        axis=1,
                    )
                    indx *= length < self.max_distance.value

                # Remove the simplices too long
                tri2D.simplices = tri2D.simplices[indx, :]
                tri2D.vertices = tri2D.vertices[indx, :]

                temp = np.arange(int(nZ * n_sounding)).reshape(
                    (nZ, n_sounding), order="F"
                )
                model_vertices.append(np.c_[np.ravel(X), np.ravel(Y), np.ravel(Z)])
                model_cells.append(tri2D.simplices + model_count)

                model_count += tri2D.points.shape[0]

        else:

            if elevations:  # Assumes non-property_group selection
                z_values = elevations[0].values
                ind = (z_values > 1e-38) & (z_values < 2e-38) == False
                locations = np.c_[locations[ind, :2], z_values[ind]]
            else:
                ind = np.ones(locations.shape[0], dtype="bool")

            if self.z_option.value == "depth":
                z_topo = topo(locations[:, 0], locations[:, 1])

                nan_z = np.isnan(z_topo)
                if np.any(nan_z):
                    _, ii = tree_topo.query(locations[nan_z, :2])
                    z_topo[nan_z] = topo_z[ii]

                locations[:, 2] = z_topo - locations[:, 2]

            tri2D = Delaunay(locations[:, :2])

            indx = np.ones(tri2D.simplices.shape[0], dtype=bool)
            for ii in range(3):
                length = np.linalg.norm(
                    tri2D.points[tri2D.simplices[:, ii], :]
                    - tri2D.points[tri2D.simplices[:, ii - 1], :],
                    axis=1,
                )
                indx *= length < self.max_distance.value

            # Remove the simplices too long
            tri2D.simplices = tri2D.simplices[indx, :]

            model_vertices = np.c_[tri2D.points, locations[:, 2]]
            model_cells = tri2D.simplices
            model = []
            for data_obj in data_list:
                model += [data_obj.values[ind]]

            if len(model) > 0:
                model = np.vstack(model).T

        if len(model_cells) > 0:
            self.surface = Surface.create(
                self.workspace,
                name=self.export_as.value,
                vertices=np.vstack(model_vertices),
                cells=np.vstack(model_cells),
                parent=self.ga_group,
            )
        else:
            print(
                "No triangulation found to export. Increase the max triangulation distance?"
            )
            return

        if self.type.value == "Sections":
            self.surface.add_data(
                {"Line": {"values": np.hstack(line_ids)},}
            )

        self.models = []
        if len(model) > 0:
            self.models = np.vstack(model).T
            for ind, field in enumerate(self.data.value):

                self.surface.add_data(
                    {field: {"values": self.models[ind, :]},}
                )

        if self.live_link.value:
            self.live_link_output(self.ga_group)

        self.workspace.finalize()

    def type_change(self, _):
        if self.type.value == "Horizon":
            self.lines.data.disabled = True
            self.elevations.add_groups = False
            self.add_groups = True
        else:
            self.lines.data.disabled = False
            self.add_groups = "only"
            self.elevations.add_groups = "only"

        self.update_data_list(None)
        self.elevations.update_data_list(None)

    def data_change(self, _):

        if self.data.value:
            self.export_as.value = self.data.value[0] + "_surface"

    def z_options_change(self):
        if self.z_option.value == "depth":
            self.elevations.data.description = "Depth:"
            self.depth_panel.children = [
                self.z_option,
                VBox(
                    [self.elevations.data, Label("Topography"), self.topography.main,]
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
            ipywidgets.RadioButtons()
        """
        return self._z_option

    @property
    def max_distance(self):
        """
            ipywidgets.FloatText()
        """
        return self._max_distance

    @property
    def export_as(self):
        """
            ipywidgets.Text()
        """
        return self._export_as

    @property
    def convert(self):
        """
            ipywidgets.ToggleButton()
        """
        return self._convert

    @property
    def topography(self):
        """
            TopographyOptions()
        """
        return self._topography

    @property
    def type(self):
        """
             ipywidgets.RadioButton()
        """
        return self._type

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

        self.lines.objects = self.objects
        self.elevations.objects = self.objects

        self.lines.workspace = workspace
        self.elevations.workspace = workspace
        self.topography.workspace = workspace


class ContourValues(PlotSelection2D):
    """
    Application for 2D contouring of spatial data.
    """

    defaults = {
        "h5file": "../../assets/FlinFlon.geoh5",
        "objects": "Gravity_Magnetics_drape60m",
        "data": "Airborne_TMI",
        "contours": "-400:2000:100,-240",
        "resolution": 50,
        "ga_group_name": "Contours",
    }

    def __init__(self, **kwargs):

        kwargs = self.apply_defaults(**kwargs)
        self._contours = Text(
            value="", description="Contours", disabled=False, continuous_update=False
        )
        self._export_as = Text(value="Contours", indent=False)

        self._z_value = Checkbox(
            value=False, indent=False, description="Assign Z from values"
        )

        super().__init__(**kwargs)

        out = interactive_output(self.compute_plot, {"contour_values": self.contours,})

        # self.export_as.observe(save_selection, names="value")
        self.data_panel = VBox([self.objects, self.data])
        self._main = VBox(
            [
                self.project_panel,
                HBox(
                    [
                        VBox(
                            [
                                Label("Input options:"),
                                self.data_panel,
                                self.contours,
                                self.window_selection,
                            ]
                        ),
                        VBox(
                            [
                                Label("Save as:"),
                                self.export_as,
                                self.z_value,
                                self.output_panel,
                            ],
                            layout=Layout(width="50%"),
                        ),
                    ]
                ),
                out,
            ]
        )

        def save_selection(_):
            self.save_selection()

        self.trigger.on_click(save_selection)
        self.trigger.description = "Export to GA"
        self.trigger.button_style = "danger"

        def update_name(_):
            self.update_name()

        self.data.observe(update_name, names="value")
        self.update_name()
        #
        # for key in self.__dict__:
        #     if isinstance(getattr(self, key, None), Widget):
        #         getattr(self, key, None).observe(save_selection, names="value")

    @property
    def contours(self):
        """
        :obj:`ipywidgets.Text`: String defining sets of contours.
        Contours can be defined over an interval `50:200:10` and/or at a fix value `215`.
        Any combination of the above can be used:
        50:200:10, 215 => Contours between values 50 and 200 every 10, with a contour at 215.
        """
        return self._contours

    @property
    def export(self):
        """
        :obj:`ipywidgets.ToggleButton`: Write contours to the target geoh5
        """
        return self._export

    @property
    def export_as(self):
        """
        :obj:`ipywidgets.Text`: Name given to the Curve object
        """
        return self._export_as

    @property
    def z_value(self):
        """
        :obj:`ipywidgets.Checkbox`: Assign z-coordinate based on contour values
        """
        return self._z_value

    def compute_plot(self, contour_values):
        """
        Get current selection and trigger update
        """
        entity, data = self.get_selected_entities()
        if data is None:
            return
        if contour_values is not None:
            self.contours.value = contour_values
        # self.save_selection()

    def update_contours(self):
        """
        Assign
        """
        if self.data.value is not None:
            self.export_as.value = self.data.value + "_" + self.contours.value

    def update_name(self):
        if self.data.value is not None:
            self.export_as.value = self.data.value
        else:
            self.export_as.value = "Contours"

    def save_selection(self):
        entity, _ = self.get_selected_entities()

        if getattr(self.contours, "contour_set", None) is not None:
            contour_set = self.contours.contour_set

            vertices, cells, values = [], [], []
            count = 0
            for segs, level in zip(contour_set.allsegs, contour_set.levels):
                for poly in segs:
                    n_v = len(poly)
                    vertices.append(poly)
                    cells.append(
                        np.c_[
                            np.arange(count, count + n_v - 1),
                            np.arange(count + 1, count + n_v),
                        ]
                    )
                    values.append(np.ones(n_v) * level)
                    count += n_v
            if vertices:
                vertices = np.vstack(vertices)
                if self.z_value.value:
                    vertices = np.c_[vertices, np.hstack(values)]
                else:
                    if isinstance(entity, (Points, Curve, Surface)):
                        z_interp = LinearNDInterpolator(
                            entity.vertices[:, :2], entity.vertices[:, 2]
                        )
                        vertices = np.c_[vertices, z_interp(vertices)]
                    else:
                        vertices = np.c_[
                            vertices, np.ones(vertices.shape[0]) * entity.origin["z"],
                        ]

                curves = [
                    child
                    for child in self.ga_group.children
                    if child.name == self.export_as.value
                ]
                if any(curves):
                    curve = curves[0]
                    curve._children = []
                    curve.vertices = vertices
                    curve.cells = np.vstack(cells).astype("uint32")

                    # Remove directly on geoh5
                    project_handle = H5Writer.fetch_h5_handle(self.h5file, entity)
                    base = list(project_handle.keys())[0]
                    obj_handle = project_handle[base]["Objects"]
                    for key in obj_handle[H5Writer.uuid_str(curve.uid)]["Data"].keys():
                        del project_handle[base]["Data"][key]
                    del obj_handle[H5Writer.uuid_str(curve.uid)]

                else:
                    curve = Curve.create(
                        self.workspace,
                        name=self.export_as.value,
                        vertices=vertices,
                        cells=np.vstack(cells).astype("uint32"),
                        parent=self.ga_group,
                    )

                curve.add_data({self.contours.value: {"values": np.hstack(values)}})

                if self.live_link.value:
                    self.live_link_output(self.ga_group)
                self.workspace.finalize()
