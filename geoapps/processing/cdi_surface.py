from ipywidgets import FloatText, HBox, Label, RadioButtons, Text, ToggleButton, VBox
import numpy as np
from geoh5py.objects import Surface
from geoh5py.workspace import Workspace
from scipy.interpolate import LinearNDInterpolator, interp1d
from scipy.spatial import cKDTree, Delaunay
from geoapps.inversion import TopographyOptions
from geoapps.selection import ObjectDataSelection


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
        self._z_option = RadioButtons(
            options=["elevation", "depth"], description="Layers reference:",
        )
        self._max_depth = FloatText(description="Max depth (m):",)
        self._max_distance = FloatText(value=50, description="Max distance (m):",)
        self._tolerance = FloatText(value=1, description="Tolerance (m):")
        self._export_as = Text("CDI_", description="Name: ")
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
        self.data.description = "Model fields: "

        def z_options_change(_):
            self.z_options_change()

        self.z_option.observe(z_options_change)
        self.depth_panel = HBox([self.z_option, self.elevations.data])

        def convert_trigger(_):
            self.convert_trigger()

        self.trigger.on_click(convert_trigger)
        self._widget = VBox(
            [
                self.project_panel,
                HBox(
                    [
                        VBox([self.widget, self.lines.data,]),
                        VBox(
                            [
                                Label("Triangulation"),
                                self.max_depth,
                                self.max_distance,
                                self.tolerance,
                            ]
                        ),
                    ]
                ),
                self.depth_panel,
                Label("Output"),
                self.export_as,
                self.trigger_panel,
            ]
        )

    def convert_trigger(self):

        if self.workspace.get_entity(self.objects.value):
            curve = self.workspace.get_entity(self.objects.value)[0]

        lines_id = curve.get_data(self.lines.data.value)[0].values
        lines = np.unique(lines_id).tolist()

        if self.z_option.value == "depth":
            if self.topography.options_button.value == "Object":

                topo_obj = self.workspace.get_entity(self.topography.objects.value)[0]

                if hasattr(topo_obj, "centroids"):
                    vertices = topo_obj.centroids.copy()
                else:
                    vertices = topo_obj.vertices.copy()

                topo_xy = vertices[:, :2]

                if self.topography.value.value == "Vertices":
                    topo_z = vertices[:, 2]
                else:
                    topo_z = topo_obj.get_data(self.topography.value.value)[0].values

            else:
                topo_xy = curve.vertices[:, :2].copy()

                if self.topography.options_button.value == "Constant":
                    topo_z = (
                        np.ones_like(curve.vertices[:, 2])
                        * self.topography.constant.value
                    )
                else:
                    topo_z = (
                        np.ones_like(curve.vertices[:, 2])
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
                    m_vals.append(self.workspace.get_entity(prop)[0].values[line_ind])

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

                M.append(m_vals[order, :][keep, :])

                if ind == 0:

                    x_loc = xyz[:, 0][order][keep]
                    y_loc = xyz[:, 1][order][keep]
                    z_loc = Z[0]

            X = np.hstack(X)
            Y = np.hstack(Y)
            Z = np.hstack(Z)
            model.append(np.vstack(M))
            line_ids.append(np.ones_like(Z.ravel()) * line)

            if np.std(y_loc) > np.std(x_loc):
                tri2D = Delaunay(np.c_[np.ravel(Y), np.ravel(Z)])
                dist = np.ravel(Y)
                topo_top = interp1d(y_loc, z_loc)
            else:
                tri2D = Delaunay(np.c_[np.ravel(X), np.ravel(Z)])
                dist = np.ravel(X)
                topo_top = interp1d(x_loc, z_loc)

                # Remove triangles beyond surface edges
            indx = np.ones(tri2D.simplices.shape[0], dtype=bool)
            for ii in range(3):
                x = np.mean(
                    np.c_[
                        dist[tri2D.simplices[:, ii]], dist[tri2D.simplices[:, ii - 1]],
                    ],
                    axis=1,
                )
                z = np.mean(
                    np.c_[Z[tri2D.simplices[:, ii]], Z[tri2D.simplices[:, ii - 1]]],
                    axis=1,
                )

                length = np.linalg.norm(
                    tri2D.points[tri2D.simplices[:, ii], :]
                    - tri2D.points[tri2D.simplices[:, ii - 1], :],
                    axis=1,
                )

                indx *= (
                    (z <= (topo_top(x) + self.tolerance.value))
                    * (z >= (topo_top(x) - self.max_depth.value - self.tolerance.value))
                    * (length < self.max_distance.value)
                )

            # Remove the simplices too long
            tri2D.simplices = tri2D.simplices[indx, :]
            tri2D.vertices = tri2D.vertices[indx, :]

            temp = np.arange(int(nZ * n_sounding)).reshape((nZ, n_sounding), order="F")
            model_vertices.append(np.c_[np.ravel(X), np.ravel(Y), np.ravel(Z)])
            model_cells.append(tri2D.simplices + model_count)

            model_count += tri2D.points.shape[0]

        surface = Surface.create(
            self.workspace,
            name=self.export_as.value,
            vertices=np.vstack(model_vertices),
            cells=np.vstack(model_cells),
            parent=self.ga_group,
        )

        surface.add_data(
            {"Line": {"values": np.hstack(line_ids)},}
        )

        models = np.vstack(model)
        for ind, field in enumerate(self.data.value):

            surface.add_data(
                {field: {"values": models[:, ind]},}
            )

    def z_options_change(self):
        if self.z_option.value == "depth":
            self.elevations.data.description = "Depth:"
            self.depth_panel.children = [
                self.z_option,
                VBox(
                    [self.elevations.data, Label("Topography"), self.topography.widget,]
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
    def export_as(self):
        """
            widgets.Text()
        """
        return self._export_as

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
