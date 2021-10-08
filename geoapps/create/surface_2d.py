#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import re

import numpy as np
from geoh5py.objects import Curve, Surface
from geoh5py.workspace import Workspace
from ipywidgets import FloatText, HBox, Label, RadioButtons, Text, ToggleButton, VBox
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay, cKDTree

from geoapps.selection import ObjectDataSelection, TopographyOptions
from geoapps.utils.formatters import string_name


class Surface2D(ObjectDataSelection):
    """
    Application for the conversion of conductivity/depth curves to
    a pseudo 3D conductivity model on surface.
    """

    defaults = {
        "h5file": "../../assets/FlinFlon.geoh5",
        "objects": "{5fa66412-3a4c-440c-8b87-6f10cb5f1c7f}",
        "data": ["{f94e8e29-6d1b-4e53-bb4a-6cb77e8f07d8}"],
        "max_distance": 250,
        "elevations": {"data": "{b154b397-9c0d-4dcf-baeb-3909fb9a2a19}"},
        "lines": {"data": "{0be5374c-3ebb-4762-962a-97d99f3cb0e1}"},
        "topography": {},
    }

    _add_groups = "only"
    _select_multiple = True
    _object_types = (Curve,)

    def __init__(self, **kwargs):
        self.defaults.update(**kwargs)
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
        self._max_distance = FloatText(
            description="Max Triangulation Distance (m):",
        )
        self._export_as = Text("CDI_", description="Surface:")
        self._convert = ToggleButton(description="Convert >>", button_style="success")

        super().__init__(**self.defaults)

        self._lines = ObjectDataSelection(
            add_groups=False,
            workspace=self.workspace,
            objects=self.objects,
            ind_value=["line"],
            **self.defaults["lines"],
        )
        self._topography = TopographyOptions(
            workspace=self.workspace, **self.defaults["topography"]
        )
        self._elevations = ObjectDataSelection(
            add_groups="only",
            workspace=self.workspace,
            objects=self.objects,
            **self.defaults["elevations"],
        )

        self.ga_group_name.value = "CDI"
        self.lines.data.description = "Line field:"
        self.elevations.data.description = "Elevations:"
        self.type.observe(self.type_change, names="value")
        self.data.observe(self.data_change, names="value")
        self.data.description = "Model fields: "
        self.z_option.observe(self.z_options_change, names="value")
        self.depth_panel = HBox([self.z_option, self.elevations.data])
        self.trigger.on_click(self.trigger_click)

    def trigger_click(self, _):

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

                try:
                    topo_z = self.workspace.get_entity(self.topography.data.value)[
                        0
                    ].values
                except IndexError:
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
            lines_id = self.workspace.get_entity(self.lines.data.value)[0].values
            lines = np.unique(lines_id).tolist()
            model_vertices = []
            model_cells = []
            model_count = 0
            self.models = []
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
                        prop = obj.find_or_create_property_group(
                            name=string_name(self.data.uid_name_map[m])
                        ).properties[ind]
                        m_vals.append(
                            self.workspace.get_entity(prop)[0].values[line_ind]
                        )

                    m_vals = np.vstack(m_vals).T
                    keep = (
                        np.isnan(z_vals) * np.any(np.isnan(m_vals), axis=1)
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

                self.models.append(np.vstack(M))
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

            self.models = list(np.vstack(self.models).T)

        else:

            if elevations:  # Assumes non-property_group selection
                z_values = elevations[0].values
                ind = np.isnan(z_values) == False
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
            self.models = []
            for data_obj in data_list:
                self.models += [data_obj.values[ind]]

        if len(model_cells) > 0:
            self.surface = Surface.create(
                self.workspace,
                name=string_name(self.export_as.value),
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
                {
                    "Line": {"values": np.hstack(line_ids)},
                }
            )

            if len(self.models) > 0:
                for uid, model in zip(self.data.value, self.models):
                    self.surface.add_data(
                        {
                            self.data.uid_name_map[uid]: {"values": model},
                        }
                    )
        else:
            for data_obj, model in zip(data_list, self.models):
                self.surface.add_data(
                    {
                        data_obj.name: {"values": model},
                    }
                )

        if self.live_link.value:
            self.live_link_output(self.export_directory.selected_path, self.ga_group)

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
            self.export_as.value = (
                self.data.uid_name_map[self.data.value[0]] + "_surface"
            )

    def z_options_change(self, _):
        if self.z_option.value == "depth":
            self.elevations.data.description = "Depth:"
            self.depth_panel.children = [
                self.z_option,
                VBox(
                    [
                        self.elevations.data,
                        Label("Topography"),
                        self.topography.main,
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
    def main(self):
        if self._main is None:
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
        return self._main

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
        self.base_workspace_changes(workspace)

        # Refresh the list of objects
        self.update_objects_list()

        self.lines.workspace = workspace
        self.elevations.workspace = workspace
        self.topography.workspace = workspace
