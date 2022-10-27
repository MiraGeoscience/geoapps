#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import re
from time import time
from uuid import UUID

import numpy as np
from geoh5py.groups import ContainerGroup
from geoh5py.objects import Curve, Surface
from geoh5py.ui_json.utils import monitored_directory_copy
from geoh5py.workspace import Workspace
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay, cKDTree

from geoapps.base.application import BaseApplication
from geoapps.base.selection import ObjectDataSelection, TopographyOptions
from geoapps.utils import warn_module_not_found
from geoapps.utils.formatters import string_name

with warn_module_not_found():
    from ipywidgets import (
        FloatText,
        HBox,
        Label,
        RadioButtons,
        Text,
        ToggleButton,
        VBox,
    )


class Surface2D(ObjectDataSelection):
    """
    Application for the conversion of conductivity/depth curves to
    a pseudo 3D conductivity model on surface.
    """

    _add_groups = "only"
    _select_multiple = True
    _object_types = (Curve,)

    def __init__(self, **kwargs):
        self._defaults = app_initializer.copy()
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
        self.models = []

    def trigger_click(self, _):
        obj, data_list = self.get_selected_entities()

        if obj is None:
            return

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
                    ori = "NS"
                else:
                    order = np.argsort(xyz[:, 0])
                    ori = "EW"

                x_locations, y_locations, z_locations, model, line_location = (
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                # Stack the z-coordinates and model
                n_locations = 0

                for ind, elev in enumerate(elevations):
                    # data = self.workspace.get_entity(z_prop)[0]
                    n_locations += 1
                    z_vals = elev.values[line_ind]

                    m_vals = []
                    for m in self.data.value:
                        prop = obj.find_or_create_property_group(
                            name=string_name(self.data.uid_name_map[m])
                        ).properties[ind]

                        values = self.workspace.get_entity(prop)[0].values
                        if values is None:
                            values = np.ones(locations.shape[0]) * np.nan

                        m_vals.append(values[line_ind])

                    m_vals = np.vstack(m_vals).T
                    keep = ~np.isnan(z_vals) * ~np.any(np.isnan(m_vals), axis=1)
                    keep[np.isnan(z_vals)] = False
                    keep[np.any(np.isnan(m_vals), axis=1)] = False
                    x_locations.append(xyz[:, 0][order][keep])
                    y_locations.append(xyz[:, 1][order][keep])

                    if self.z_option.value == "depth":
                        z_topo = topo(xyz[:, 0][order][keep], xyz[:, 1][order][keep])

                        nan_z = np.isnan(z_topo)
                        if np.any(nan_z):
                            _, indices = tree_topo.query(xyz[:, :2][order][keep][nan_z])
                            z_topo[nan_z] = topo_z[indices]

                        z_locations.append(z_topo + z_vals[order][keep])

                    else:
                        z_locations.append(z_vals[order][keep])

                    line_location.append(
                        np.ones_like(z_vals[order][keep])
                        * -int(re.findall(r"\d+", elev.name)[-1])
                    )
                    model.append(m_vals[order, :][keep, :])

                x_locations = np.hstack(x_locations)
                y_locations = np.hstack(y_locations)
                z_locations = np.hstack(z_locations)
                line_location = np.hstack(line_location)

                self.models.append(np.vstack(model))
                line_ids.append(np.ones_like(z_locations.ravel()) * line)

                if ori == "NS":
                    delaunay_2d = Delaunay(
                        np.c_[np.ravel(y_locations), np.ravel(line_location)]
                    )
                else:
                    delaunay_2d = Delaunay(
                        np.c_[np.ravel(x_locations), np.ravel(line_location)]
                    )

                    # Remove triangles beyond surface edges
                delaunay_2d.points[  # pylint: disable=unsupported-assignment-operation
                    :, 1
                ] = np.ravel(z_locations)
                indx = np.ones(delaunay_2d.simplices.shape[0], dtype=bool)
                for i in range(3):
                    length = np.linalg.norm(
                        delaunay_2d.points[  # pylint: disable=unsubscriptable-object
                            delaunay_2d.simplices[:, i], :
                        ]
                        - delaunay_2d.points[  # pylint: disable=unsubscriptable-object
                            delaunay_2d.simplices[:, i - 1], :
                        ],
                        axis=1,
                    )
                    indx *= length < self.max_distance.value

                # Remove the simplices too long
                delaunay_2d.simplices = delaunay_2d.simplices[indx, :]
                model_vertices.append(
                    np.c_[
                        np.ravel(x_locations),
                        np.ravel(y_locations),
                        np.ravel(z_locations),
                    ]
                )
                model_cells.append(delaunay_2d.simplices + model_count)
                model_count += delaunay_2d.points.shape[0]  # pylint: disable=no-member

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
                    _, i = tree_topo.query(locations[nan_z, :2])
                    z_topo[nan_z] = topo_z[i]

                locations[:, 2] = z_topo - locations[:, 2]

            delaunay_2d = Delaunay(locations[:, :2])

            indx = np.ones(delaunay_2d.simplices.shape[0], dtype=bool)
            for i in range(3):
                length = np.linalg.norm(
                    delaunay_2d.points[  # pylint: disable=unsubscriptable-object
                        delaunay_2d.simplices[:, i], :
                    ]
                    - delaunay_2d.points[  # pylint: disable=unsubscriptable-object
                        delaunay_2d.simplices[:, i - 1], :
                    ],
                    axis=1,
                )
                indx *= length < self.max_distance.value

            # Remove the simplices too long
            delaunay_2d.simplices = delaunay_2d.simplices[indx, :]

            model_vertices = np.c_[delaunay_2d.points, locations[:, 2]]
            model_cells = delaunay_2d.simplices
            self.models = []
            for data_obj in data_list:
                self.models += [data_obj.values[ind]]

        temp_geoh5 = f"{string_name(self.export_as.value)}_{time():.0f}.geoh5"

        ws, self.live_link.value = BaseApplication.get_output_workspace(
            self.live_link.value, self.export_directory.selected_path, temp_geoh5
        )
        with ws as workspace:
            out_entity = ContainerGroup.create(workspace, name=self.ga_group_name.value)

            if len(model_cells) > 0:
                surface = Surface.create(
                    workspace,
                    name=string_name(self.export_as.value),
                    vertices=np.vstack(model_vertices),
                    cells=np.vstack(model_cells),
                    parent=out_entity,
                )
            else:
                print(
                    "No triangulation found to export. Increase the max triangulation distance?"
                )
                return

            if self.type.value == "Sections":
                surface.add_data(
                    {
                        "Line": {"values": np.hstack(line_ids)},
                    }
                )

                if len(self.models) > 0:
                    for uid, model in zip(self.data.value, self.models):
                        surface.add_data(
                            {
                                self.data.uid_name_map[uid]: {"values": model},
                            }
                        )
            else:
                for data_obj, model in zip(data_list, self.models):
                    surface.add_data(
                        {
                            data_obj.name: {"values": model},
                        }
                    )

        if self.live_link.value:
            monitored_directory_copy(self.export_directory.selected_path, out_entity)

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


app_initializer = {
    "geoh5": "../../assets/FlinFlon.geoh5",
    "objects": UUID("{5fa66412-3a4c-440c-8b87-6f10cb5f1c7f}"),
    "data": [UUID("{f94e8e29-6d1b-4e53-bb4a-6cb77e8f07d8}")],
    "max_distance": 250,
    "elevations": {"data": UUID("{b154b397-9c0d-4dcf-baeb-3909fb9a2a19}")},
    "lines": {"data": UUID("{50f6be8d-226d-4f07-9a1c-531e722df260}")},
    "topography": {},
}
