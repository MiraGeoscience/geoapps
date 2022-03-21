#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import numpy as np
from discretize.utils import mesh_utils
from geoh5py.objects import BlockModel, ObjectBase
from geoh5py.workspace import Workspace
from ipywidgets import Dropdown, FloatText, HBox, Label, RadioButtons, Text, VBox
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree

from geoapps.base.selection import ObjectDataSelection, TopographyOptions
from geoapps.utils.utils import get_locations, weighted_average


class DataInterpolation(ObjectDataSelection):
    """
    Transfer data from one object to another, or onto a 3D BlockModel
    """

    defaults = {
        "h5file": "../../assets/FlinFlon.geoh5",
        "objects": "{2e814779-c35f-4da0-ad6a-39a6912361f9}",
        "data": ["{f3e36334-be0a-4210-b13e-06933279de25}"],
        "core_cell_size": "50, 50, 50",
        "depth_core": 500,
        "expansion_fact": 1.05,
        "max_distance": 2e3,
        "max_depth": 1e3,
        "method": "Inverse Distance",
        "new_grid": "InterpGrid",
        "no_data_value": 1e-8,
        "out_mode": "To Object",
        "out_object": "{7450be38-1327-4336-a9e4-5cff587b6715}",
        "padding_distance": "0, 0, 0, 0, 0, 0",
        "skew_angle": 0,
        "skew_factor": 1.0,
        "space": "Log",
        "topography": {
            "options": "Object",
            "objects": "{ab3c2083-6ea8-4d31-9230-7aad3ec09525}",
            "data": "Z",
        },
    }

    _select_multiple = True
    _topography = None

    def __init__(self, use_defaults=True, **kwargs):

        if use_defaults:
            self.defaults.update(**kwargs)

        self._core_cell_size = Text(
            description="Smallest cells",
        )
        self._depth_core = FloatText(
            description="Core depth (m)",
        )
        self._expansion_fact = FloatText(
            description="Expansion factor",
        )
        self._max_distance = FloatText(
            description="Maximum distance (m)",
        )
        self._max_depth = FloatText(
            description="Maximum depth (m)",
        )
        self._method = RadioButtons(
            options=["Nearest", "Inverse Distance"],
        )
        self._new_grid = Text(
            description="Name",
        )
        self._no_data_value = FloatText()
        self._out_mode = RadioButtons(
            options=["To Object", "Create 3D Grid"],
        )
        self._out_object = Dropdown()
        self._padding_distance = Text(
            description="Pad Distance (W, E, S, N, D, U)",
        )
        self._skew_angle = FloatText(
            description="Azimuth (d.dd)",
        )
        self._skew_factor = FloatText(
            description="Factor (>0)",
        )
        self._space = RadioButtons(options=["Linear", "Log"])
        self._xy_extent = Dropdown(
            description="Object hull",
        )
        self._xy_reference = Dropdown(
            description="Lateral Extent", style={"description_width": "initial"}
        )
        self.objects.observe(self.object_pick, names="value")
        self.method_skew = VBox(
            [Label("Skew parameters"), self.skew_angle, self.skew_factor]
        )
        self.method_panel = VBox([self.method])
        self.destination_panel = VBox([self.out_mode, self.out_object])
        self.new_grid_panel = VBox(
            [
                self.new_grid,
                self.xy_reference,
                self.core_cell_size,
                self.depth_core,
                self.padding_distance,
                self.expansion_fact,
            ]
        )
        self.method.observe(self.method_update)
        self.out_mode.observe(self.out_update)
        self.parameters = {
            "Method": self.method_panel,
            "Scaling": self.space,
            "Horizontal Extent": VBox(
                [
                    self.max_distance,
                    self.xy_extent,
                ]
            ),
            "Vertical Extent": VBox([]),
            "No-data-value": self.no_data_value,
        }

        self.parameter_choices = Dropdown(
            description="Interpolation Parameters",
            options=list(self.parameters.keys()),
            style={"description_width": "initial"},
        )

        self.parameter_panel = HBox([self.parameter_choices, self.method_panel])
        self.ga_group_name.description = "Output Label:"
        self.ga_group_name.value = "_Interp"
        self.parameter_choices.observe(self.parameter_change)

        super().__init__(**self.defaults)

        self.parameters["Vertical Extent"].children = [
            self.topography.main,
            self.max_depth,
        ]
        self.trigger.on_click(self.trigger_click)
        self.trigger.description = "Interpolate"

    @property
    def core_cell_size(self):
        """
        :obj:`ipywidgets.Text()`
        """
        return self._core_cell_size

    @property
    def depth_core(self):
        """
        :obj:`ipywidgets.FloatText()`
        """
        return self._depth_core

    @property
    def expansion_fact(self):
        """
        :obj:`ipywidgets.FloatText()`
        """
        return self._expansion_fact

    @property
    def main(self):
        if self._main is None:
            self._main = VBox(
                [
                    self.project_panel,
                    HBox(
                        [
                            VBox([Label("Source"), self.data_panel]),
                            VBox([Label("Destination"), self.destination_panel]),
                        ]
                    ),
                    self.parameter_panel,
                    self.output_panel,
                ]
            )
        return self._main

    @property
    def max_distance(self):
        """
        :obj:`ipywidgets.FloatText()`
        """
        return self._max_distance

    @property
    def max_depth(self):
        """
        :obj:`ipywidgets.FloatText()`
        """
        return self._max_depth

    @property
    def method(self):
        """
        :obj:`ipywidgets.RadioButtons()`
        """
        return self._method

    @property
    def new_grid(self):
        """
        :obj:`ipywidgets.Text()`
        """
        return self._new_grid

    @property
    def no_data_value(self):
        """
        :obj:`ipywidgets.FloatText()`
        """
        return self._no_data_value

    @property
    def out_mode(self):
        """
        :obj:`ipywidgets.RadioButtons()`
        """
        return self._out_mode

    @property
    def out_object(self):
        """
        :obj:`ipywidgets.Dropdown()`
        """
        return self._out_object

    @property
    def padding_distance(self):
        """
        :obj:`ipywidgets.Text()`
        """
        return self._padding_distance

    @property
    def skew_angle(self):
        """
        :obj:`ipywidgets.FloatText()`
        """
        return self._skew_angle

    @property
    def skew_factor(self):
        """
        :obj:`ipywidgets.FloatText()`
        """
        return self._skew_factor

    @property
    def space(self):
        """
        :obj:`ipywidgets.RadioButtons()`
        """
        return self._space

    @property
    def topography(self):
        """
        :obj:`geoapps.TopographyOptions()`
        """
        if getattr(self, "_topography", None) is None:
            self._topography = TopographyOptions(
                option_list=["None", "Object", "Constant"],
                workspace=self.workspace,
                **self.defaults["topography"],
            )

        return self._topography

    @property
    def xy_extent(self):
        """
        :obj:`ipywidgets.Dropdown()`
        """
        return self._xy_extent

    @property
    def xy_reference(self):
        """
        :obj:`ipywidgets.Dropdown()`
        """
        return self._xy_reference

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

        self.update_objects_choices()

        if getattr(self, "_topography", None) is not None:
            self.topography.workspace = workspace

    def parameter_change(self, _):
        self.parameter_panel.children = [
            self.parameter_choices,
            self.parameters[self.parameter_choices.value],
        ]

    def update_objects_choices(self):
        # Refresh the list of objects for all
        self.update_objects_list()

        value = self.out_object.value
        self.out_object.options = self.objects.options
        if value in list(dict(self.out_object.options).values()):
            self.out_object.value = value

        value = self.xy_reference.value
        self.xy_reference.options = self.objects.options
        if value in list(dict(self.xy_reference.options).values()):
            self.xy_reference.value = value

        value = self.xy_extent.value
        self.xy_extent.options = self.objects.options
        if value in list(dict(self.xy_extent.options).values()):
            self.xy_extent.value = value

    def method_update(self, _):
        if self.method.value == "Inverse Distance":
            self.method_panel.children = [self.method, self.method_skew]
        elif self.method.value == "Linear":
            self.method_panel.children = [
                self.method,
                Label("Warning! Very slow on 3D objects"),
            ]
        else:
            self.method_panel.children = [self.method]

    def out_update(self, _):
        if self.out_mode.value == "To Object":
            self.destination_panel.children = [self.out_mode, self.out_object]
        else:
            self.destination_panel.children = [self.out_mode, self.new_grid_panel]

    def object_base(self, object):
        for entity in self._workspace.get_entity(object):
            if isinstance(entity, ObjectBase):
                return entity
        return None

    @staticmethod
    def truncate_locs_depths(locs, depth_core):
        zmax = locs[:, 2].max()  # top of locs
        below_core_ind = (zmax - locs[:, 2]) > depth_core
        core_bottom_elev = zmax - depth_core
        locs[
            below_core_ind, 2
        ] = core_bottom_elev  # sets locations below core to core bottom
        return locs

    @staticmethod
    def minimum_depth_core(locs, depth_core, core_z_cell_size):
        zrange = locs[:, 2].max() - locs[:, 2].min()  # locs z range
        if depth_core >= zrange:
            return depth_core - zrange + core_z_cell_size
        else:
            return depth_core

    @staticmethod
    def find_top_padding(obj, core_z_cell_size):
        pad_sum = 0
        for h in np.abs(np.diff(obj.z_cell_delimiters)):
            if h != core_z_cell_size:
                pad_sum += h
            else:
                return pad_sum

    @staticmethod
    def get_block_model(workspace, name, locs, h, depth_core, pads, expansion_factor):

        locs = DataInterpolation.truncate_locs_depths(locs, depth_core)
        depth_core = DataInterpolation.minimum_depth_core(locs, depth_core, h[2])
        mesh = mesh_utils.mesh_builder_xyz(
            locs,
            h,
            padding_distance=[
                [pads[0], pads[1]],
                [pads[2], pads[3]],
                [pads[4], pads[5]],
            ],
            depth_core=depth_core,
            expansion_factor=expansion_factor,
        )

        object_out = BlockModel.create(
            workspace,
            origin=[mesh.x0[0], mesh.x0[1], locs[:, 2].max()],
            u_cell_delimiters=mesh.vectorNx - mesh.x0[0],
            v_cell_delimiters=mesh.vectorNy - mesh.x0[1],
            z_cell_delimiters=-(mesh.x0[2] + mesh.hz.sum() - mesh.vectorNz[::-1]),
            name=name,
        )

        top_padding = DataInterpolation.find_top_padding(object_out, h[2])
        object_out.origin["z"] += top_padding

        return object_out

    def trigger_click(self, _):

        object_from = self.object_base(self.objects.value)
        xyz = get_locations(self.workspace, object_from)
        if xyz is None:
            return

        if len(self.data.value) == 0:
            print("No data selected")
            return

        # Create a tree for the input mesh
        tree = cKDTree(xyz)

        if self.out_mode.value == "To Object":

            self.object_out = self.object_base(self.out_object.value)
            xyz_out = get_locations(self._workspace, self.object_out)

        else:

            xyz_ref = get_locations(self._workspace, self.xy_reference.value)
            if xyz_ref is None:
                print(
                    "No object selected for 'Lateral Extent'. Defaults to input object."
                )
                xyz_ref = xyz.copy()

            # Find extent of grid
            h = np.asarray(self.core_cell_size.value.split(",")).astype(float).tolist()

            pads = (
                np.asarray(self.padding_distance.value.split(","))
                .astype(float)
                .tolist()
            )

            self.object_out = DataInterpolation.get_block_model(
                self._workspace,
                self.new_grid.value,
                xyz_ref,
                h,
                self.depth_core.value,
                pads,
                self.expansion_fact.value,
            )

            # Try to recenter on nearest
            # Find nearest cells
            rad, ind = tree.query(self.object_out.centroids)
            ind_nn = np.argmin(rad)

            d_xyz = self.object_out.centroids[ind_nn, :] - xyz[ind[ind_nn], :]

            self.object_out.origin = np.r_[self.object_out.origin.tolist()] - d_xyz

            xyz_out = self.object_out.centroids.copy()

        xyz_out_orig = xyz_out.copy()

        values, sign, dtype = {}, {}, {}
        for field in self.data.value:

            if isinstance(field, str) and field in "XYZ":
                values[field] = xyz[:, "XYZ".index(field)]
                dtype[field] = values[field].dtype
            else:
                model_in = self.workspace.get_entity(field)[0]
                values[field] = np.asarray(model_in.values, dtype=float).copy()
                dtype[field] = model_in.values.dtype

            values[field][values[field] == self.no_data_value.value] = np.nan
            if self.space.value == "Log":
                sign[field] = np.sign(values[field])
                values[field] = np.log(np.abs(values[field]))
            else:
                sign[field] = np.ones_like(values[field])

        values_interp = {}
        rad, ind = tree.query(xyz_out)
        if self.method.value == "Linear":

            for key, value in values.items():
                F = LinearNDInterpolator(xyz, value)
                values_interp[key] = F(xyz_out)

        elif self.method.value == "Inverse Distance":

            angle = np.deg2rad((450.0 - np.asarray(self.skew_angle.value)) % 360.0)
            rotation = np.r_[
                np.c_[np.cos(angle), np.sin(angle)],
                np.c_[-np.sin(angle), np.cos(angle)],
            ]
            center = np.mean(xyz, axis=0).reshape((3, 1))
            xyz -= np.kron(center, np.ones(xyz.shape[0])).T
            xyz[:, :2] = np.dot(rotation, xyz[:, :2].T).T
            xyz[:, 1] *= self.skew_factor.value
            xyz_out -= np.kron(center, np.ones(xyz_out.shape[0])).T
            xyz_out[:, :2] = np.dot(rotation, xyz_out[:, :2].T).T
            xyz_out[:, 1] *= self.skew_factor.value
            vals, ind_inv = weighted_average(
                xyz,
                xyz_out,
                list(values.values()),
                threshold=1e-1,
                n=8,
                return_indices=True,
            )

            for key, val in zip(list(values.keys()), vals):
                values_interp[key] = val
                sign[key] = sign[key][ind_inv[:, 0]]

        else:
            # Find nearest cells
            for key, value in values.items():

                values_interp[key] = value[ind]
                sign[key] = sign[key][ind]

        for key in values_interp.keys():
            if self.space.value == "Log":
                values_interp[key] = sign[key] * np.exp(values_interp[key])

            values_interp[key][np.isnan(values_interp[key])] = self.no_data_value.value
            values_interp[key][rad > self.max_distance.value] = self.no_data_value.value

        # if hasattr(self.object_out, "centroids"):
        #     xyz_out = self.object_out.centroids
        # elif hasattr(self.object_out, "vertices"):
        #     xyz_out = self.object_out.vertices

        top = np.zeros(xyz_out.shape[0], dtype="bool")
        bottom = np.zeros(xyz_out.shape[0], dtype="bool")
        if self.topography.options.value == "Object" and self.workspace.get_entity(
            self.topography.objects.value
        ):

            for entity in self._workspace.get_entity(self.topography.objects.value):
                if isinstance(entity, ObjectBase):
                    topo_obj = entity

            if getattr(topo_obj, "vertices", None) is not None:
                topo = topo_obj.vertices
            else:
                topo = topo_obj.centroids

            if self.topography.data.uid_name_map[self.topography.data.value] != "Z":
                topo[:, 2] = self.workspace.get_entity(self.topography.data.value)[
                    0
                ].values

            lin_interp = LinearNDInterpolator(topo[:, :2], topo[:, 2])
            z_interp = lin_interp(xyz_out_orig[:, :2])

            ind_nan = np.isnan(z_interp)
            if any(ind_nan):
                tree = cKDTree(topo[:, :2])
                _, ind = tree.query(xyz_out_orig[ind_nan, :2])
                z_interp[ind_nan] = topo[ind, 2]

            top = xyz_out_orig[:, 2] > z_interp
            if self.max_depth.value is not None:
                bottom = np.abs(xyz_out_orig[:, 2] - z_interp) > self.max_depth.value

        elif (
            self.topography.options.value == "Constant"
            and self.topography.constant.value is not None
        ):
            top = xyz_out_orig[:, 2] > self.topography.constant.value
            if self.max_depth.value is not None:
                bottom = (
                    np.abs(xyz_out_orig[:, 2] - self.topography.constant.value)
                    > self.max_depth.value
                )

        for key in values_interp.keys():
            values_interp[key][top] = self.no_data_value.value
            values_interp[key][bottom] = self.no_data_value.value

        if self.xy_extent.value is not None and self.workspace.get_entity(
            self.xy_extent.value
        ):

            for entity in self._workspace.get_entity(self.xy_extent.value):
                if isinstance(entity, ObjectBase):
                    xy_ref = entity
            if hasattr(xy_ref, "centroids"):
                xy_ref = xy_ref.centroids
            elif hasattr(xy_ref, "vertices"):
                xy_ref = xy_ref.vertices

            tree = cKDTree(xy_ref[:, :2])
            rad, _ = tree.query(xyz_out_orig[:, :2])
            for key in values_interp.keys():
                values_interp[key][
                    rad > self.max_distance.value
                ] = self.no_data_value.value

        for key in values_interp.keys():
            if dtype[field] == np.dtype("int32"):
                primitive = "integer"
                vals = np.round(values_interp[key]).astype(dtype[field])
            else:
                primitive = "float"
                vals = values_interp[key].astype(dtype[field])

            self.object_out.add_data(
                {
                    self.data.uid_name_map[key]
                    + self.ga_group_name.value: {"values": vals, "type": primitive}
                }
            )

        if self.live_link.value:
            self.live_link_output(self.export_directory.selected_path, self.object_out)

        self.workspace.finalize()
        self.update_objects_choices()

    def object_pick(self, _):
        if self.objects.value in list(dict(self.xy_reference.options).values()):
            self.xy_reference.value = self.objects.value
