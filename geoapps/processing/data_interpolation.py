import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import LinearNDInterpolator
import discretize
from geoh5py.objects import BlockModel
from geoh5py.workspace import Workspace
from ipywidgets import (
    Dropdown,
    Text,
    FloatText,
    VBox,
    HBox,
    ToggleButton,
    Label,
    RadioButtons,
)
from geoapps.selection import ObjectDataSelection
from geoapps.inversion import TopographyOptions


class DataInterpolation(ObjectDataSelection):
    """
    Transfer data from one object to another, or onto a 3D BlockModel
    """

    defaults = {
        "select_multiple": True,
        "h5file": "../../assets/FlinFlon.geoh5",
    }

    def __init__(self, **kwargs):
        kwargs = self.apply_defaults(**kwargs)

        self.out_mode = RadioButtons(
            options=["To Object:", "Create 3D Grid"], value="To Object:", disabled=False
        )
        self.mesh = Dropdown()
        self.xy_reference = Dropdown(description="XY Extent from:",)

        def object_pick(_):
            self.object_pick()

        self.objects.observe(object_pick, names="value")

        self.new_grid = Text(
            value="InterpGrid", description="New grid name:", disabled=False,
        )
        self.core_cell_size = Text(
            value="25, 25, 25", description="Smallest cells", disabled=False,
        )
        self.depth_core = FloatText(
            value=500, description="Core depth (m)", disabled=False,
        )
        self.padding_distance = Text(
            value="0, 0, 0, 0, 0, 0",
            description="Pad Distance (W, E, N, S, D, U)",
            disabled=False,
        )
        self.expansion_fact = FloatText(
            value=1.05, description="Expansion factor", disabled=False,
        )
        self.space = RadioButtons(
            options=["Linear", "Log"], value="Linear", disabled=False
        )
        self.method = RadioButtons(
            options=["Nearest", "Linear", "Inverse Distance"],
            value="Nearest",
            disabled=False,
        )
        self.max_distance = FloatText(value=1e3, description="Maximum distance XY (m)",)
        self.max_depth = FloatText(value=1e3, description="Maximum distance Z (m)",)
        self.no_data_value = FloatText(value=-99999, description="No-Data-Value",)
        self.skew_angle = FloatText(
            value=0, description="Azimuth (d.dd)", disabled=False,
        )
        self.skew_factor = FloatText(value=1, description="Factor", disabled=False,)
        self.method_skew = VBox(
            [Label("Skew interpolation"), self.skew_angle, self.skew_factor]
        )
        self.method_panel = HBox([self.method])
        self.interpolate = ToggleButton(
            value=False, description="Interpolate", icon="check"
        )
        self.xy_extent = Dropdown(description="Trim xy extent with:",)
        super().__init__(**kwargs)

        def method_update(_):
            self.method_update()

        self.method.observe(method_update)

        def out_update(_):
            self.out_update()

        self.out_mode.observe(out_update)

        def interpolate_call(_):
            self.interpolate_call()

        self.interpolate.observe(interpolate_call)

        self.topography = TopographyOptions()
        self.topography.workspace = self._workspace
        self.topography.offset.disabled = True
        self.topography.options.options = ["Object", "Constant", "None"]
        self.topography.options.value = "Object"

        if "topography" in kwargs.keys():
            self.topography.__populate__(**kwargs["topography"])

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

        self.out_panel = HBox([self.out_mode, self.mesh])

        self._widget = VBox(
            [
                self.project_panel,
                HBox(
                    [
                        VBox([Label("Input"), self.widget]),
                        VBox([Label("Output"), self.out_panel]),
                    ]
                ),
                VBox(
                    [
                        Label("Interpolation Parameters"),
                        HBox(
                            [
                                VBox([Label("Space"), self.space]),
                                VBox([Label("Method"), self.method_panel]),
                            ]
                        ),
                        self.no_data_value,
                        self.max_distance,
                        self.max_depth,
                        VBox([Label("Cut with topo"), self.topography.widget]),
                        self.xy_extent,
                    ]
                ),
                self.interpolate,
            ]
        )

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

        self.mesh.options = self.objects.options
        self.xy_reference.options = self.objects.options
        self.xy_extent.options = self.objects.options

        if getattr(self, "topography", None) is not None:
            self.topography.workspace = workspace

    def method_update(self):
        if self.method.value == "Inverse Distance":
            self.method_panel.children = [self.method, self.method_skew]
        else:
            self.method_panel.children = [self.method]

    def out_update(self):
        if self.out_mode.value == "To Object:":
            self.out_panel.children = [self.out_mode, self.mesh]
        else:
            self.out_panel.children = [self.out_mode, self.new_grid_panel]

    def interpolate_call(self):
        if self.interpolate.value:

            object_from = self.workspace.get_entity(self.objects.value)[0]

            if hasattr(object_from, "centroids"):
                xyz = object_from.centroids.copy()
            elif hasattr(object_from, "vertices"):
                xyz = object_from.vertices.copy()

            # Create a tree for the input mesh
            tree = cKDTree(xyz)

            if self.out_mode.value == "To Object:":

                object_to = self.workspace.get_entity(self.mesh.value)[0]

                if hasattr(object_to, "centroids"):
                    xyz_out = object_to.centroids.copy()
                elif hasattr(object_to, "vertices"):
                    xyz_out = object_to.vertices.copy()

            else:

                ref_in = self.workspace.get_entity(self.xy_reference.value)[0]

                if hasattr(ref_in, "centroids"):
                    xyz_ref = ref_in.centroids
                elif hasattr(ref_in, "vertices"):
                    xyz_ref = ref_in.vertices

                # Find extent of grid
                h = (
                    np.asarray(self.core_cell_size.value.split(","))
                    .astype(float)
                    .tolist()
                )

                pads = (
                    np.asarray(self.padding_distance.value.split(","))
                    .astype(float)
                    .tolist()
                )

                # Use discretize to build a tensor mesh
                mesh = discretize.utils.meshutils.mesh_builder_xyz(
                    xyz_ref,
                    h,
                    padding_distance=[
                        [pads[0], pads[1]],
                        [pads[2], pads[3]],
                        [pads[4], pads[5]],
                    ],
                    depth_core=self.depth_core.value,
                    expansion_factor=self.expansion_fact.value,
                )

                object_to = BlockModel.create(
                    self.workspace,
                    origin=[mesh.x0[0], mesh.x0[1], xyz_ref[:, 2].max()],
                    u_cell_delimiters=mesh.vectorNx - mesh.x0[0],
                    v_cell_delimiters=mesh.vectorNy - mesh.x0[1],
                    z_cell_delimiters=-(xyz_ref[:, 2].max() - mesh.vectorNz[::-1]),
                    name=self.new_grid.value,
                )

                # Try to recenter on nearest
                # Find nearest cells
                rad, ind = tree.query(object_to.centroids)
                ind_nn = np.argmin(rad)

                d_xyz = object_to.centroids[ind_nn, :] - xyz[ind[ind_nn], :]

                object_to.origin = np.r_[object_to.origin.tolist()] - d_xyz

                xyz_out = object_to.centroids.copy()

            values = {}
            for field in self.data.value:
                model_in = object_from.get_data(field)[0]
                values[field] = model_in.values.copy()

                values[field][values[field] == self.no_data_value.value] = np.nan
                if self.space.value == "Log":
                    values[field] = np.log(values[field])

            values_interp = {}
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

                tree = cKDTree(xyz)

                xyz_out -= np.kron(center, np.ones(xyz_out.shape[0])).T
                xyz_out[:, :2] = np.dot(rotation, xyz_out[:, :2].T).T
                xyz_out[:, 1] *= self.skew_factor.value

                # Find nearest cells
                rad, ind = tree.query(xyz_out, 8)

                for key, value in values.items():
                    values_interp[key] = np.zeros(xyz_out.shape[0])
                    weight = np.zeros(xyz_out.shape[0])

                    for ii in range(8):
                        values_interp[key] += value[ind[:, ii]] / (rad[:, ii] + 1e-1)
                        weight += 1.0 / (rad[:, ii] + 1e-1)

                    values_interp[key] /= weight

            else:
                # Find nearest cells
                for key, value in values.items():
                    rad, ind = tree.query(xyz_out)
                    values_interp[key] = value[ind]

            for key in values_interp.keys():
                if self.space.value == "Log":
                    values_interp[key] = np.exp(values_interp[key])

                values_interp[key][
                    np.isnan(values_interp[key])
                ] = self.no_data_value.value

                if self.method.value == "Inverse Distance":
                    values_interp[key][
                        rad[:, 0] > self.max_distance.value
                    ] = self.no_data_value.value
                    if self.max_depth.value is not None:
                        values_interp[key][
                            np.abs(xyz_out[:, 2] - xyz[ind[:, 0], 2])
                            > self.max_depth.value
                        ] = self.no_data_value.value

                else:
                    values_interp[key][
                        rad > self.max_distance.value
                    ] = self.no_data_value.value

                    if self.max_depth.value is not None:
                        values_interp[key][
                            np.abs(xyz_out[:, 2] - xyz[ind, 2]) > self.max_depth.value
                        ] = self.no_data_value.value

            if hasattr(object_to, "centroids"):
                xyz_out = object_to.centroids
            elif hasattr(object_to, "vertices"):
                xyz_out = object_to.vertices

            if self.topography.options.value == "Object" and self.workspace.get_entity(
                self.topography.objects.value
            ):
                topo_obj = self.workspace.get_entity(self.topography.objects.value)[0]
                if getattr(topo_obj, "vertices", None) is not None:
                    topo = topo_obj.vertices
                else:
                    topo = topo_obj.centroids

                if self.topography.data.value != "Z":
                    topo[:, 2] = topo_obj.get_data(self.topography.data.value)[0].values

                F = LinearNDInterpolator(topo[:, :2], topo[:, 2])
                z_interp = F(xyz_out[:, :2])

                ind_nan = np.isnan(z_interp)
                if any(ind_nan):
                    tree = cKDTree(topo[:, :2])
                    _, ind = tree.query(xyz_out[ind_nan, :2])
                    z_interp[ind_nan] = topo[ind, 2]

                for key in values_interp.keys():
                    values_interp[key][
                        xyz_out[:, 2] > z_interp
                    ] = self.no_data_value.value
            elif (
                self.topography.options.value == "Constant"
                and self.topography.constant.value is not None
            ):
                for key in values_interp.keys():
                    values_interp[key][
                        xyz_out[:, 2] > self.topography.constant.value
                    ] = self.no_data_value.value

            if self.xy_extent.value is not None and self.workspace.get_entity(
                self.xy_extent.value
            ):
                xy_ref = self.workspace.get_entity(self.xy_extent.value)[0]
                if hasattr(xy_ref, "centroids"):
                    xy_ref = xy_ref.centroids
                elif hasattr(xy_ref, "vertices"):
                    xy_ref = xy_ref.vertices

                tree = cKDTree(xy_ref[:, :2])
                rad, _ = tree.query(xyz_out[:, :2])
                for key in values_interp.keys():
                    values_interp[key][
                        rad > self.max_distance.value
                    ] = self.no_data_value.value

            for key in values_interp.keys():
                object_to.add_data({key + "_interp": {"values": values_interp[key]}})

            self.interpolate.value = False
            self.workspace.finalize()

    def object_pick(self):

        if self.objects.value in self.xy_reference.options:
            self.xy_reference.value = self.objects.value
