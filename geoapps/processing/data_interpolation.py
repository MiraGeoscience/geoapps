import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import LinearNDInterpolator
import discretize
from geoh5py.objects import BlockModel, ObjectBase
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
from geoapps.selection import ObjectDataSelection, TopographyOptions


class DataInterpolation(ObjectDataSelection):
    """
    Transfer data from one object to another, or onto a 3D BlockModel
    """

    defaults = {
        "select_multiple": True,
        "h5file": "../../assets/FlinFlon.geoh5",
        "objects": "Inversion_VTEM_Model",
        "data": ["Iteration_7_model"],
        "core_cell_size": "50, 50, 50",
        "depth_core": 500,
        "expansion_fact": 1.05,
        "max_distance": 1e3,
        "max_depth": 1e3,
        "method": "Inverse Distance",
        "new_grid": "InterpGrid",
        "no_data_value": 1e-8,
        "out_mode": "TO Object",
        "out_object": "O2O_Interp_25m",
        "padding_distance": "0, 0, 0, 0, 0, 0",
        "skew_angle": 0,
        "skew_factor": 1.0,
        "space": "Log",
        "topography": {"objects": "Topography", "data": "Z"},
    }

    def __init__(self, use_defaults=True, **kwargs):

        if use_defaults:
            kwargs = self.apply_defaults(**kwargs)

        self._core_cell_size = Text(description="Smallest cells",)
        self._depth_core = FloatText(description="Core depth (m)",)
        self._expansion_fact = FloatText(description="Expansion factor",)
        self._max_distance = FloatText(description="Maximum distance (m)",)
        self._max_depth = FloatText(description="Maximum depth (m)",)
        self._method = RadioButtons(options=["Nearest", "Linear", "Inverse Distance"],)
        self._new_grid = Text(description="Name",)
        self._no_data_value = FloatText()
        self._out_mode = RadioButtons(options=["To Object", "Create 3D Grid"],)
        self._out_object = Dropdown()
        self._padding_distance = Text(description="Pad Distance (W, E, N, S, D, U)",)
        self._skew_angle = FloatText(description="Azimuth (d.dd)",)
        self._skew_factor = FloatText(description="Factor (>0)",)
        self._space = RadioButtons(options=["Linear", "Log"])
        self._xy_extent = Dropdown(description="Object hull",)
        self._xy_reference = Dropdown(description="Lateral Extent",)

        def object_pick(_):
            self.object_pick()

        self.objects.observe(object_pick, names="value")

        self.method_skew = VBox(
            [Label("Skew parameters"), self.skew_angle, self.skew_factor]
        )
        self.method_panel = VBox([self.method])
        self.out_panel = VBox([self.out_mode, self.out_object])
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

        super().__init__(**kwargs)

        def interpolate_call(_):
            self.interpolate_call()
            self.update_objects_choices()

        self.trigger.on_click(interpolate_call)
        self.trigger.description = "Interpolate"
        self._topography = TopographyOptions()
        self.topography.offset.disabled = True
        self.topography.options.options = ["Object", "Constant", "None"]

        if getattr(self, "_workspace", None) is not None:
            self.topography.workspace = self._workspace
            if "topography" in kwargs.keys():
                self.topography.__populate__(**kwargs["topography"])

        self.parameter_choices = Dropdown(
            description="Interpolation Parameters",
            options=[
                "Method",
                "Scaling",
                "Horizontal Extent",
                "Vertical Extent",
                "No-data-value",
            ],
            style={"description_width": "initial"},
        )
        self.parameters = {
            "Method": self.method_panel,
            "Scaling": self.space,
            "Horizontal Extent": VBox([self.xy_extent, self.max_distance]),
            "Vertical Extent": VBox([self.topography.widget, self.max_depth]),
            "No-data-value": self.no_data_value,
        }

        self.parameter_panel = HBox([self.parameter_choices, self.method_panel])

        self.parameter_choices.observe(self.parameter_change)
        self._widget = VBox(
            [
                self.project_panel,
                HBox(
                    [
                        VBox([Label("Source"), self.widget]),
                        VBox([Label("Destination"), self.out_panel]),
                    ]
                ),
                self.parameter_panel,
                self.trigger,
                self.live_link_panel,
            ]
        )

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
    def widget(self):
        return self._widget

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

        self.update_objects_choices()

        if getattr(self, "topography", None) is not None:
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
        if value in self.out_object.options:
            self.out_object.value = value

        value = self.xy_reference.value
        self.xy_reference.options = self.objects.options
        if value in self.xy_reference.options:
            self.xy_reference.value = value

        value = self.xy_extent.value
        self.xy_extent.options = self.objects.options
        if value in self.xy_extent.options:
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
            self.out_panel.children = [self.out_mode, self.out_object]
        else:
            self.out_panel.children = [self.out_mode, self.new_grid_panel]

    def interpolate_call(self):

        for entity in self._workspace.get_entity(self.objects.value):
            if isinstance(entity, ObjectBase):
                object_from = entity

        if hasattr(object_from, "centroids"):
            xyz = object_from.centroids.copy()
        elif hasattr(object_from, "vertices"):
            xyz = object_from.vertices.copy()
        else:
            return

        # Create a tree for the input mesh
        tree = cKDTree(xyz)

        if self.out_mode.value == "To Object":

            for entity in self._workspace.get_entity(self.out_object.value):
                if isinstance(entity, ObjectBase):
                    object_to = entity

            if hasattr(object_to, "centroids"):
                xyz_out = object_to.centroids.copy()
            elif hasattr(object_to, "vertices"):
                xyz_out = object_to.vertices.copy()

        else:

            for entity in self._workspace.get_entity(self.xy_reference.value):
                if isinstance(entity, ObjectBase):
                    ref_in = entity

            if hasattr(ref_in, "centroids"):
                xyz_ref = ref_in.centroids
            elif hasattr(ref_in, "vertices"):
                xyz_ref = ref_in.vertices

            # Find extent of grid
            h = np.asarray(self.core_cell_size.value.split(",")).astype(float).tolist()

            pads = (
                np.asarray(self.padding_distance.value.split(","))
                .astype(float)
                .tolist()
            )

            # Use discretize to build a tensor mesh
            delta_z = xyz_ref[:, 2].max() - xyz_ref[:, 2]
            xyz_ref = xyz_ref[delta_z < self.depth_core.value, :]
            depth_core = (
                self.depth_core.value
                - (xyz_ref[:, 2].max() - xyz_ref[:, 2].min())
                + h[2]
            )
            mesh = discretize.utils.meshutils.mesh_builder_xyz(
                xyz_ref,
                h,
                padding_distance=[
                    [pads[0], pads[1]],
                    [pads[2], pads[3]],
                    [pads[4], pads[5]],
                ],
                depth_core=depth_core,
                expansion_factor=self.expansion_fact.value,
            )
            object_to = BlockModel.create(
                self.workspace,
                origin=[mesh.x0[0], mesh.x0[1], xyz_ref[:, 2].max()],
                u_cell_delimiters=mesh.vectorNx - mesh.x0[0],
                v_cell_delimiters=mesh.vectorNy - mesh.x0[1],
                z_cell_delimiters=-(mesh.x0[2] + mesh.hz.sum() - mesh.vectorNz[::-1]),
                name=self.new_grid.value,
            )

            # Try to recenter on nearest
            # Find nearest cells
            rad, ind = tree.query(object_to.centroids)
            ind_nn = np.argmin(rad)

            d_xyz = object_to.centroids[ind_nn, :] - xyz[ind[ind_nn], :]

            object_to.origin = np.r_[object_to.origin.tolist()] - d_xyz

            xyz_out = object_to.centroids.copy()

        values, sign, dtype = {}, {}, {}
        for field in self.data.value:
            model_in = object_from.get_data(field)[0]
            values[field] = np.asarray(model_in.values, dtype=float).copy()
            dtype[field] = model_in.values.dtype
            values[field][values[field] == self.no_data_value.value] = np.nan
            if self.space.value == "Log":
                sign[field] = np.sign(values[field])
                values[field] = np.log(np.abs(values[field]))
            else:
                sign[field] = np.ones_like(values[field])

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
                sign[key] = sign[key][ind[:, 0]]

        else:
            # Find nearest cells
            for key, value in values.items():
                rad, ind = tree.query(xyz_out)
                values_interp[key] = value[ind]
                sign[key] = sign[key][ind]

        for key in values_interp.keys():
            if self.space.value == "Log":
                values_interp[key] = sign[key] * np.exp(values_interp[key])

            values_interp[key][np.isnan(values_interp[key])] = self.no_data_value.value

            if self.method.value == "Inverse Distance":
                values_interp[key][
                    rad[:, 0] > self.max_distance.value
                ] = self.no_data_value.value
                if self.max_depth.value is not None:
                    values_interp[key][
                        np.abs(xyz_out[:, 2] - xyz[ind[:, 0], 2]) > self.max_depth.value
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

            for entity in self._workspace.get_entity(self.topography.objects.value):
                if isinstance(entity, ObjectBase):
                    topo_obj = entity

            if getattr(topo_obj, "vertices", None) is not None:
                topo = topo_obj.vertices
            else:
                topo = topo_obj.centroids

            if self.topography.data.value != "Z":
                topo[:, 2] = topo_obj.get_data(self.topography.data.value)[0].values

            lin_interp = LinearNDInterpolator(topo[:, :2], topo[:, 2])
            z_interp = lin_interp(xyz_out[:, :2])

            ind_nan = np.isnan(z_interp)
            if any(ind_nan):
                tree = cKDTree(topo[:, :2])
                _, ind = tree.query(xyz_out[ind_nan, :2])
                z_interp[ind_nan] = topo[ind, 2]

            for key in values_interp.keys():
                values_interp[key][xyz_out[:, 2] > z_interp] = self.no_data_value.value
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

            for entity in self._workspace.get_entity(self.xy_extent.objects.value):
                if isinstance(entity, ObjectBase):
                    xy_ref = entity
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
            if dtype[field] == np.dtype("int32"):
                primitive = "integer"
                vals = np.round(values_interp[key]).astype(dtype[field])
            else:
                primitive = "float"
                vals = values_interp[key].astype(dtype[field])

            object_to.add_data({key + "_interp": {"values": vals, "type": primitive}})

        if self.live_link.value:
            self.live_link_output(object_to)

        self.workspace.finalize()

    def object_pick(self):
        if self.objects.value in self.xy_reference.options:
            self.xy_reference.value = self.objects.value
