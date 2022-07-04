#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


from __future__ import annotations

import sys
from os import path

import numpy as np
from discretize.utils import mesh_utils
from geoh5py.objects import BlockModel, ObjectBase
from geoh5py.ui_json import InputFile
from geoh5py.ui_json.utils import monitored_directory_copy
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree

from geoapps.interpolation.params import DataInterpolationParams
from geoapps.shared_utils.utils import get_locations, weighted_average


class DataInterpolationDriver:
    def __init__(self, params: DataInterpolationParams):
        self.params: DataInterpolationParams = params
        self._unique_object = {}
        self.object_out = None

    def object_base(self, object):
        for entity in self.params.geoh5.get_entity(object):
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

        locs = DataInterpolationDriver.truncate_locs_depths(locs, depth_core)
        depth_core = DataInterpolationDriver.minimum_depth_core(locs, depth_core, h[2])
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

        top_padding = DataInterpolationDriver.find_top_padding(object_out, h[2])
        object_out.origin["z"] += top_padding

        return object_out

    def run(self):
        object_from = self.object_base(self.params.objects)
        xyz = get_locations(self.params.geoh5, object_from).copy()
        if xyz is None:
            return

        if self.params.data is None:
            print("No data selected")
            return

        # Create a tree for the input mesh
        tree = cKDTree(xyz)

        # temp_geoh5 = f"Interpolation_{time():.3f}.geoh5"

        self.object_out = self.object_base(self.params.out_object).copy(
            parent=self.params.geoh5
        )
        xyz_out = get_locations(self.params.geoh5, self.object_out).copy()

        # 3D grid ***
        """
        xyz_ref = get_locations(self._workspace, self.xy_reference.value).copy()
        if xyz_ref is None:
            print(
                "No object selected for 'Lateral Extent'. Defaults to input object."
            )
            xyz_ref = xyz.copy()

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

        self.object_out = DataInterpolationDriver.get_block_model(
            workspace,
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
        """

        xyz_out_orig = xyz_out.copy()

        values, sign, dtype = {}, {}, {}
        field = self.params.data

        if isinstance(field, str) and field in "XYZ":
            values[field] = xyz[:, "XYZ".index(field)]
            dtype[field] = values[field].dtype
        else:
            model_in = self.params.geoh5.get_entity(field)[0]
            values[field] = np.asarray(model_in.values, dtype=float).copy()
            dtype[field] = model_in.values.dtype

        values[field][values[field] == self.params.no_data_value] = np.nan
        if self.params.space == "Log":
            sign[field] = np.sign(values[field])
            values[field] = np.log(np.abs(values[field]))
        else:
            sign[field] = np.ones_like(values[field])

        values_interp = {}
        rad, ind = tree.query(xyz_out)
        if self.params.method == "Linear":

            for key, value in values.items():
                F = LinearNDInterpolator(xyz, value)
                values_interp[key] = F(xyz_out)

        elif self.params.method == "Inverse Distance":

            # ooh could prolly use rotation function here ***
            angle = np.deg2rad((450.0 - np.asarray(self.params.skew_angle)) % 360.0)
            rotation = np.r_[
                np.c_[np.cos(angle), np.sin(angle)],
                np.c_[-np.sin(angle), np.cos(angle)],
            ]
            center = np.mean(xyz, axis=0).reshape((3, 1))
            xyz -= np.kron(center, np.ones(xyz.shape[0])).T
            xyz[:, :2] = np.dot(rotation, xyz[:, :2].T).T
            xyz[:, 1] *= self.params.skew_factor
            xyz_out -= np.kron(center, np.ones(xyz_out.shape[0])).T
            xyz_out[:, :2] = np.dot(rotation, xyz_out[:, :2].T).T
            xyz_out[:, 1] *= self.params.skew_factor
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
            if self.params.space == "Log":
                values_interp[key] = sign[key] * np.exp(values_interp[key])

            values_interp[key][np.isnan(values_interp[key])] = self.params.no_data_value
            values_interp[key][
                rad > self.params.max_distance
            ] = self.params.no_data_value

        top = np.zeros(xyz_out.shape[0], dtype="bool")
        bottom = np.zeros(xyz_out.shape[0], dtype="bool")
        if self.params.topography_options == "Object" and self.params.geoh5.get_entity(
            self.params.topography_objects
        ):

            for entity in self.params.geoh5.get_entity(self.params.topography_objects):
                if isinstance(entity, ObjectBase):
                    topo_obj = entity

            if getattr(topo_obj, "vertices", None) is not None:
                topo = topo_obj.vertices
            else:
                topo = topo_obj.centroids

            if self.params.topography_data is not None:
                topo[:, 2] = self.params.geoh5.get_entity(self.params.topography_data)[
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
            if self.params.max_depth is not None:
                bottom = np.abs(xyz_out_orig[:, 2] - z_interp) > self.params.max_depth

        elif (
            self.params.topography_options == "Constant"
            and self.params.topography_constant is not None
        ):
            top = xyz_out_orig[:, 2] > self.params.topography_constant
            if self.params.max_depth is not None:
                bottom = (
                    np.abs(xyz_out_orig[:, 2] - self.params.topography_constant)
                    > self.params.max_depth
                )

        for key in values_interp.keys():
            values_interp[key][top] = self.params.no_data_value
            values_interp[key][bottom] = self.params.no_data_value

        if self.params.xy_extent is not None and self.params.geoh5.get_entity(
            self.params.xy_extent
        ):

            for entity in self.params.geoh5.get_entity(self.params.xy_extent):
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
                    rad > self.params.max_distance
                ] = self.params.no_data_value

        self.object_out.workspace.open()
        for key in values_interp.keys():
            if dtype[field] == np.dtype("int32"):
                primitive = "integer"
                vals = np.round(values_interp[key]).astype(dtype[field])
            else:
                primitive = "float"
                vals = values_interp[key].astype(dtype[field])

            self.object_out.add_data(
                {
                    # self.data.uid_name_map[key]
                    self.params.data.name
                    + self.params.ga_group_name: {"values": vals, "type": primitive}
                }
            )

        self.object_out.workspace.close()

        if self.params.monitoring_directory is not None and path.exists(
            self.params.monitoring_directory
        ):
            monitored_directory_copy(self.params.monitoring_directory, self.object_out)


if __name__ == "__main__":
    print("Loading geoh5 file . . .")
    file = sys.argv[1]
    ifile = InputFile.read_ui_json(file)
    params = DataInterpolationParams(ifile)
    driver = DataInterpolationDriver(params)
    print("Loaded. Running data transfer . . .")
    driver.run()
    print("Saved to " + ifile.path)
