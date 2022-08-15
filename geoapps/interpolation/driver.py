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
from geoh5py.ui_json import InputFile
from geoh5py.ui_json.utils import monitored_directory_copy
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree

from geoapps.interpolation.params import DataInterpolationParams
from geoapps.shared_utils.utils import get_locations, weighted_average


class DataInterpolationDriver:
    def __init__(self, params: DataInterpolationParams):
        self.params: DataInterpolationParams = params

    def run(self):
        xyz = get_locations(self.params.geoh5, self.params.objects)

        if xyz is None:
            raise ValueError("Input object has no centroids or vertices.")

        # Create a tree for the input mesh
        tree = cKDTree(xyz)

        xyz_out = get_locations(self.params.geoh5, self.params.out_object)
        xyz_out_orig = xyz_out.copy()

        values, sign, dtype = {}, {}, {}
        field = self.params.data.name

        if isinstance(field, str) and field in "XYZ":
            values[field] = xyz[:, "XYZ".index(field)]
            dtype[field] = values[field].dtype
        else:
            model_in = self.params.geoh5.get_entity(field)[0]
            values[field] = np.asarray(model_in.values, dtype=float)
            dtype[field] = model_in.values.dtype

        values[field][values[field] == self.params.no_data_value] = np.nan
        if self.params.space == "Log":
            sign[field] = np.sign(values[field])
            values[field] = np.log(np.abs(values[field]))
        else:
            sign[field] = np.ones_like(values[field])

        values_interp = {}
        rad, ind = tree.query(xyz_out)
        if self.params.method == "Nearest":

            print("Computing nearest neighbor interpolation")
            # Find nearest cells
            for key, value in values.items():
                values_interp[key] = value[ind]
                sign[key] = sign[key][ind]

            for key in values_interp.keys():
                if self.params.space == "Log":
                    values_interp[key] = sign[key] * np.exp(values_interp[key])
                values_interp[key][
                    np.isnan(values_interp[key])
                ] = self.params.no_data_value
                if self.params.max_distance is not None:
                    values_interp[key][
                        rad > self.params.max_distance
                    ] = self.params.no_data_value
        elif self.params.method == "Inverse Distance":
            print("Computing inverse distance interpolation")
            # Inverse distance
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
            raise ValueError(f"Unrecognized method: {self.params.method}.")
        top = np.zeros(xyz_out.shape[0], dtype="bool")
        bottom = np.zeros(xyz_out.shape[0], dtype="bool")
        if self.params.topography["objects"] is not None:
            if getattr(self.params.topography["objects"], "vertices", None) is not None:
                topo = self.params.topography["objects"].vertices
            else:
                topo = self.params.topography["objects"].centroids

            if self.params.topography["data"] is not None:
                topo[:, 2] = self.params.topography["data"].values

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
        elif self.params.max_depth is not None:
            bottom = xyz_out_orig[:, 2] > self.params.max_depth

        for key in values_interp.keys():
            values_interp[key][top] = self.params.no_data_value
            values_interp[key][bottom] = self.params.no_data_value

        if self.params.xy_extent is not None and self.params.geoh5.get_entity(
            self.params.xy_extent
        ):
            xy_ref = self.params.xy_extent
            if hasattr(xy_ref, "centroids"):
                xy_ref = xy_ref.centroids
            elif hasattr(xy_ref, "vertices"):
                xy_ref = xy_ref.vertices

            tree = cKDTree(xy_ref[:, :2])
            rad, _ = tree.query(xyz_out_orig[:, :2])
            for key in values_interp.keys():
                if self.params.max_distance is not None:
                    values_interp[key][
                        rad > self.params.max_distance
                    ] = self.params.no_data_value

        for key in values_interp.keys():
            if dtype[key] == np.dtype("int32"):
                primitive = "integer"
                vals = np.round(values_interp[key]).astype(dtype[key])
            else:
                primitive = "float"
                vals = values_interp[key].astype(dtype[key])

            self.params.out_object.add_data(
                {key + self.params.ga_group_name: {"values": vals, "type": primitive}}
            )

        if self.params.monitoring_directory is not None and path.exists(
            self.params.monitoring_directory
        ):
            monitored_directory_copy(
                self.params.monitoring_directory, self.params.out_object
            )


if __name__ == "__main__":
    print("Loading geoh5 file . . .")
    file = sys.argv[1]
    ifile = InputFile.read_ui_json(file)
    params_class = DataInterpolationParams(ifile)
    params_class.geoh5.close()
    driver = DataInterpolationDriver(params_class)
    print("Loaded. Starting data transfer . . .")
    with params_class.geoh5.open(mode="r+"):
        driver.run()

    print("Saved to " + params_class.geoh5.h5file)
