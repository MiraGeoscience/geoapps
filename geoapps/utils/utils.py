#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import gc
import json
import os

import dask
import dask.array as da
import geoh5py
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
from geoh5py.data import FloatData, IntegerData
from geoh5py.groups import Group
from geoh5py.objects import (
    BlockModel,
    CurrentElectrode,
    Grid2D,
    Octree,
    PotentialElectrode,
    Surface,
)
from geoh5py.shared import Entity
from geoh5py.workspace import Workspace
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, interp1d
from scipy.spatial import ConvexHull, Delaunay, cKDTree
from shapely.geometry import LineString, mapping
from SimPEG.electromagnetics.static.resistivity import Survey
from skimage.measure import marching_cubes
from sklearn.neighbors import KernelDensity


def get_surface_parts(surface: Surface) -> np.ndarray:
    """
    Find the connected cells from a surface.
    :param surface: Input surface with cells property.
    :return parts: shape(*, 3)
        Array of parts for each of the surface vertices.
    """
    cell_sorted = np.sort(surface.cells, axis=1)
    cell_sorted = cell_sorted[np.argsort(cell_sorted[:, 0]), :]

    parts = np.zeros(surface.vertices.shape[0], dtype="int")
    count = 1
    for ii in range(cell_sorted.shape[0] - 1):

        if (
            (cell_sorted[ii, 0] in cell_sorted[ii + 1 :, :])
            or (cell_sorted[ii, 1] in cell_sorted[ii + 1 :, :])
            or (cell_sorted[ii, 2] in cell_sorted[ii + 1 :, :])
        ):
            parts[cell_sorted[ii, :]] = count
        else:
            parts[cell_sorted[ii, :]] = count
            count += 1

    parts[cell_sorted[-1, :]] = count

    return parts


def tensor_2_block_model(workspace, mesh, name=None, parent=None, data={}):
    """
    Function to convert a tensor mesh from :obj:`~discretize.TensorMesh` to
    :obj:`~geoh5py.objects.block_model.BlockModel`
    """

    block_model = BlockModel.create(
        workspace,
        origin=[mesh.x0[0], mesh.x0[1], mesh.x0[2]],
        u_cell_delimiters=(mesh.vectorNx - mesh.x0[0]),
        v_cell_delimiters=(mesh.vectorNy - mesh.x0[1]),
        z_cell_delimiters=(mesh.vectorNz - mesh.x0[2]),
        name=name,
        parent=parent,
    )

    for name, model in data.items():
        modelMat = mesh.reshape(model, "CC", "CC", "M")

        # Transpose the axes
        modelMatT = modelMat.transpose((2, 0, 1))
        modelMatTR = modelMatT.reshape((-1, 1), order="F")

        block_model.add_data({name: {"values": modelMatTR}})

    return block_model


def block_model_2_tensor(block_model, models=[]):
    """
    Function to convert a :obj:`~geoh5py.objects.block_model.BlockModel`
    to tensor mesh :obj:`~discretize.TensorMesh`
    """

    from discretize import TensorMesh

    tensor = TensorMesh(
        [
            np.abs(block_model.u_cells),
            np.abs(block_model.v_cells),
            np.abs(block_model.z_cells),
        ],
        x0="CC0",
    )
    tensor.x0 = [
        block_model.origin["x"] + block_model.u_cells[block_model.u_cells < 0].sum(),
        block_model.origin["y"] + block_model.v_cells[block_model.v_cells < 0].sum(),
        block_model.origin["z"] + block_model.z_cells[block_model.z_cells < 0].sum(),
    ]
    out = []

    for model in models:
        values = model.copy().reshape((tensor.nCz, tensor.nCx, tensor.nCy), order="F")

        if tensor.x0[2] != block_model.origin["z"]:
            values = values[::-1, :, :]
        values = np.transpose(values, (1, 2, 0))

        values = values.reshape((-1, 1), order="F")
        out += [values]

    return tensor, out


def csv_2_zarr(input_csv, out_dir="zarr", rowchunks=100000, dask_chunks="64MB"):
    """
    Zarr conversion for large CSV files
    NOTE: Need testing
    """
    # Need to run this part only once
    if ~os.path.exists(out_dir):
        for ii, chunk in enumerate(pd.read_csv(input_csv, chunksize=rowchunks)):
            array = chunk.to_numpy()[1:, :]
            da_array = da.from_array(array, chunks=dask_chunks)
            da.to_zarr(da_array, url=out_dir + rf"\Tile{ii}")

    # Just read the header
    header = pd.read_csv(input_csv, nrows=1)

    # Stack all the blocks in one big zarr
    count = len([name for name in os.listdir(out_dir)])
    dask_arrays = []
    for ii in range(count):
        block = da.from_zarr(out_dir + f"/Tile{ii}")
        dask_arrays.append(block)

    return header, da.vstack(dask_arrays)


def data_2_zarr(h5file, entity_name, downsampling=1, fields=[], zarr_file="data.zarr"):
    """
    Convert an data entity and values to a dictionary of zarr's
    """

    workspace = Workspace(h5file)
    entity = workspace.get_entity(entity_name)[0]

    if getattr(entity, "vertices", None) is not None:
        n_data = entity.n_vertices
    elif getattr(entity, "centroids", None) is not None:
        n_data = entity.n_cells
    del workspace, entity

    vec_len = int(np.ceil(n_data / downsampling))

    def load(field):
        """
        Load one column from geoh5
        """
        workspace = Workspace(h5file)
        entity = workspace.get_entity(entity_name)[0]
        obj = entity.get_data(field)[0]
        values = obj.values[::downsampling]
        if isinstance(obj, FloatData) and values.shape[0] == vec_len:
            values[(values > 1e-38) * (values < 2e-38)] = -99999
        else:
            values = np.ones(vec_len) * -99999
        del workspace, obj, entity
        gc.collect()
        return values

    row = dask.delayed(load, pure=True)

    make_rows = [row(field) for field in fields]

    delayed_array = [
        da.from_delayed(
            make_row, dtype=np.float32, shape=(np.ceil(n_data / downsampling),)
        )
        for make_row in make_rows
    ]

    stack = da.vstack(delayed_array)

    if os.path.exists(zarr_file):

        data_mat = da.from_zarr(zarr_file)

        if np.all(
            np.r_[
                np.any(np.r_[data_mat.chunks[0]] == stack.chunks[0]),
                np.any(np.r_[data_mat.chunks[1]] == stack.chunks[1]),
                np.r_[data_mat.shape] == np.r_[stack.shape],
            ]
        ):
            # Check that loaded G matches supplied data and mesh
            print("Zarr file detected with same shape and chunksize ... re-loading")

            return data_mat
        else:

            print("Zarr file detected with wrong shape and chunksize ... over-writing")

    with ProgressBar():
        print("Saving G to zarr: " + zarr_file)
        data_mat = da.to_zarr(
            stack,
            zarr_file,
            compute=True,
            return_stored=True,
            overwrite=True,
        )

    return data_mat


def rotate_vertices(xyz, center, phi, theta):
    """
    Rotate scatter points in column format around a center location
    INPUT
    :param: xyz nDx3 matrix
    :param: center xyz location of rotation
    :param: theta angle rotation around z-axis
    :param: phi angle rotation around x-axis
    """
    xyz -= np.kron(np.ones((xyz.shape[0], 1)), np.r_[center])

    phi = -np.deg2rad(np.asarray(phi))
    theta = np.deg2rad((450.0 - np.asarray(theta)) % 360.0)

    Rx = np.asarray(
        [[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]]
    )

    Rz = np.asarray(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )

    R = Rz.dot(Rx)

    xyzRot = R.dot(xyz.T).T

    return xyzRot + np.kron(np.ones((xyz.shape[0], 1)), np.r_[center])


def rotate_azimuth_dip(azimuth, dip):
    """
    dipazm_2_xyz(dip,azimuth)
    Function converting degree angles for dip and azimuth from north to a
    3-components in cartesian coordinates.
    INPUT
    dip     : Value or vector of dip from horizontal in DEGREE
    azimuth   : Value or vector of azimuth from north in DEGREE
    OUTPUT
    M       : [n-by-3] Array of xyz components of a unit vector in cartesian
    Created on Dec, 20th 2015
    @author: dominiquef
    """

    azimuth = np.asarray(azimuth)
    dip = np.asarray(dip)

    # Number of elements
    nC = azimuth.size

    M = np.zeros((nC, 3))

    # Modify azimuth from North to cartesian-X
    inc = -np.deg2rad(np.asarray(dip))
    dec = np.deg2rad((450.0 - np.asarray(azimuth)) % 360.0)

    M[:, 0] = np.cos(inc) * np.cos(dec)
    M[:, 1] = np.cos(inc) * np.sin(dec)
    M[:, 2] = np.sin(inc)

    return M


class RectangularBlock:
    """
    Define a rotated rectangular block in 3D space
    :param
        - length, width, depth: width, length and height of prism
        - center : center of prism in horizontal plane
        - dip, azimuth : dip and azimuth of prism
    """

    def __init__(self, **kwargs):

        self._center = [0.0, 0.0, 0.0]
        self._length = 1.0
        self._width = 1.0
        self._depth = 1.0
        self._dip = 0.0
        self._azimuth = 0.0
        self._vertices = None

        self.triangles = np.vstack(
            [
                [0, 1, 2],
                [1, 2, 3],
                [0, 1, 4],
                [1, 4, 5],
                [1, 3, 5],
                [3, 5, 7],
                [2, 3, 6],
                [3, 6, 7],
                [0, 2, 4],
                [2, 4, 6],
                [4, 5, 6],
                [5, 6, 7],
            ]
        )

        for attr, item in kwargs.items():
            try:
                setattr(self, attr, item)
            except AttributeError:
                continue

    @property
    def center(self):
        """Prism center"""
        return self._center

    @center.setter
    def center(self, value):
        self._center = value
        self._vertices = None

    @property
    def length(self):
        """"""
        return self._length

    @length.setter
    def length(self, value):
        self._length = value
        self._vertices = None

    @property
    def width(self):
        """"""
        return self._width

    @width.setter
    def width(self, value):
        self._width = value
        self._vertices = None

    @property
    def depth(self):
        """"""
        return self._depth

    @depth.setter
    def depth(self, value):
        self._depth = value
        self._vertices = None

    @property
    def dip(self):
        """"""
        return self._dip

    @dip.setter
    def dip(self, value):
        self._dip = value
        self._vertices = None

    @property
    def azimuth(self):
        """"""
        return self._azimuth

    @azimuth.setter
    def azimuth(self, value):
        self._azimuth = value
        self._vertices = None

    @property
    def vertices(self):
        """
        Prism eight corners in 3D space
        """

        if getattr(self, "_vertices", None) is None:
            x1, x2 = [
                -self.length / 2.0 + self.center[0],
                self.length / 2.0 + self.center[0],
            ]
            y1, y2 = [
                -self.width / 2.0 + self.center[1],
                self.width / 2.0 + self.center[1],
            ]
            z1, z2 = [
                -self.depth / 2.0 + self.center[2],
                self.depth / 2.0 + self.center[2],
            ]

            block_xyz = np.asarray(
                [
                    [x1, x2, x1, x2, x1, x2, x1, x2],
                    [y1, y1, y2, y2, y1, y1, y2, y2],
                    [z1, z1, z1, z1, z2, z2, z2, z2],
                ]
            )

            xyz = rotate_vertices(block_xyz.T, self.center, self.dip, self.azimuth)

            self._vertices = xyz

        return self._vertices


def raw_moment(data, i_order, j_order):
    nrows, ncols = data.shape
    y_indices, x_indicies = np.mgrid[:nrows, :ncols]

    return (data * x_indicies**i_order * y_indices**j_order).sum()


def moments_cov(data):
    data_sum = data.sum()
    m10 = raw_moment(data, 1, 0)
    m01 = raw_moment(data, 0, 1)
    x_centroid = m10 / data_sum
    y_centroid = m01 / data_sum
    u11 = (raw_moment(data, 1, 1) - x_centroid * m01) / data_sum
    u20 = (raw_moment(data, 2, 0) - x_centroid * m10) / data_sum
    u02 = (raw_moment(data, 0, 2) - y_centroid * m01) / data_sum
    cov = np.array([[u20, u11], [u11, u02]])
    return [x_centroid, y_centroid], cov


def ij_2_ind(coordinates, shape):
    """
    Return the index of ij coordinates
    """
    return [ij[0] * shape[1] + ij[1] for ij in coordinates]


def ind_2_ij(indices, shape):
    """
    Return the index of ij coordinates
    """
    return [[int(np.floor(ind / shape[1])), ind % shape[1]] for ind in indices]


def get_neighbours(index, shape):
    """
    Get all neighbours of cell in a 2D grid
    """
    j, i = int(np.floor(index / shape[1])), index % shape[1]
    vec_i = np.r_[i - 1, i, i + 1]
    vec_j = np.r_[j - 1, j, j + 1]

    vec_i = vec_i[(vec_i >= 0) * (vec_i < shape[1])]
    vec_j = vec_j[(vec_j >= 0) * (vec_j < shape[0])]

    ii, jj = np.meshgrid(vec_i, vec_j)

    return ij_2_ind(np.c_[jj.ravel(), ii.ravel()].tolist(), shape)


def get_active_neighbors(index, shape, model, threshold, blob_indices):
    """
    Given an index, append to a list if active
    """
    out = []
    for ind in get_neighbours(index, shape):
        if (model[ind] > threshold) and (ind not in blob_indices):
            out.append(ind)
    return out


def get_blob_indices(index, shape, model, threshold, blob_indices=[]):
    """
    Function to return indices of cells inside a model value blob
    """
    out = get_active_neighbors(index, shape, model, threshold, blob_indices)

    for neigh in out:
        blob_indices += [neigh]
        blob_indices = get_blob_indices(
            neigh, shape, model, threshold, blob_indices=blob_indices
        )

    return blob_indices


def load_json_params(file: str):
    """
    Read input parameters from json
    """
    with open(file) as f:
        input_dict = json.load(f)

    params = {}
    for key, param in input_dict.items():
        if isinstance(param, dict):
            params[key] = param["value"]
        else:
            params[key] = param

    return params


def direct_current_from_simpeg(
    workspace: Workspace, survey: Survey, name: str = None, data: dict = None
):
    """
    Convert a inversion direct-current survey to geoh5 format.
    """
    u_src_poles, src_pole_id = np.unique(
        np.r_[survey.locations_a, survey.locations_b], axis=0, return_inverse=True
    )
    n_src = int(src_pole_id.shape[0] / 2.0)
    u_src_cells, src_id = np.unique(
        np.c_[src_pole_id[:n_src], src_pole_id[n_src:]], axis=0, return_inverse=True
    )
    u_rcv_poles, rcv_pole_id = np.unique(
        np.r_[survey.locations_m, survey.locations_n], axis=0, return_inverse=True
    )
    n_rcv = int(rcv_pole_id.shape[0] / 2.0)
    u_rcv_cells = np.c_[rcv_pole_id[:n_rcv], rcv_pole_id[n_rcv:]]
    currents = CurrentElectrode.create(
        workspace, name=name, vertices=u_src_poles, cells=u_src_cells.astype("uint32")
    )
    currents.add_default_ab_cell_id()

    potentials = PotentialElectrode.create(
        workspace, name=name, vertices=u_rcv_poles, cells=u_rcv_cells.astype("uint32")
    )
    potentials.current_electrodes = currents
    potentials.ab_cell_id = np.asarray(src_id + 1, dtype="int32")

    if data is not None:
        potentials.add_data({key: {"values": value} for key, value in data.items()})

    return currents, potentials


colors = [
    "#000000",
    "#FFFF00",
    "#1CE6FF",
    "#FF34FF",
    "#FF4A46",
    "#008941",
    "#006FA6",
    "#A30059",
    "#FFDBE5",
    "#7A4900",
    "#0000A6",
    "#63FFAC",
    "#B79762",
    "#004D43",
    "#8FB0FF",
    "#997D87",
    "#5A0007",
    "#809693",
    "#FEFFE6",
    "#1B4400",
    "#4FC601",
    "#3B5DFF",
    "#4A3B53",
    "#FF2F80",
    "#61615A",
    "#BA0900",
    "#6B7900",
    "#00C2A0",
    "#FFAA92",
    "#FF90C9",
    "#B903AA",
    "#D16100",
    "#DDEFFF",
    "#000035",
    "#7B4F4B",
    "#A1C299",
    "#300018",
    "#0AA6D8",
    "#013349",
    "#00846F",
    "#372101",
    "#FFB500",
    "#C2FFED",
    "#A079BF",
    "#CC0744",
    "#C0B9B2",
    "#C2FF99",
    "#001E09",
    "#00489C",
    "#6F0062",
    "#0CBD66",
    "#EEC3FF",
    "#456D75",
    "#B77B68",
    "#7A87A1",
    "#788D66",
    "#885578",
    "#FAD09F",
    "#FF8A9A",
    "#D157A0",
    "#BEC459",
    "#456648",
    "#0086ED",
    "#886F4C",
    "#34362D",
    "#B4A8BD",
    "#00A6AA",
    "#452C2C",
    "#636375",
    "#A3C8C9",
    "#FF913F",
    "#938A81",
    "#575329",
    "#00FECF",
    "#B05B6F",
    "#8CD0FF",
    "#3B9700",
    "#04F757",
    "#C8A1A1",
    "#1E6E00",
    "#7900D7",
    "#A77500",
    "#6367A9",
    "#A05837",
    "#6B002C",
    "#772600",
    "#D790FF",
    "#9B9700",
    "#549E79",
    "#FFF69F",
    "#201625",
    "#72418F",
    "#BC23FF",
    "#99ADC0",
    "#3A2465",
    "#922329",
    "#5B4534",
    "#FDE8DC",
    "#404E55",
    "#0089A3",
    "#CB7E98",
    "#A4E804",
    "#324E72",
    "#6A3A4C",
]
