#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import itertools
import os
import random

import numpy as np
import pytest
from discretize import TreeMesh
from geoh5py.objects import BlockModel, Grid2D
from geoh5py.shared.utils import compare_entities
from geoh5py.workspace import Workspace

from geoapps.utils.testing import Geoh5Tester
from geoapps.utils.utils import (
    block_model_2_tensor,
    calculate_2D_trend,
    downsample_grid,
    downsample_xy,
    filter_xy,
    get_locations,
    octree_2_treemesh,
    rotate_xy,
    running_mean,
    sorted_alphanumeric_list,
    sorted_children_dict,
    string_2_numeric,
    tensor_2_block_model,
    treemesh_2_octree,
    weighted_average,
    window_xy,
)

workspace = Workspace("./FlinFlon.geoh5")


def test_string_2_numeric():
    assert string_2_numeric("test") == "test"
    assert string_2_numeric("2.1") == 2.1
    assert string_2_numeric("34") == 34
    assert string_2_numeric("1e-2") == 0.01
    assert string_2_numeric("1.05e2") == 105


def test_sorted_alphanumeric_list():
    test = [
        "Iteration_3.2e-1_data",
        "Iteration_1_data",
        "Iteration_2_data",
        "Iteration_3_data",
        "Iteration_5.11_data",
        "Iteration_5.2_data",
        "Iteration_6_data",
        "Iteration_7_data",
        "Iteration_8e0_data",
        "Iteration_9.0_data",
        "Iteration_10_data",
        "Iteration_11_data",
        "Iteration_2_model",
        "Iteration_12_model",
        "interp_01",
        "interp_02",
        "interp_11",
        "iteration_2_model",
        "iteration_12_model",
        "topo",
        "uncert",
    ]

    sorted_list = sorted_alphanumeric_list(random.sample(test, len(test)))
    assert all([sorted_list[i] == test[i] for i in range(len(test))])


def test_sorted_children_dict(tmp_path):
    ws = Workspace(os.path.join(tmp_path, "test.geoh5"))
    n_x, n_y = 10, 15
    grid = Grid2D.create(
        ws,
        origin=[0, 0, 0],
        u_cell_size=20.0,
        v_cell_size=30.0,
        u_count=n_x,
        v_count=n_y,
        name="test_grid",
        allow_move=False,
    )

    test_data = grid.add_data({"Iteration_10_data": {"values": np.ones(10 * 15)}})
    test_data = grid.add_data({"Iteration_1_data": {"values": np.ones(10 * 15)}})
    test_data = grid.add_data({"Iteration_5_data": {"values": np.ones(10 * 15)}})
    test_data = grid.add_data({"Iteration_3_data": {"values": np.ones(10 * 15)}})
    test_data = grid.add_data({"Iteration_2_data": {"values": np.ones(10 * 15)}})
    test_data = grid.add_data({"Iteration_4_data": {"values": np.ones(10 * 15)}})
    test_data = grid.add_data({"Iteration_9.0_data": {"values": np.ones(10 * 15)}})
    test_data = grid.add_data({"Iteration_8e0_data": {"values": np.ones(10 * 15)}})
    test_data = grid.add_data({"Iteration_11_data": {"values": np.ones(10 * 15)}})
    test_data = grid.add_data({"Iteration_6_data": {"values": np.ones(10 * 15)}})
    test_data = grid.add_data({"Iteration_7_data": {"values": np.ones(10 * 15)}})
    test_data = grid.add_data({"interp_02": {"values": np.ones(10 * 15)}})
    test_data = grid.add_data({"interp_01": {"values": np.ones(10 * 15)}})
    test_data = grid.add_data({"interp_11": {"values": np.ones(10 * 15)}})
    test_data = grid.add_data({"iteration_2_model": {"values": np.ones(10 * 15)}})
    test_data = grid.add_data({"iteration_12_model": {"values": np.ones(10 * 15)}})
    test_data = grid.add_data({"Iteration_2_model": {"values": np.ones(10 * 15)}})
    test_data = grid.add_data({"Iteration_12_model": {"values": np.ones(10 * 15)}})
    test_data = grid.add_data({"topo": {"values": np.ones(10 * 15)}})
    test_data = grid.add_data({"uncert": {"values": np.ones(10 * 15)}})

    d = sorted_children_dict(grid)
    d = list(d.keys())
    assert d[0] == "Iteration_1_data"
    assert d[1] == "Iteration_2_data"
    assert d[7] == "Iteration_8e0_data"
    assert d[8] == "Iteration_9.0_data"
    assert d[-2] == "topo"
    assert d[-1] == "uncert"


def test_rotation_xy():
    vec = np.c_[1, 0, 0]
    rot_vec = rotate_xy(vec, [0, 0], 45)

    assert (
        np.linalg.norm(np.cross(rot_vec, [0.7071, 0.7071, 0])) < 1e-8
    ), "Error on positive rotation about origin."

    rot_vec = rotate_xy(vec, [1, 1], -90)

    assert (
        np.linalg.norm(np.cross(rot_vec, [0, 1, 0])) < 1e-8
    ), "Error on negative rotation about point."


def test_running_mean():
    vec = np.random.randn(100)
    mean_forw = running_mean(vec, method="forward")
    mean_back = running_mean(vec, method="backward")
    mean_cent = running_mean(vec, method="centered")

    mean_test = (vec[1:] + vec[:-1]) / 2

    assert (
        np.linalg.norm(mean_back[:-1] - mean_test) < 1e-12
    ), "Backward averaging does not match expected values."
    assert (
        np.linalg.norm(mean_forw[1:] - mean_test) < 1e-12
    ), "Forward averaging does not match expected values."
    assert (
        np.linalg.norm((mean_test[1:] + mean_test[:-1]) / 2 - mean_cent[1:-1]) < 1e-12
    ), "Centered averaging does not match expected values."


def test_weigted_average():

    # in loc == out loc -> in val == out val
    xyz_out = np.array([[0, 0, 0]])
    xyz_in = np.array([[0, 0, 0]])
    values = [np.array([99])]
    out = weighted_average(xyz_in, xyz_out, values)
    assert out[0] == 99

    # two point same distance away -> arithmetic mean
    xyz_in = np.array([[1, 0, 0], [0, 1, 0]])
    values = [np.array([99, 100])]
    out = weighted_average(xyz_in, xyz_out, values)
    assert (out[0] - 99.5) < 1e-10

    # two points different distance away but close to infinity -> arithmetic mean
    xyz_in = np.array([[1e30, 0, 0], [1e30 + 1, 0, 0]])
    values = [np.array([99, 100])]
    out = weighted_average(xyz_in, xyz_out, values)
    assert (out[0] - 99.5) < 1e-10

    # one point close to infinity, one not -> out val is near-field value
    xyz_in = np.array([[1, 0, 0], [1e30, 0, 0]])
    values = [np.array([99, 100])]
    out = weighted_average(xyz_in, xyz_out, values)
    assert (out[0] - 99.0) < 1e-10

    # one values vector and n out locs -> one out vector of length 20
    xyz_in = np.random.rand(10, 3)
    xyz_out = np.random.rand(20, 3)
    values = [np.random.rand(10)]
    out = weighted_average(xyz_in, xyz_out, values)
    assert len(out) == 1
    assert len(out[0]) == 20

    # two values vectors and n out locs -> two out vectors of length 20 each
    xyz_in = np.random.rand(10, 3)
    xyz_out = np.random.rand(20, 3)
    values = [np.random.rand(10), np.random.rand(10)]
    out = weighted_average(xyz_in, xyz_out, values)
    assert len(out) == 2
    assert len(out[0]) == 20
    assert len(out[1]) == 20

    # max distance keeps out points from average
    xyz_in = np.array([[1, 0, 0], [3, 0, 0]])
    xyz_out = np.array([[0, 0, 0]])
    values = [np.array([1, 100])]
    out = weighted_average(xyz_in, xyz_out, values, max_distance=2)
    assert out[0] == 1

    # n caps the number of points that go into the average
    xyz_in = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    xyz_out = np.array([[0, 0, 0]])
    values = [np.array([1, 2, 3])]
    out = weighted_average(xyz_in, xyz_out, values, n=3)
    assert out[0] == 2
    out = weighted_average(xyz_in, xyz_out, values, n=2)
    assert out[0] == 1.5

    # return indices with n=1 returns closest in loc to the out loc
    xyz_in = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 2]])
    xyz_out = np.array([[0, 0, 0]])
    values = [np.array([1, 2, 3])]
    out, ind = weighted_average(xyz_in, xyz_out, values, n=1, return_indices=True)
    assert ind[0][0] == 1

    # threshold >> r -> arithmetic mean
    xyz_in = np.array([[1, 0, 0], [0, 100, 0], [0, 0, 1000]])
    xyz_out = np.array([[0, 0, 0]])
    values = [np.array([1, 2, 3])]
    out = weighted_average(xyz_in, xyz_out, values, threshold=1e30)
    assert out[0] == 2


def test_treemesh_2_octree(tmp_path):

    geotest = Geoh5Tester(workspace, tmp_path, "test.geoh5")
    ws = geotest.make()
    mesh = TreeMesh([[10] * 16, [10] * 4, [10] * 8], [0, 0, 0])
    mesh.insert_cells([10, 10, 10], mesh.max_level, finalize=True)
    omesh = treemesh_2_octree(ws, mesh, name="test_mesh")
    assert omesh.n_cells == mesh.n_cells
    assert np.all((omesh.centroids - mesh.cell_centers[mesh._ubc_order]) < 1e-14)
    expected_refined_cells = [
        (0, 0, 6),
        (0, 0, 7),
        (1, 0, 6),
        (1, 0, 7),
        (0, 1, 6),
        (0, 1, 7),
        (1, 1, 6),
        (1, 1, 7),
    ]
    ijk_refined = omesh.octree_cells[["I", "J", "K"]][
        omesh.octree_cells["NCells"] == 1
    ].tolist()
    assert np.all([k in ijk_refined for k in expected_refined_cells])
    assert np.all([k in expected_refined_cells for k in ijk_refined])


def test_octree_2_treemesh(tmp_path):

    geotest = Geoh5Tester(workspace, tmp_path, "test.geoh5")
    ws = geotest.make()
    mesh = TreeMesh([[10] * 4, [10] * 4, [10] * 4], [0, 0, 0])
    mesh.insert_cells([5, 5, 5], mesh.max_level, finalize=True)
    omesh = treemesh_2_octree(ws, mesh)
    for p in itertools.product("uvw", repeat=3):
        omesh.origin = [0, 0, 0]
        for axis in "uvw":
            attr = axis + "_cell_size"
            setattr(omesh, attr, np.abs(getattr(omesh, attr)))
        for axis in np.unique(p):
            attr = axis + "_cell_size"
            setattr(omesh, attr, -1 * getattr(omesh, attr))
            omesh.origin["xyz"["uvw".find(axis)]] = 40
        tmesh = octree_2_treemesh(omesh)
        assert np.all((tmesh.cell_centers - mesh.cell_centers) < 1e-14)


def test_window_xy():
    xg, yg = np.meshgrid(np.arange(11), np.arange(11))
    x = xg.ravel()
    y = yg.ravel()
    window = {
        "center": [5, 5],
        "size": [1, 1],
    }
    ind, xw, yw = window_xy(x, y, window)
    assert len(xw) == 1
    assert len(yw) == 1
    assert xw[0] == 5
    assert yw[0] == 5
    assert sum(ind) == 1

    window = {"center": [6, 2.5], "size": [3, 2]}
    ind, xw, yw = window_xy(x, y, window)
    assert [p in xw for p in [5, 6, 7]]
    assert [p in [5, 6, 7] for p in xw]
    assert [p in yw for p in [3, 4]]
    assert [p in [3, 4] for p in yw]


def test_downsample_xy():
    xg, yg = np.meshgrid(np.arange(11), np.arange(11))
    x = xg.ravel()
    y = yg.ravel()
    ind, xd, yd = downsample_xy(x, y, 0)
    assert np.all(x == xd)
    assert np.all(y == yd)

    ind, xd, yd = downsample_xy(x, y, 1)
    assert np.all(x[::2] == xd)
    assert np.all(y[::2] == yd)


def test_downsample_grid():

    # Test a simple grid equal spacing in x, y
    xg, yg = np.meshgrid(np.arange(11), np.arange(11))
    ind, xd, yd = downsample_grid(xg, yg, 2)
    assert np.all(np.diff(yd.reshape(6, 6), axis=0) == 2)
    assert np.all(np.diff(xd.reshape(6, 6), axis=1) == 2)

    # Test a rotated grid equal spacing in u, v
    xy_rot = rotate_xy(np.c_[xg.ravel(), yg.ravel()], [5, 5], 30)
    xg_rot = xy_rot[:, 0].reshape(11, 11)
    yg_rot = xy_rot[:, 1].reshape(11, 11)
    ind, xd, yd = downsample_grid(xg_rot, yg_rot, 2)
    xy = rotate_xy(np.c_[xd, yd], [5, 5], -30)
    xg_test = xy[:, 0].reshape(6, 6)
    yg_test = xy[:, 1].reshape(6, 6)
    np.testing.assert_allclose(np.diff(xg_test, axis=1), np.full((6, 5), 2))
    np.testing.assert_allclose(np.diff(yg_test, axis=0), np.full((5, 6), 2))

    # Test unequal spacing in x, y
    xg, yg = np.meshgrid(np.arange(11), np.linspace(0, 10, 21))
    ind, xd, yd = downsample_grid(xg, yg, 2)
    xg_test = xd.reshape(6, 6)
    yg_test = yd.reshape(6, 6)
    np.testing.assert_allclose(np.diff(xg_test, axis=1), np.full((6, 5), 2))
    np.testing.assert_allclose(np.diff(yg_test, axis=0), np.full((5, 6), 2))


def test_filter_xy():

    xg, yg = np.meshgrid(np.arange(11), np.arange(11))
    xy_rot = rotate_xy(np.c_[xg.ravel(), yg.ravel()], [5, 5], 30)
    xg_rot = xy_rot[:, 0].reshape(11, 11)
    yg_rot = xy_rot[:, 1].reshape(11, 11)
    window = {
        "center": [5, 5],
        "size": [9, 5],
    }
    # Test the windowing functionality
    w_mask = filter_xy(xg, yg, window=window)
    xg_test, yg_test = xg[w_mask].reshape(5, 9), yg[w_mask].reshape(5, 9)
    np.testing.assert_allclose(
        xg_test, np.meshgrid(np.arange(1, 10), np.arange(3, 8))[0]
    )
    np.testing.assert_allclose(
        yg_test, np.meshgrid(np.arange(1, 10), np.arange(3, 8))[1]
    )

    # Test the downsampling functionality
    ds_mask = filter_xy(xg, yg, distance=2)
    xg_test, yg_test = xg[ds_mask].reshape(6, 6), yg[ds_mask].reshape(6, 6)
    np.testing.assert_allclose(np.diff(xg_test, axis=1), np.full((6, 5), 2))
    np.testing.assert_allclose(np.diff(yg_test, axis=0), np.full((5, 6), 2))

    # Test the combo functionality
    comb_mask = filter_xy(xg, yg, distance=2, window=window)
    assert np.all(comb_mask == (w_mask & ds_mask))
    xg_test, yg_test = xg[comb_mask].reshape(2, 4), yg[comb_mask].reshape(2, 4)
    assert np.all((xg_test >= 1) & (xg_test <= 9))
    assert np.all((yg_test >= 3) & (yg_test <= 7))
    np.testing.assert_allclose(np.diff(xg_test, axis=1), np.full((2, 3), 2))
    np.testing.assert_allclose(np.diff(yg_test, axis=0), np.full((1, 4), 2))

    # Test rotation options
    combo_mask = filter_xy(xg_rot, yg_rot, distance=2, window=window, angle=-30)
    xg_test, yg_test = xg_rot[comb_mask], yg_rot[comb_mask]
    xy_rot = rotate_xy(np.c_[xg_test, yg_test], [5, 5], -30)
    xg_rot_test, yg_rot_test = xy_rot[:, 0].reshape(2, 4), xy_rot[:, 1].reshape(2, 4)
    assert np.all((xg_rot_test >= 1) & (xg_rot_test <= 9))
    assert np.all((yg_rot_test >= 3) & (yg_rot_test <= 7))
    np.testing.assert_allclose(np.diff(xg_rot_test, axis=1), np.full((2, 3), 2))
    np.testing.assert_allclose(np.diff(yg_rot_test, axis=0), np.full((1, 4), 2))

    window["azimuth"] = -30
    combo_mask_test = filter_xy(xg_rot, yg_rot, distance=2, window=window)
    assert np.all(combo_mask_test == combo_mask)


def test_detrend_xy():
    xg, yg = np.meshgrid(np.arange(64), np.arange(64))
    xy = np.c_[xg.flatten(), yg.flatten()]
    coefficients = np.random.randn(3)
    values = coefficients[0] + coefficients[1] * xy[:, 1] + coefficients[2] * xy[:, 0]
    ind_nan = np.random.randint(0, high=values.shape[0] - 1, size=32)
    nan_values = values.copy()
    nan_values[ind_nan] = np.nan

    # Should return a plane even for order=5
    comp_trend, comp_params = calculate_2D_trend(xy, nan_values, order=5, method="all")
    np.testing.assert_almost_equal(values, comp_trend)
    # Should return same plane parameter for 'perimeter' or 'all'
    corner_trend, corner_params = calculate_2D_trend(
        xy, nan_values, order=1, method="perimeter"
    )
    np.testing.assert_almost_equal(values, corner_trend)

    with pytest.raises(ValueError) as excinfo:
        calculate_2D_trend(xy[:3, :], nan_values[:3], order=2)
    assert "Provided 3 values for a 2th" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        calculate_2D_trend(xy, nan_values, order=1.1)
    assert "Value of 1.1 provided." in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        calculate_2D_trend(xy, nan_values, order=-2)
    assert "> 0. Value of -2" in str(excinfo.value)


def test_blockmodel_nonuniqueness(tmp_path):
    ws = Workspace(os.path.join(tmp_path, "test.geoh5"))
    nx, ny, nz = 11, 11, 11
    x = np.linspace(0, 10, nx)
    y = np.linspace(0, 10, ny)
    z = np.linspace(0, 10, nz)
    block_model_1 = BlockModel.create(
        ws,
        origin=[0.0, 0.0, -10.0],
        u_cell_delimiters=x,
        v_cell_delimiters=y,
        z_cell_delimiters=z,
        name="test_block_model",
        allow_move=False,
    )
    block_model_2 = BlockModel.create(
        ws,
        origin=[0.0, 0.0, 0.0],
        u_cell_delimiters=x,
        v_cell_delimiters=y,
        z_cell_delimiters=-z,
        name="test_block_model_2",
        allow_move=False,
    )

    tensor_1, _ = block_model_2_tensor(block_model_1, models=[])
    tensor_2, _ = block_model_2_tensor(block_model_2, models=[])
    np.testing.assert_array_equal(tensor_2.cell_centers, tensor_1.cell_centers)
    np.testing.assert_array_equal(tensor_2.origin, tensor_1.origin)

    # Conclusion - can't expect to recover the original block_model.  There isn't
    # a single block model that is equivalent to the tensor mesh.  The circular
    # blockModel -> TensorMesh -> blockModel should pass for a bottom soutwest corner
    # positive cell delimiters original block model though.


def blockmodel_tensormesh_equivalency(ws, origin, x, y, z, ignore_origin=False):

    block_model = BlockModel.create(
        ws,
        origin=origin,
        u_cell_delimiters=x,
        v_cell_delimiters=y,
        z_cell_delimiters=z,
        name="test_block_model",
        allow_move=False,
    )

    data = block_model.add_data(
        {
            "DataValues": {
                "association": "CELL",
                "values": np.random.randint(
                    0, 100, size=len(block_model.centroids)
                ).astype(np.int32),
            }
        }
    )

    tensor_mesh, model = block_model_2_tensor(block_model, models=[data.values])

    block_model_test = tensor_2_block_model(
        ws, tensor_mesh, data={"block_test": model[0]}
    )

    ignore_list = ["_uid", "_name", "_allow_move"]
    if ignore_origin:
        ignore_list += ["_origin"]

    compare_entities(block_model, block_model_test, ignore=ignore_list)


def test_block_model_2_tensor(tmp_path):

    ws = Workspace(os.path.join(tmp_path, "test.geoh5"))

    # Generate a 3D array
    nx, ny, nz = 21, 11, 31

    x = np.linspace(0, 10, nx)
    y = np.linspace(0, 10, ny)
    z = np.linspace(0, 10, nz)

    blockmodel_tensormesh_equivalency(ws, [0.0, 0.0, 0.0], x, y, z)
    blockmodel_tensormesh_equivalency(ws, [10.0, 10.0, 10], x, y, z)
    blockmodel_tensormesh_equivalency(ws, [-10.0, -10.0, -10], x, y, z)


def test_block_model_2_tensor_negative_z(tmp_path):

    ws = Workspace(os.path.join(tmp_path, "test.geoh5"))

    # Generate a 3D array
    nx, ny, nz = 6, 6, 5

    x = np.linspace(0, 10, nx)
    y = np.linspace(0, 5, ny)
    z = np.linspace(0, 8, nz)

    origin = [0.0, 0.0, 0.0]

    block_model = BlockModel.create(
        ws,
        origin=origin,
        u_cell_delimiters=x,
        v_cell_delimiters=y,
        z_cell_delimiters=-z,
        name="test_block_model",
        allow_move=False,
    )

    data = block_model.add_data(
        {
            "DataValues": {
                "association": "CELL",
                "values": np.random.randint(
                    0, 100, size=len(block_model.centroids)
                ).astype(np.int32),
            }
        }
    )

    tensor_mesh, model = block_model_2_tensor(block_model, models=[data.values])

    t2bm = []
    for i, cc in enumerate(block_model.centroids):
        t2bm.append(np.where(np.all(tensor_mesh.cell_centers == cc, axis=1))[0])
    t2bm = np.array(t2bm).flatten()

    np.testing.assert_array_almost_equal(
        block_model.centroids, tensor_mesh.cell_centers[t2bm]
    )
    assert tensor_mesh.x0[2] == np.min(block_model.z_cell_delimiters)
    assert np.all(model[0][t2bm].flatten() == data.values)


def test_get_locations(tmp_path):

    ws = Workspace(os.path.join(tmp_path, "test.geoh5"))
    n_x, n_y = 10, 15
    grid = Grid2D.create(
        ws,
        origin=[0, 0, 0],
        u_cell_size=20.0,
        v_cell_size=30.0,
        u_count=n_x,
        v_count=n_y,
        name="test_grid",
        allow_move=False,
    )
    base_locs = get_locations(ws, grid)

    test_data = grid.add_data({"test_data": {"values": np.ones(10 * 15)}})
    data_locs = get_locations(ws, test_data)

    np.testing.assert_array_equal(base_locs, data_locs)
