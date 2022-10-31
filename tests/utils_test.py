#  Copyright (c) 2022 Mira Geoscience Ltd.
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

import geoh5py.objects
import numpy as np
import pytest
from discretize import TreeMesh
from geoh5py.objects import Grid2D
from geoh5py.objects.surveys.direct_current import CurrentElectrode, PotentialElectrode
from geoh5py.workspace import Workspace

from geoapps.driver_base.utils import running_mean, treemesh_2_octree
from geoapps.inversion.utils import calculate_2D_trend
from geoapps.shared_utils.utils import (
    cell_centers_to_faces,
    downsample_grid,
    downsample_xy,
    drape_2_tensor,
    filter_xy,
    get_locations,
    octree_2_treemesh,
    rotate_xyz,
    weighted_average,
    window_xy,
)
from geoapps.utils import warn_module_not_found
from geoapps.utils.list import find_value, sorted_alphanumeric_list
from geoapps.utils.models import RectangularBlock, get_drape_model
from geoapps.utils.statistics import is_outlier
from geoapps.utils.string import string_to_numeric
from geoapps.utils.surveys import (
    compute_alongline_distance,
    extract_dcip_survey,
    find_endpoints,
    find_unique_tops,
    new_neighbors,
    split_dcip_survey,
    survey_lines,
)
from geoapps.utils.testing import Geoh5Tester
from geoapps.utils.workspace import sorted_children_dict

geoh5 = Workspace("./FlinFlon.geoh5")
dc_geoh5 = "./FlinFlon_dcip.geoh5"


def test_get_drape_model(tmp_path):
    ws = Workspace(os.path.join(tmp_path, "test.geoh5"))
    x = np.arange(11)
    y = -x + 10
    locs = np.c_[x, y, np.zeros_like(x)]
    h = [0.5, 0.5]
    depth_core = 5.0
    pads = [0, 0, 0, 0]  # [5, 5, 3, 1]
    expansion_factor = 1.1
    model, mesh, sorting = get_drape_model(  # pylint: disable=W0632
        ws,
        "drape_test",
        locs,
        h,
        depth_core,
        pads,
        expansion_factor,
        return_colocated_mesh=True,
        return_sorting=True,
    )
    ws.close()
    resorted_mesh_centers = mesh.cell_centers[sorting, :]
    model_centers = compute_alongline_distance(model.centroids)
    model_centers[:, 0] += h[0] / 2
    assert np.allclose(model_centers, resorted_mesh_centers)


def test_find_unique_tops_xz():
    x = np.linspace(0, 1, 5)
    z = np.linspace(-1, 1, 4)
    X, Z = np.meshgrid(x, z)
    Z[:, 2] += 1
    Y = np.zeros_like(X)
    locs = np.c_[X.flatten(), Y.flatten(), Z.flatten()]
    test = find_unique_tops(locs)
    assert test[2, 2] == 2


def test_find_unique_tops_xyz():
    x = np.arange(-5, 6)
    y = -x
    z = np.arange(-10, 1)
    X, Z = np.meshgrid(x, z)
    locs = np.c_[X.flatten(), np.tile(y, len(x)), Z.flatten()]
    tops = find_unique_tops(locs)
    assert np.all(tops[:, 2] == 0)
    assert np.allclose(tops[:, :2], np.c_[x, y])


def test_compute_alongline_distance():
    x = np.arange(11)
    y = -x + 10
    locs = np.c_[x, y]
    test = compute_alongline_distance(locs)
    assert np.max(test) == np.sqrt(2) * 10

    x = np.linspace(0, 1, 5)
    z = np.linspace(-1, 1, 4)
    X, Z = np.meshgrid(x, z)
    Z[:, 2] += 1
    Y = np.zeros_like(X)
    locs = np.c_[X.flatten(), Y.flatten(), Z.flatten()]
    test = compute_alongline_distance(locs)
    assert True


def test_find_endpoints():
    x = np.arange(11)
    y = -x + 10
    p = find_endpoints(np.c_[x, y])
    assert np.allclose(p, [[10, 0], [0, 10]])


def test_cell_centers_to_faces():
    test_faces = np.array([0, 3, 5, 6, 7, 8, 10, 13], dtype=float)
    centers = running_mean(test_faces, width=1, method="forward")[1:]
    faces = cell_centers_to_faces(centers)
    assert np.allclose(test_faces, faces)


def test_find_unique_tops():
    x = np.arange(-5, 6)
    y = -x
    z = np.arange(-10, 1)
    X, Z = np.meshgrid(x, z)
    locs = np.c_[X.flatten(), np.tile(y, len(x)), Z.flatten()]
    tops = find_unique_tops(locs)
    assert np.all(tops[:, 2] == 0)
    assert np.allclose(tops[:, :2], np.c_[x, y])


def test_is_outlier():
    assert is_outlier([25.1, 25.3], 50.0)
    assert not is_outlier([25.1, 25.3], 25.2)
    assert is_outlier([25.1, 25.3], 25.4, 1)
    assert not is_outlier([25.1, 25.3], 25.4, 3)
    assert is_outlier([25, 25], 26)
    assert not is_outlier([25, 25], 25)


def test_new_neighbors():

    nodes = [2, 3, 4, 5, 6]
    dist = np.array([25, 50, 0])
    neighbors = np.array([1, 2, 3])
    neighbor_id = new_neighbors(dist, neighbors, nodes)
    assert len(neighbor_id) == 1
    assert neighbor_id[0] == 1


def test_survey_lines(tmp_path):

    name = "TestCurrents"
    n_data = 15
    path = tmp_path / r"testDC.geoh5"
    workspace = Workspace(path)

    # Create sources along line
    x_loc, y_loc = np.meshgrid(np.linspace(0, 10, n_data), np.arange(-1, 3))
    vertices = np.c_[x_loc.ravel(), y_loc.ravel(), np.zeros_like(x_loc).ravel()]
    parts = np.kron(np.arange(4), np.ones(n_data)).astype("int")
    currents = CurrentElectrode.create(
        workspace, name=name, vertices=vertices, parts=parts
    )
    currents.add_default_ab_cell_id()
    potentials = PotentialElectrode.create(
        workspace, name=name + "_rx", vertices=vertices
    )
    n_dipoles = 9
    dipoles = []
    current_id = []
    for val in currents.ab_cell_id.values:
        cell_id = int(currents.ab_map[val]) - 1

        for dipole in range(n_dipoles):
            dipole_ids = currents.cells[cell_id, :] + 2 + dipole

            if (
                any(dipole_ids > (potentials.n_vertices - 1))
                or len(np.unique(parts[dipole_ids])) > 1
            ):
                continue

            dipoles += [dipole_ids]
            current_id += [val]

    potentials.cells = np.vstack(dipoles).astype("uint32")
    potentials.ab_cell_id = np.hstack(current_id).astype("int32")
    potentials.current_electrodes = currents
    currents.potential_electrodes = potentials
    _ = survey_lines(potentials, [-1, -1], save="test_line_ids")
    assert np.all(
        np.unique(workspace.get_entity("test_line_ids")[0].values) == np.arange(1, 5)
    )


def test_extract_dcip_survey(tmp_path):

    name = "TestCurrents"
    n_data = 12
    path = tmp_path / r"testDC.geoh5"
    workspace = Workspace(path)

    # Create sources along line
    x_loc, y_loc = np.meshgrid(np.arange(n_data), np.arange(-1, 3))
    vertices = np.c_[x_loc.ravel(), y_loc.ravel(), np.zeros_like(x_loc).ravel()]
    parts = np.kron(np.arange(4), np.ones(n_data)).astype("int")
    currents = CurrentElectrode.create(
        workspace, name=name, vertices=vertices, parts=parts
    )
    currents.add_default_ab_cell_id()
    potentials = PotentialElectrode.create(
        workspace, name=name + "_rx", vertices=vertices
    )
    n_dipoles = 9
    dipoles = []
    current_id = []
    for val in currents.ab_cell_id.values:
        cell_id = int(currents.ab_map[val]) - 1

        for dipole in range(n_dipoles):
            dipole_ids = currents.cells[cell_id, :] + 2 + dipole

            if (
                any(dipole_ids > (potentials.n_vertices - 1))
                or len(np.unique(parts[dipole_ids])) > 1
            ):
                continue

            dipoles += [dipole_ids]
            current_id += [val]

    potentials.cells = np.vstack(dipoles).astype("uint32")
    potentials.ab_cell_id = np.hstack(current_id).astype("int32")
    potentials.current_electrodes = currents
    currents.potential_electrodes = potentials
    extract_dcip_survey(workspace, potentials, parts, 3, "test_survey_line")
    assert workspace.get_entity("test_survey_line 3")[0] is not None


def test_split_dcip_survey(tmp_path):

    name = "TestCurrents"
    n_data = 12
    path = tmp_path / r"testDC.geoh5"
    workspace = Workspace(path)

    # Create sources along line
    x_loc, y_loc = np.meshgrid(np.arange(n_data), np.arange(-1, 3))
    vertices = np.c_[x_loc.ravel(), y_loc.ravel(), np.zeros_like(x_loc).ravel()]
    parts = np.kron(np.arange(4), np.ones(n_data)).astype("int")
    currents = CurrentElectrode.create(
        workspace, name=name, vertices=vertices, parts=parts
    )
    currents.add_default_ab_cell_id()
    potentials = PotentialElectrode.create(
        workspace, name=name + "_rx", vertices=vertices
    )
    n_dipoles = 9
    dipoles = []
    current_id = []
    for val in currents.ab_cell_id.values:
        cell_id = int(currents.ab_map[val]) - 1

        for dipole in range(n_dipoles):
            dipole_ids = currents.cells[cell_id, :] + 2 + dipole

            if (
                any(dipole_ids > (potentials.n_vertices - 1))
                or len(np.unique(parts[dipole_ids])) > 1
            ):
                continue

            dipoles += [dipole_ids]
            current_id += [val]

    potentials.cells = np.vstack(dipoles).astype("uint32")
    potentials.ab_cell_id = np.hstack(current_id).astype("int32")
    potentials.current_electrodes = currents
    currents.potential_electrodes = potentials
    split_dcip_survey(potentials, parts + 1, "DC Survey Line", workspace)
    assert workspace.get_entity("DC Survey Line 4")[0] is not None


def test_rectangular_block():
    block = RectangularBlock(
        center=[10.0, 10.0, 10.0],
        length=10.0,
        width=10.0,
        depth=10.0,
        dip=0.0,
        azimuth=0.0,
    )
    vertices = block.vertices.tolist()
    assert [15.0, 5.0, 5.0] in vertices
    assert [15.0, 15.0, 5.0] in vertices
    assert [5.0, 5.0, 5.0] in vertices
    assert [5.0, 15.0, 5.0] in vertices
    assert [15.0, 5.0, 15.0] in vertices
    assert [15.0, 15.0, 15.0] in vertices
    assert [5.0, 5.0, 15.0] in vertices
    assert [5.0, 15.0, 15.0] in vertices

    block = RectangularBlock(
        center=[0.0, 0.0, 0.0], length=0.0, width=10.0, depth=0.0, dip=45.0, azimuth=0.0
    )
    pos = (5 * np.cos(np.deg2rad(45))).round(5)
    vertices = block.vertices.round(5).tolist()
    assert [pos, 0.0, pos] in vertices
    assert [-pos, 0.0, -pos] in vertices

    block = RectangularBlock(
        center=[0.0, 0.0, 0.0],
        length=0.0,
        width=0.0,
        depth=10.0,
        dip=0.0,
        azimuth=90.0,
        reference="top",
    )
    vertices = block.vertices.round(5).tolist()
    assert [0.0, 0.0, -10.0] in vertices
    assert [0.0, 0.0, 0.0] in vertices

    block = RectangularBlock(
        center=[0.0, 0.0, 0.0],
        length=10.0,
        width=10.0,
        depth=10.0,
        dip=0.0,
        azimuth=45.0,
    )

    pos = (10 * np.cos(np.deg2rad(45))).round(5)
    vertices = block.vertices.round(5).tolist()
    assert [0.0, -pos, -5.0] in vertices
    assert [pos, 0.0, -5.0] in vertices
    assert [-pos, 0.0, -5.0] in vertices
    assert [0.0, pos, -5.0] in vertices
    assert [0.0, -pos, 5.0] in vertices
    assert [pos, 0.0, 5.0] in vertices
    assert [-pos, 0.0, 5.0] in vertices
    assert [0.0, pos, 5.0] in vertices

    with pytest.raises(ValueError) as error:
        setattr(block, "center", -180.0)

    assert "Input value for 'center' must be a list of floats len(3)." in str(error)

    for attr in ["length", "width", "depth"]:
        with pytest.raises(ValueError) as error:
            setattr(block, attr, -10.0)

        assert f"Input value for '{attr}' must be a float >0." in str(error)

    with pytest.raises(ValueError) as error:
        setattr(block, "dip", -180.0)

    assert (
        "Input value for 'dip' must be a float on the interval [-90, 90] degrees."
        in str(error)
    )

    with pytest.raises(ValueError) as error:
        setattr(block, "azimuth", -450.0)

    assert (
        "Input value for 'azimuth' must be a float on the interval [-360, 360] degrees."
        in str(error)
    )

    with pytest.raises(ValueError) as error:
        setattr(block, "reference", "abc")

    assert (
        "Input value for 'reference' point should be a str from ['center', 'top']."
        in str(error)
    )


def test_find_value():
    labels = ["inversion_01_model", "inversion_01_data", "inversion_02_model"]
    assert find_value(labels, ["data"]) == "inversion_01_data"
    assert find_value(labels, ["inversion", "02"]) == "inversion_02_model"
    assert find_value(labels, ["inversion"]) == "inversion_02_model"
    assert find_value(labels, ["lskdfjsd"]) is None
    labels = [["inversion_01_model", 1], ["inversion_01_data", 2]]
    assert find_value(labels, ["model"]) == 1
    assert find_value(labels, ["data"]) == 2
    assert find_value(labels, ["lskdjf"]) is None


def test_string_to_numeric():
    assert string_to_numeric("test") == "test"
    assert string_to_numeric("2.1") == 2.1
    assert string_to_numeric("34") == 34
    assert string_to_numeric("1e-2") == 0.01
    assert string_to_numeric("1.05e2") == 105


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
    assert all(elem == tester for elem, tester in zip(sorted_list, test))


def test_no_warn_module_not_found(recwarn):
    with warn_module_not_found():
        import os as test_import  # pylint: disable=W0404

    assert test_import == os

    with warn_module_not_found():
        from os import system as test_import_from
    assert test_import_from == os.system

    with warn_module_not_found():
        import geoh5py.objects as test_import_submodule  # pylint: disable=W0404
    assert test_import_submodule == geoh5py.objects

    with warn_module_not_found():
        from geoh5py.objects import ObjectBase as test_import_from_submodule
    assert test_import_from_submodule == geoh5py.objects.ObjectBase

    assert len(recwarn) == 0


def test_warn_module_not_found():
    # pylint: disable=import-error
    # pylint: disable=no-name-in-module

    def noop(_):
        return None

    with pytest.warns(match="Module 'nonexisting' is missing from the environment."):
        with warn_module_not_found():
            import nonexisting as test_import
    with pytest.raises(NameError):
        noop(test_import)

    with pytest.warns(match="Module 'nonexisting' is missing from the environment."):
        with warn_module_not_found():
            from nonexisting import nope as test_import_from
    with pytest.raises(NameError):
        noop(test_import_from)

    with pytest.warns(match="Module 'os.nonexisting' is missing from the environment."):
        with warn_module_not_found():
            import os.nonexisting as test_import_os_submodule
    with pytest.raises(NameError):
        noop(test_import_os_submodule)

    with pytest.warns(match="Module 'os.nonexisting' is missing from the environment."):
        with warn_module_not_found():
            from os.nonexisting import nope as test_import_from_os_submodule
    with pytest.raises(NameError):
        noop(test_import_from_os_submodule)

    with pytest.warns(match="Module 'nonexisting' is missing from the environment."):
        with warn_module_not_found():
            import nonexisting.nope as test_import_nonexising_submodule
    with pytest.raises(NameError):
        noop(test_import_nonexising_submodule)

    with pytest.warns(match="Module 'nonexisting' is missing from the environment."):
        with warn_module_not_found():
            from nonexisting.nope import nada as test_import_from_nonexisting_submodule
    with pytest.raises(NameError):
        noop(test_import_from_nonexisting_submodule)


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

    grid.add_data({"Iteration_10_data": {"values": np.ones(10 * 15)}})
    grid.add_data({"Iteration_1_data": {"values": np.ones(10 * 15)}})
    grid.add_data({"Iteration_5_data": {"values": np.ones(10 * 15)}})
    grid.add_data({"Iteration_3_data": {"values": np.ones(10 * 15)}})
    grid.add_data({"Iteration_2_data": {"values": np.ones(10 * 15)}})
    grid.add_data({"Iteration_4_data": {"values": np.ones(10 * 15)}})
    grid.add_data({"Iteration_9.0_data": {"values": np.ones(10 * 15)}})
    grid.add_data({"Iteration_8e0_data": {"values": np.ones(10 * 15)}})
    grid.add_data({"Iteration_11_data": {"values": np.ones(10 * 15)}})
    grid.add_data({"Iteration_6_data": {"values": np.ones(10 * 15)}})
    grid.add_data({"Iteration_7_data": {"values": np.ones(10 * 15)}})
    grid.add_data({"interp_02": {"values": np.ones(10 * 15)}})
    grid.add_data({"interp_01": {"values": np.ones(10 * 15)}})
    grid.add_data({"interp_11": {"values": np.ones(10 * 15)}})
    grid.add_data({"iteration_2_model": {"values": np.ones(10 * 15)}})
    grid.add_data({"iteration_12_model": {"values": np.ones(10 * 15)}})
    grid.add_data({"Iteration_2_model": {"values": np.ones(10 * 15)}})
    grid.add_data({"Iteration_12_model": {"values": np.ones(10 * 15)}})
    grid.add_data({"topo": {"values": np.ones(10 * 15)}})
    grid.add_data({"uncert": {"values": np.ones(10 * 15)}})

    d = sorted_children_dict(grid)
    d = list(d.keys())
    assert d[0] == "Iteration_1_data"
    assert d[1] == "Iteration_2_data"
    assert d[7] == "Iteration_8e0_data"
    assert d[8] == "Iteration_9.0_data"
    assert d[-2] == "topo"
    assert d[-1] == "uncert"


def test_rotation_xyz():
    vec = np.c_[1, 0, 0]
    rot_vec = rotate_xyz(vec, [0, 0], 45)

    assert (
        np.linalg.norm(np.cross(rot_vec, [0.7071, 0.7071, 0])) < 1e-8
    ), "Error on positive rotation about origin."

    rot_vec = rotate_xyz(vec, [1, 1], -90)

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

    geotest = Geoh5Tester(geoh5, tmp_path, "test.geoh5")
    with geotest.make() as workspace:
        mesh = TreeMesh([[10] * 16, [10] * 4, [10] * 8], [0, 0, 0])
        mesh.insert_cells([10, 10, 10], mesh.max_level, finalize=True)
        omesh = treemesh_2_octree(workspace, mesh, name="test_mesh")
        assert omesh.n_cells == mesh.n_cells
        assert np.all(
            (omesh.centroids - mesh.cell_centers[getattr(mesh, "_ubc_order")]) < 1e-14
        )
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


def test_drape_2_tensormesh(tmp_path):
    ws = Workspace(os.path.join(tmp_path, "test.geoh5"))
    x = np.linspace(358600, 359500, 10)
    y = np.linspace(5885500, 5884600, 10)
    z = 300 * np.ones_like(x)
    locs = np.c_[x, y, z]
    h = [20, 40]
    depth_core = 200
    pads = [500] * 4
    expfact = 1.1
    drape, tensor, _ = get_drape_model(  # pylint: disable=W0632
        ws,
        "test_drape",
        locs,
        h,
        depth_core,
        pads,
        expansion_factor=expfact,
        parent=None,
        return_colocated_mesh=True,
        return_sorting=True,
    )

    new_tensor = drape_2_tensor(drape)

    assert np.allclose(new_tensor.cell_centers, tensor.cell_centers)


def test_octree_2_treemesh(tmp_path):

    geotest = Geoh5Tester(geoh5, tmp_path, "test.geoh5")
    with geotest.make() as workspace:
        mesh = TreeMesh([[10] * 4, [10] * 4, [10] * 4], [0, 0, 0])
        mesh.insert_cells([5, 5, 5], mesh.max_level, finalize=True)
        omesh = treemesh_2_octree(workspace, mesh)
        for prod in itertools.product("uvw", repeat=3):
            omesh.origin = [0, 0, 0]
            for axis in "uvw":
                attr = axis + "_cell_size"
                setattr(omesh, attr, np.abs(getattr(omesh, attr)))
            for axis in np.unique(prod):
                attr = axis + "_cell_size"
                setattr(omesh, attr, -1 * getattr(omesh, attr))
                omesh.origin["xyz"["uvw".find(axis)]] = 40
            tmesh = octree_2_treemesh(omesh)
            assert np.all((tmesh.cell_centers - mesh.cell_centers) < 1e-14)


def test_window_xy():
    x, y = np.meshgrid(np.arange(11), np.arange(11))
    x = x.ravel()
    y = y.ravel()
    window = {
        "center": [5, 5],
        "size": [1, 1],
    }
    ind, x_win, y_win = window_xy(x, y, window)
    assert len(x_win) == 1
    assert len(y_win) == 1
    assert x_win[0] == 5
    assert y_win[0] == 5
    assert sum(ind) == 1

    window = {"center": [6, 2.5], "size": [3, 2]}
    ind, x_win, y_win = window_xy(x, y, window)
    assert [p in x_win for p in [5, 6, 7]]
    assert [p in [5, 6, 7] for p in x_win]
    assert [p in y_win for p in [3, 4]]
    assert [p in [3, 4] for p in y_win]


def test_downsample_xy():
    x_grid, y_grid = np.meshgrid(np.arange(11), np.arange(11))
    x = x_grid.ravel()
    y = y_grid.ravel()
    _, x_down, y_down = downsample_xy(x, y, 0)
    assert np.all(x == x_down)
    assert np.all(y == y_down)

    _, x_down, y_down = downsample_xy(x, y, 1)
    assert np.all(x[::2] == x_down)
    assert np.all(y[::2] == y_down)


def test_downsample_grid():

    # Test a simple grid equal spacing in x, y
    x_grid, y_grid = np.meshgrid(np.arange(11), np.arange(11))
    _, x_down, y_down = downsample_grid(x_grid, y_grid, 2)
    assert np.all(np.diff(y_down.reshape(6, 6), axis=0) == 2)
    assert np.all(np.diff(x_down.reshape(6, 6), axis=1) == 2)

    # Test a rotated grid equal spacing in u, v
    xy_rot = rotate_xyz(np.c_[x_grid.ravel(), y_grid.ravel()], [5, 5], 30)
    x_grid_rot = xy_rot[:, 0].reshape(11, 11)
    y_grid_rot = xy_rot[:, 1].reshape(11, 11)
    _, xd, yd = downsample_grid(x_grid_rot, y_grid_rot, 2)
    xy = rotate_xyz(np.c_[xd, yd], [5, 5], -30)
    xg_test = xy[:, 0].reshape(6, 6)
    yg_test = xy[:, 1].reshape(6, 6)
    np.testing.assert_allclose(np.diff(xg_test, axis=1), np.full((6, 5), 2))
    np.testing.assert_allclose(np.diff(yg_test, axis=0), np.full((5, 6), 2))

    # Test unequal spacing in x, y
    x_grid, y_grid = np.meshgrid(np.arange(11), np.linspace(0, 10, 21))
    _, x_down, y_down = downsample_grid(x_grid, y_grid, 2)
    x_grid_test = x_down.reshape(6, 6)
    y_grid_test = y_down.reshape(6, 6)
    np.testing.assert_allclose(np.diff(x_grid_test, axis=1), np.full((6, 5), 2))
    np.testing.assert_allclose(np.diff(y_grid_test, axis=0), np.full((5, 6), 2))


def test_filter_xy():
    x_grid, y_grid = np.meshgrid(np.arange(11), np.arange(11))
    xy_rot = rotate_xyz(np.c_[x_grid.ravel(), y_grid.ravel()], [5, 5], 30)
    x_grid_rot = xy_rot[:, 0].reshape(11, 11)
    y_grid_rot = xy_rot[:, 1].reshape(11, 11)
    window = {
        "center": [5, 5],
        "size": [9, 5],
    }
    # Test the windowing functionality
    w_mask = filter_xy(x_grid, y_grid, window=window)
    x_grid_test, y_grid_test = x_grid[w_mask].reshape(5, 9), y_grid[w_mask].reshape(
        5, 9
    )
    np.testing.assert_allclose(
        x_grid_test, np.meshgrid(np.arange(1, 10), np.arange(3, 8))[0]
    )
    np.testing.assert_allclose(
        y_grid_test, np.meshgrid(np.arange(1, 10), np.arange(3, 8))[1]
    )

    # Test the downsampling functionality
    ds_mask = filter_xy(x_grid, y_grid, distance=2)
    x_grid_test, y_grid_test = x_grid[ds_mask].reshape(6, 6), y_grid[ds_mask].reshape(
        6, 6
    )
    np.testing.assert_allclose(np.diff(x_grid_test, axis=1), np.full((6, 5), 2))
    np.testing.assert_allclose(np.diff(y_grid_test, axis=0), np.full((5, 6), 2))

    # Test the combo functionality
    comb_mask = filter_xy(x_grid, y_grid, distance=2, window=window)
    assert np.all(comb_mask == (w_mask & ds_mask))
    x_grid_test, y_grid_test = x_grid[comb_mask].reshape(2, 4), y_grid[
        comb_mask
    ].reshape(2, 4)
    assert np.all((x_grid_test >= 1) & (x_grid_test <= 9))
    assert np.all((y_grid_test >= 3) & (y_grid_test <= 7))
    np.testing.assert_allclose(np.diff(x_grid_test, axis=1), np.full((2, 3), 2))
    np.testing.assert_allclose(np.diff(y_grid_test, axis=0), np.full((1, 4), 2))

    # Test rotation options
    combo_mask = filter_xy(x_grid_rot, y_grid_rot, distance=2, window=window, angle=-30)
    xg_test, yg_test = x_grid_rot[comb_mask], y_grid_rot[comb_mask]
    xy_rot = rotate_xyz(np.c_[xg_test, yg_test], [5, 5], -30)
    x_grid_rot_test, y_grid_rot_test = xy_rot[:, 0].reshape(2, 4), xy_rot[:, 1].reshape(
        2, 4
    )
    assert np.all((x_grid_rot_test >= 1) & (x_grid_rot_test <= 9))
    assert np.all((y_grid_rot_test >= 3) & (y_grid_rot_test <= 7))
    np.testing.assert_allclose(np.diff(x_grid_rot_test, axis=1), np.full((2, 3), 2))
    np.testing.assert_allclose(np.diff(y_grid_rot_test, axis=0), np.full((1, 4), 2))

    window["azimuth"] = -30
    combo_mask_test = filter_xy(x_grid_rot, y_grid_rot, distance=2, window=window)
    assert np.all(combo_mask_test == combo_mask)


def test_detrend_xy():
    x_grid, y_grid = np.meshgrid(np.arange(64), np.arange(64))
    xy = np.c_[x_grid.flatten(), y_grid.flatten()]
    coefficients = np.random.randn(3)
    values = coefficients[0] + coefficients[1] * xy[:, 1] + coefficients[2] * xy[:, 0]
    ind_nan = np.random.randint(0, high=values.shape[0] - 1, size=32)
    nan_values = values.copy()
    nan_values[ind_nan] = np.nan

    # Should return a plane even for order=5
    comp_trend, _ = calculate_2D_trend(xy, nan_values, order=5, method="all")
    np.testing.assert_almost_equal(values, comp_trend)
    # Should return same plane parameter for 'perimeter' or 'all'
    corner_trend, _ = calculate_2D_trend(xy, nan_values, order=1, method="perimeter")
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


def test_get_locations(tmp_path):

    with Workspace(os.path.join(tmp_path, "test.geoh5")) as workspace:
        n_x, n_y = 10, 15
        grid = Grid2D.create(
            workspace,
            origin=[0, 0, 0],
            u_cell_size=20.0,
            v_cell_size=30.0,
            u_count=n_x,
            v_count=n_y,
            name="test_grid",
            allow_move=False,
        )
        base_locs = get_locations(workspace, grid)

        test_data = grid.add_data({"test_data": {"values": np.ones(10 * 15)}})
        data_locs = get_locations(workspace, test_data)

        np.testing.assert_array_equal(base_locs, data_locs)
