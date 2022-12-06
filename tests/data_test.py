#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import os

import numpy as np
import SimPEG
from discretize.utils import mesh_builder_xyz, refine_tree_xyz
from geoh5py.objects import Points
from geoh5py.workspace import Workspace

from geoapps.driver_base.utils import treemesh_2_octree
from geoapps.inversion.components import InversionData
from geoapps.inversion.driver import InversionDriver
from geoapps.inversion.potential_fields import MagneticVectorParams
from geoapps.utils.testing import Geoh5Tester

geoh5 = Workspace("./FlinFlon.geoh5")


def setup_params(tmp):
    geotest = Geoh5Tester(geoh5, tmp, "test.geoh5", params_class=MagneticVectorParams)
    geotest.set_param("mesh", "{a8f3b369-10bd-4ca8-8bd6-2d2595bddbdf}")
    geotest.set_param("data_object", "{538a7eb1-2218-4bec-98cc-0a759aa0ef4f}")
    geotest.set_param("topography_object", "{ab3c2083-6ea8-4d31-9230-7aad3ec09525}")
    geotest.set_param("tmi_channel", "{44822654-b6ae-45b0-8886-2d845f80f422}")
    geotest.set_param("gyz_channel", "{3d19bd53-8bb8-4634-aeae-4e3a90e9d19e}")
    geotest.set_param("topography", "{a603a762-f6cb-4b21-afda-3160e725bf7d}")
    geotest.set_param("out_group", "MVIInversion")
    return geotest.make()


def test_survey_data(tmp_path):
    X, Y, Z = np.meshgrid(np.linspace(0, 100, 3), np.linspace(0, 100, 3), 0)
    verts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    with Workspace(os.path.join(tmp_path, "test_workspace.geoh5")) as workspace:
        test_data_object = Points.create(
            workspace, vertices=verts, name="test_data_object"
        )
        bxx_data, byy_data, bzz_data = test_data_object.add_data(
            {
                "bxx": {
                    "association": "VERTEX",
                    "values": np.arange(len(verts)).astype(float),
                },
                "byy": {
                    "association": "VERTEX",
                    "values": len(verts) + np.arange(len(verts)).astype(float),
                },
                "bzz": {
                    "association": "VERTEX",
                    "values": 2 * len(verts) + np.arange(len(verts)).astype(float),
                },
            }
        )
        test_topo_object = Points.create(
            workspace, vertices=verts, name="test_topo_object"
        )
        _ = test_topo_object.add_data(
            {"elev": {"association": "VERTEX", "values": 100 * np.ones(len(verts))}}
        )
        topo = workspace.get_entity("elev")[0]
        mesh = mesh_builder_xyz(
            verts,
            [20, 20, 20],
            depth_core=50,
            mesh_type="TREE",
        )
        mesh = refine_tree_xyz(
            mesh,
            test_topo_object.vertices,
            method="surface",
            finalize=True,
        )

        mesh = treemesh_2_octree(workspace, mesh)
        params = MagneticVectorParams(
            forward_only=False,
            geoh5=workspace,
            data_object=test_data_object.uid,
            topography_object=test_topo_object.uid,
            topography=topo,
            bxx_channel=bxx_data.uid,
            bxx_uncertainty=0.1,
            byy_channel=byy_data.uid,
            byy_uncertainty=0.2,
            bzz_channel=bzz_data.uid,
            bzz_uncertainty=0.3,
            mesh=mesh.uid,
            starting_model=0.0,
            tile_spatial=2,
            z_from_topo=True,
            receivers_offset_z=50.0,
            resolution=0.0,
        )

        driver = InversionDriver(params, warmstart=False)

    local_survey_a = driver.inverse_problem.dmisfit.objfcts[0].simulation.survey
    local_survey_b = driver.inverse_problem.dmisfit.objfcts[1].simulation.survey

    # test locations

    np.testing.assert_array_equal(
        verts[driver.sorting[0], :2], local_survey_a.receiver_locations[:, :2]
    )
    np.testing.assert_array_equal(
        verts[driver.sorting[1], :2], local_survey_b.receiver_locations[:, :2]
    )
    assert all(
        local_survey_a.receiver_locations[:, 2] == 150.0
    )  # 150 = 100 (z_from_topo) + 50 (receivers_offset_z)
    assert all(local_survey_b.receiver_locations[:, 2] == 150.0)

    # test observed data
    sorting = np.hstack(driver.sorting)
    expected_dobs = np.column_stack(
        [bxx_data.values, byy_data.values, bzz_data.values]
    )[sorting].ravel()
    survey_dobs = [local_survey_a.dobs, local_survey_b.dobs]
    np.testing.assert_array_equal(expected_dobs, np.hstack(survey_dobs))

    # test savegeoh5iteration data
    driver.directive_list[-2].save_components(99, survey_dobs)

    with workspace.open():
        bxx_test = workspace.get_entity("Iteration_99_bxx")[0].values
        byy_test = workspace.get_entity("Iteration_99_byy")[0].values
        bzz_test = workspace.get_entity("Iteration_99_bzz")[0].values

    np.testing.assert_array_equal(bxx_test, bxx_data.values)
    np.testing.assert_array_equal(byy_test, byy_data.values)
    np.testing.assert_array_equal(bzz_test, bzz_data.values)

    driver.directive_list[-1].save_components(99, survey_dobs)

    with workspace.open():
        assert np.all(
            workspace.get_entity("Iteration_99_bxx_Residual")[0].values == 0
        ), "Residual data should be zero."
        assert np.all(
            workspace.get_entity("Iteration_99_byy_Residual")[0].values == 0
        ), "Residual data should be zero."
        assert np.all(
            workspace.get_entity("Iteration_99_bzz_Residual")[0].values == 0
        ), "Residual data should be zero."


def test_save_data(tmp_path):
    ws, params = setup_params(tmp_path)
    locs = params.data_object.centroids
    window = {"center": [np.mean(locs[:, 0]), np.mean(locs[:, 1])], "size": [100, 100]}
    data = InversionData(ws, params, window)

    assert len(data.entity.vertices) > 0


def test_has_tensor():
    assert InversionData.check_tensor(["Gxx"])
    assert InversionData.check_tensor(["Gxy"])
    assert InversionData.check_tensor(["Gxz"])
    assert InversionData.check_tensor(["Gyy"])
    assert InversionData.check_tensor(["Gyx"])
    assert InversionData.check_tensor(["Gyz"])
    assert InversionData.check_tensor(["Gzz"])
    assert InversionData.check_tensor(["Gzx"])
    assert InversionData.check_tensor(["Gzy"])
    assert InversionData.check_tensor(["Gxx", "Gyy", "tmi"])
    assert not InversionData.check_tensor(["tmi"])


def test_get_uncertainty_component(tmp_path):
    ws, params = setup_params(tmp_path)
    locs = params.data_object.centroids
    window = {"center": [np.mean(locs[:, 0]), np.mean(locs[:, 1])], "size": [100, 100]}
    params.tmi_uncertainty = 1.0
    data = InversionData(ws, params, window)
    unc = data.get_data()[2]["tmi"]
    assert len(np.unique(unc)) == 1
    assert np.unique(unc)[0] == 1
    assert len(unc) == len(data.mask)


def test_parse_ignore_values(tmp_path):
    ws, params = setup_params(tmp_path)
    locs = params.data_object.centroids
    window = {"center": [np.mean(locs[:, 0]), np.mean(locs[:, 1])], "size": [100, 100]}
    params.ignore_values = "<99"
    data = InversionData(ws, params, window)
    val, logic = data.parse_ignore_values()
    assert val == 99
    assert logic == "<"

    params.ignore_values = ">99"
    data = InversionData(ws, params, window)
    val, logic = data.parse_ignore_values()
    assert val == 99
    assert logic == ">"

    params.ignore_values = "99"
    data = InversionData(ws, params, window)
    val, logic = data.parse_ignore_values()
    assert val == 99
    assert logic == "="


def test_set_infinity_uncertainties(tmp_path):
    ws, params = setup_params(tmp_path)
    locs = params.data_object.centroids
    window = {"center": [np.mean(locs[:, 0]), np.mean(locs[:, 1])], "size": [100, 100]}
    data = InversionData(ws, params, window)
    test_data = np.array([0, 1, 2, 3, 4, 5])
    test_unc = np.array([0.1] * 6)
    data.ignore_value = 3
    data.ignore_type = "="
    unc = data.set_infinity_uncertainties(test_unc, test_data)
    where_inf = np.where(np.isinf(unc))[0]
    assert len(where_inf) == 1
    assert where_inf == 3

    data.ignore_value = 3
    data.ignore_type = "<"
    unc = data.set_infinity_uncertainties(test_unc, test_data)
    where_inf = np.where(np.isinf(unc))[0]
    assert len(where_inf) == 4
    assert np.all(where_inf == [0, 1, 2, 3])

    data.ignore_value = 3
    data.ignore_type = ">"
    unc = data.set_infinity_uncertainties(test_unc, test_data)
    where_inf = np.where(np.isinf(unc))[0]
    assert len(where_inf) == 3
    assert np.all(where_inf == [3, 4, 5])

    data.ignore_value = None
    data.ignore_type = None
    unc = data.set_infinity_uncertainties(test_unc, test_data)
    assert np.all(test_unc == unc)


def test_displace(tmp_path):
    ws, params = setup_params(tmp_path)
    locs = params.data_object.centroids
    window = {"center": [np.mean(locs[:, 0]), np.mean(locs[:, 1])], "size": [100, 100]}
    data = InversionData(ws, params, window)
    test_locs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    test_offset = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    expected_locs = np.array([[2.0, 2.0, 3.0], [5.0, 5.0, 6.0], [8.0, 8.0, 9.0]])
    displaced_locs = data.displace(test_locs, test_offset)
    assert np.all(displaced_locs == expected_locs)

    test_offset = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    expected_locs = np.array([[1.0, 3.0, 3.0], [4.0, 6.0, 6.0], [7.0, 9.0, 9.0]])
    displaced_locs = data.displace(test_locs, test_offset)
    assert np.all(displaced_locs == expected_locs)

    test_offset = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    expected_locs = np.array([[1.0, 2.0, 4.0], [4.0, 5.0, 7.0], [7.0, 8.0, 10.0]])
    displaced_locs = data.displace(test_locs, test_offset)
    assert np.all(displaced_locs == expected_locs)


def test_drape(tmp_path):
    ws, params = setup_params(tmp_path)
    locs = params.data_object.centroids
    window = {"center": [np.mean(locs[:, 0]), np.mean(locs[:, 1])], "size": [100, 100]}
    data = InversionData(ws, params, window)
    test_locs = np.array([[1.0, 2.0, 1.0], [2.0, 1.0, 1.0], [8.0, 9.0, 1.0]])
    radar_ch = np.array([1.0, 2.0, 3.0])
    expected_locs = np.array([[1.0, 2.0, 2.0], [2.0, 1.0, 3.0], [8.0, 9.0, 4.0]])
    draped_locs = data.drape(test_locs, radar_ch)

    assert np.all(draped_locs == expected_locs)


def test_normalize(tmp_path):
    ws, params = setup_params(tmp_path)
    locs = params.data_object.centroids
    window = {"center": [np.mean(locs[:, 0]), np.mean(locs[:, 1])], "size": [100, 100]}
    data = InversionData(ws, params, window)
    data.observed = {"tmi": np.array([1.0, 2.0, 3.0]), "gz": np.array([1.0, 2.0, 3.0])}
    data.components = list(data.observed.keys())
    data.normalizations = data.get_normalizations()
    test_data = data.normalize(data.observed)
    assert list(data.normalizations.values()) == [1, -1]
    assert all(test_data["gz"] == (-1 * data.observed["gz"]))


def test_get_survey(tmp_path):
    ws, params = setup_params(tmp_path)
    locs = params.data_object.centroids
    window = {"center": [np.mean(locs[:, 0]), np.mean(locs[:, 1])], "size": [100, 100]}
    data = InversionData(ws, params, window)
    survey, _ = data.create_survey()
    assert isinstance(survey, SimPEG.potential_fields.magnetics.Survey)
