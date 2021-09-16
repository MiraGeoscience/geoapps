#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from os import path

import numpy as np
from discretize.utils import active_from_xyz, mesh_builder_xyz, refine_tree_xyz
from geoh5py.objects import Points, Surface
from geoh5py.workspace import Workspace
from scipy.spatial import Delaunay
from SimPEG import utils

from geoapps.utils import get_inversion_output, treemesh_2_octree

n_grid_points = 20  # Full run: 20
h = 5.0  # Full run: 5.0
max_iterations = 30  # Full run: 30


def setup_workspace(work_dir, phys_prop):
    project = path.join(work_dir, "mag_test.geoh5")
    workspace = Workspace(project)

    # Topography
    xx, yy = np.meshgrid(np.linspace(-200.0, 200.0, 50), np.linspace(-200.0, 200.0, 50))
    b = 100
    A = 50
    zz = A * np.exp(-0.5 * ((xx / b) ** 2.0 + (yy / b) ** 2.0))
    topo = np.c_[utils.mkvc(xx), utils.mkvc(yy), utils.mkvc(zz)]
    triang = Delaunay(topo[:, :2])

    surf = Surface.create(
        workspace, vertices=topo, cells=triang.simplices, name="topography"
    )

    # Observation points
    xr = np.linspace(-100.0, 100.0, n_grid_points)
    yr = np.linspace(-100.0, 100.0, n_grid_points)
    X, Y = np.meshgrid(xr, yr)
    Z = A * np.exp(-0.5 * ((X / b) ** 2.0 + (Y / b) ** 2.0)) + 5.0
    points = Points.create(
        workspace,
        vertices=np.c_[utils.mkvc(X.T), utils.mkvc(Y.T), utils.mkvc(Z.T)],
        name="survey",
    )

    # Create a mesh
    padDist = np.ones((3, 2)) * 100
    nCpad = [4, 4, 2]
    mesh = mesh_builder_xyz(
        points.vertices,
        [h] * 3,
        padding_distance=padDist,
        mesh_type="TREE",
    )
    mesh = refine_tree_xyz(
        mesh,
        topo,
        method="surface",
        octree_levels=nCpad,
        octree_levels_padding=nCpad,
        finalize=True,
    )
    octree = treemesh_2_octree(workspace, mesh, name="mesh")
    active = active_from_xyz(mesh, surf.vertices, grid_reference="N")

    # Model
    model = utils.model_builder.addBlock(
        mesh.gridCC,
        np.zeros(mesh.nC),
        np.r_[-20, -20, -15],
        np.r_[20, 20, 20],
        phys_prop,
    )
    model[~active] = np.nan
    octree.add_data({"model": {"values": model[mesh._ubc_order]}})
    octree.copy()  # Keep a copy around for ref

    return workspace


def test_susceptibility_run(tmp_path):
    from geoapps.drivers.magnetic_scalar_inversion import MagneticScalarDriver
    from geoapps.io.MagneticScalar.params import MagneticScalarParams

    # Inducing field
    H0 = (50000.0, 90.0, 0.0)
    # Values stored from pre-runs
    target = {
        "data_norm": 716.999,
        "phi_m": 0.006225,
        "phi_d": 92500.0,
    }
    # Run the forward
    workspace = setup_workspace(tmp_path, 0.05)
    model = workspace.get_entity("model")[0]
    params = MagneticScalarParams(
        forward_only=True,
        workspace=workspace,
        mesh=model.parent,
        topography_object=workspace.get_entity("topography")[0],
        inducing_field_strength=H0[0],
        inducing_field_inclination=H0[1],
        inducing_field_declination=H0[2],
        resolution=0.0,
        data_object=workspace.get_entity("survey")[0],
        starting_model_object=model.parent,
        starting_model=model,
    )
    fwr_driver = MagneticScalarDriver(params)
    fwr_driver.run()

    workspace = Workspace(workspace.h5file)
    tmi = workspace.get_entity("tmi")[0]
    np.testing.assert_almost_equal(
        np.linalg.norm(tmi.values), target["data_norm"], decimal=3
    )

    # Run the inverse
    np.random.seed(0)
    params = MagneticScalarParams(
        workspace=workspace,
        mesh=workspace.get_entity("mesh")[0],
        topography_object=workspace.get_entity("topography")[0],
        inducing_field_strength=H0[0],
        inducing_field_inclination=H0[1],
        inducing_field_declination=H0[2],
        resolution=0.0,
        data_object=tmi.parent,
        starting_model=1e-4,
        s_norm=0.0,
        x_norm=0.0,
        y_norm=0.0,
        z_norm=0.0,
        gradient_type="components",
        lower_bound=0.0,
        tmi_channel_bool=True,
        z_from_topo=False,
        tmi_channel=tmi,
        tmi_uncertainty=1.0,
        max_iterations=max_iterations,
    )
    driver = MagneticScalarDriver(params)
    driver.run()

    # Re-open the workspace and get iterations
    output = get_inversion_output(
        driver.params.workspace.h5file, driver.params.out_group.uid
    )
    # np.testing.assert_almost_equal(output["phi_m"][1], target["phi_m"])
    # np.testing.assert_almost_equal(output["phi_d"][1], target["phi_d"])

    ##########################################################################
    # Solution should satisfy this condition if run to completion.
    # To get validation with full run set:
    # n_grid_points = 20
    # h =  5
    # max_iterations = 30
    # residual = (
    #         np.linalg.norm(driver.inverse_problem.model - fwr_driver.starting_model) /
    #         np.linalg.norm(fwr_driver.starting_model) * 100.
    # )
    # assert residual < 2.0, (
    #     f"Deviation from the true solution is {residual}%. Please revise."
    # )


def test_magnetic_vector_run(tmp_path):
    from geoapps.drivers.magnetic_vector_inversion import MagneticVectorDriver
    from geoapps.io.MagneticVector.params import MagneticVectorParams

    np.random.seed(0)

    # Inducing field
    H0 = (50000.0, 90.0, 0.0)
    # Values stored from pre-runs
    target = {
        "data_norm": 716.999,
        "phi_m": 0.0007209,
        "phi_d": 147600.0,
    }
    # Run the forward
    workspace = setup_workspace(tmp_path, 0.05)
    model = workspace.get_entity("model")[0]
    params = MagneticVectorParams(
        forward_only=True,
        workspace=workspace,
        mesh=model.parent,
        topography_object=workspace.get_entity("topography")[0],
        inducing_field_strength=H0[0],
        inducing_field_inclination=H0[1],
        inducing_field_declination=H0[2],
        resolution=0.0,
        data_object=workspace.get_entity("survey")[0],
        starting_model_object=model.parent,
        starting_model=model,
    )
    driver = MagneticVectorDriver(params)
    driver.run()

    workspace = Workspace(workspace.h5file)
    tmi = workspace.get_entity("tmi")[0]
    np.testing.assert_almost_equal(
        np.linalg.norm(tmi.values), target["data_norm"], decimal=3
    )

    # Run the inverse
    params = MagneticVectorParams(
        workspace=workspace,
        mesh=workspace.get_entity("mesh")[0],
        topography_object=workspace.get_entity("topography")[0],
        inducing_field_strength=H0[0],
        inducing_field_inclination=H0[1],
        inducing_field_declination=H0[2],
        resolution=0.0,
        data_object=tmi.parent,
        starting_model=1e-4,
        s_norm=0.0,
        x_norm=0.0,
        y_norm=0.0,
        z_norm=0.0,
        gradient_type="components",
        tmi_channel_bool=True,
        z_from_topo=False,
        tmi_channel=tmi,
        tmi_uncertainty=1.0,
        max_iterations=max_iterations,
    )
    driver = MagneticVectorDriver(params)
    driver.run()

    # Re-open the workspace and get iterations
    output = get_inversion_output(
        driver.params.workspace.h5file, driver.params.out_group.uid
    )
    # np.testing.assert_almost_equal(output["phi_m"][1], target["phi_m"])
    # np.testing.assert_almost_equal(output["phi_d"][1], target["phi_d"])

    ##########################################################################
    # Solution should satisfy this condition if run to completion.
    # To get validated with full run if values in 'target' need to be updated.
    #
    residual = (
        np.linalg.norm(driver.inverse_problem.model - model)
        / np.linalg.norm(model)
        * 100.0
    )
    assert (
        residual < 0.1
    ), f"Deviation from the true solution is {residual}%. Please revise."
