#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from os import path

import numpy as np
from dask.distributed import Client, LocalCluster, get_client
from discretize.utils import active_from_xyz, mesh_builder_xyz, refine_tree_xyz
from geoh5py.objects import Points, Surface
from geoh5py.workspace import Workspace
from scipy.spatial import Delaunay
from SimPEG import maps, utils
from SimPEG.potential_fields import magnetics as mag

from geoapps.drivers.magnetic_scalar_inversion import MagneticScalarDriver
from geoapps.io.MagneticScalar.params import MagneticScalarParams
from geoapps.utils import get_inversion_output, treemesh_2_octree

target = {
    "phi_m": 0.0226,
    "phi_d": 84410.0,
}


def setup_driver(tmp_path, max_iterations=1):
    project = path.join(tmp_path, "mag_test.geoh5")
    workspace = Workspace(project)

    try:
        get_client()
    except ValueError:
        cluster = LocalCluster(processes=False)
        Client(cluster)

    np.random.seed(0)

    # Inducing field
    H0 = (50000.0, 90.0, 0.0)

    # Topography
    xx, yy = np.meshgrid(np.linspace(-200.0, 200.0, 50), np.linspace(-200.0, 200.0, 50))
    b = 100
    A = 50
    zz = A * np.exp(-0.5 * ((xx / b) ** 2.0 + (yy / b) ** 2.0))
    topo = np.c_[utils.mkvc(xx), utils.mkvc(yy), utils.mkvc(zz)]
    triang = Delaunay(topo[:, :2])

    topo_surf = Surface.create(
        workspace, vertices=topo, cells=triang.simplices, name="topography"
    )

    # Observation points
    xr = np.linspace(-100.0, 100.0, 20)
    yr = np.linspace(-100.0, 100.0, 20)
    X, Y = np.meshgrid(xr, yr)
    Z = A * np.exp(-0.5 * ((X / b) ** 2.0 + (Y / b) ** 2.0)) + 5.0
    xyzLoc = np.c_[utils.mkvc(X.T), utils.mkvc(Y.T), utils.mkvc(Z.T)]

    # Create a mag survey
    rxLoc = mag.Point(xyzLoc)
    srcField = mag.SourceField([rxLoc], parameters=H0)
    survey = mag.Survey(srcField)

    # Create a mesh
    h = [5.0, 5.0, 5.0]
    padDist = np.ones((3, 2)) * 100
    nCpad = [2, 4, 2]
    mesh = mesh_builder_xyz(
        xyzLoc,
        h,
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
    actv = active_from_xyz(mesh, topo, grid_reference="N")
    nC = int(actv.sum())

    # Model
    model = utils.model_builder.addBlock(
        mesh.gridCC,
        np.zeros(mesh.nC),
        np.r_[-20, -20, -15],
        np.r_[20, 20, 20],
        0.05,
    )[actv]

    # Simulation
    actvMap = maps.InjectActiveCells(mesh, actv, np.nan)
    idenMap = maps.IdentityMap(nP=nC)
    sim = mag.Simulation3DIntegral(
        mesh,
        survey=survey,
        chiMap=idenMap,
        actInd=actv,
        store_sensitivities="ram",
    )
    octree.add_data({"model": {"values": (actvMap * model)[mesh._ubc_order]}})
    octree.copy()  # Keep a copy around for ref
    data = sim.make_synthetic_data(
        model, relative_error=0.0, noise_floor=1.0, add_noise=True
    )
    # Output to geoh5
    point_survey = Points.create(workspace, vertices=xyzLoc, name="survey")
    tmi, uncerts = point_survey.add_data(
        {"observed": {"values": data.dobs}, "std": {"values": data.standard_deviation}}
    )
    params = MagneticScalarParams(
        workspace=workspace,
        mesh=octree,
        topography_object=topo_surf,
        inducing_field_strength=H0[0],
        inducing_field_inclination=H0[1],
        inducing_field_declination=H0[2],
        data_object=point_survey,
        starting_model=1e-4,
        resolution=0.0,
        s_norm=0.0,
        x_norm=0.0,
        y_norm=0.0,
        z_norm=0.0,
        gradient_type="components",
        lower_bound=0.0,
        tmi_channel_bool=True,
        z_from_topo=False,
        tmi_channel=tmi,
        tmi_uncertainty=uncerts,
        max_iterations=max_iterations,
    )
    return MagneticScalarDriver(params)


def test_susceptibility_run(tmp_path):
    driver = setup_driver(tmp_path, max_iterations=1)
    driver.run()

    # Re-open the workspace and get iterations
    output = get_inversion_output(
        driver.params.workspace.h5file, driver.params.out_group.uid
    )
    np.testing.assert_almost_equal(output["phi_m"][-1], target["phi_m"])
    np.testing.assert_almost_equal(output["phi_d"][-1], target["phi_d"])

    ##############################################################
    # Solution should satisfy this condition if run to completion.
    # To be validated if the runtest above needs to be updated.
    #
    # residual = (
    #         np.linalg.norm(driver.inverse_problem.model - model) /
    #         np.linalg.norm(model) * 100.
    # )
    # assert residual < 0.1, (
    #     f"Deviation from the true solution is {residual}%. Please revise."
    # )
