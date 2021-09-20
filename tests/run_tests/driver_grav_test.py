#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import numpy as np
from geoh5py.workspace import Workspace

from geoapps.utils import get_inversion_output
from geoapps.utils.testing import setup_inversion_workspace

# To test the full run and validate the inversion.
# Move this file out of the test directory and run.

target_gravity_run = {
    "data_norm": 7.33970,
    "phi_d": 0.6644,
    "phi_m": 2.112e-5,
}


def test_gravity_run(
    tmp_path,
    n_grid_points=2,
    max_iterations=1,
    pytest=True,
    refinement=(2,),
):
    from geoapps.drivers.grav_inversion import GravityDriver
    from geoapps.io.Gravity.params import GravityParams

    np.random.seed(0)
    # Run the forward
    workspace = setup_inversion_workspace(
        tmp_path, 0.25, n_grid_points=n_grid_points, refinement=refinement
    )
    model = workspace.get_entity("model")[0]
    params = GravityParams(
        forward_only=True,
        workspace=workspace,
        mesh=model.parent,
        topography_object=workspace.get_entity("topography")[0],
        resolution=0.0,
        z_from_topo=False,
        data_object=workspace.get_entity("survey")[0],
        starting_model_object=model.parent,
        starting_model=model,
    )
    fwr_driver = GravityDriver(params)
    fwr_driver.run()
    workspace = Workspace(workspace.h5file)
    gz = workspace.get_entity("gz")[0]
    # Run the inverse
    np.random.seed(0)
    params = GravityParams(
        workspace=workspace,
        mesh=workspace.get_entity("mesh")[0],
        topography_object=workspace.get_entity("topography")[0],
        resolution=0.0,
        data_object=gz.parent,
        starting_model=1e-4,
        s_norm=0.0,
        x_norm=0.0,
        y_norm=0.0,
        z_norm=0.0,
        gradient_type="components",
        gz_channel_bool=True,
        z_from_topo=False,
        gz_channel=gz,
        gz_uncertainty=5e-4,
        max_iterations=max_iterations,
        initial_beta_ratio=1e0,
    )
    driver = GravityDriver(params)
    driver.run()
    output = get_inversion_output(
        driver.params.workspace.h5file, driver.params.out_group.uid
    )
    if pytest:
        np.testing.assert_almost_equal(
            np.linalg.norm(gz.values),
            target_gravity_run["data_norm"],
            decimal=3,
        )
        np.testing.assert_almost_equal(output["phi_m"][1], target_gravity_run["phi_m"])
        np.testing.assert_almost_equal(output["phi_d"][1], target_gravity_run["phi_d"])
    else:
        return fwr_driver.starting_model, driver.inverse_problem.model


if __name__ == "__main__":
    # Full run
    m_start, m_rec = test_gravity_run(
        "./", n_grid_points=20, max_iterations=30, pytest=False, refinement=(4, 6)
    )
    residual = np.linalg.norm(m_rec - m_start) / np.linalg.norm(m_start) * 100.0
    assert (
        residual < 15.0
    ), f"Deviation from the true solution is {residual:.2f}%. Validate the solution!"
    print("Susceptibility model is within 15% of the answer. Well done you!")
