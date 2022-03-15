#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import numpy as np
from geoh5py.workspace import Workspace
from SimPEG import utils

from geoapps.utils import get_inversion_output
from geoapps.utils.testing import setup_inversion_workspace

# import pytest
# pytest.skip("eliminating conflicting test.", allow_module_level=True)

# To test the full run and validate the inversion.
# Move this file out of the test directory and run.

target_susceptibility_run = {
    "data_norm": 11.707134,
    "phi_d": 1.556,
    "phi_m": 8.932e-6,
}


def test_susceptibility_run(
    tmp_path,
    n_grid_points=2,
    max_iterations=1,
    pytest=True,
    refinement=(2,),
):
    from geoapps.drivers.magnetic_scalar.magnetic_scalar_inversion import (
        MagneticScalarDriver,
    )
    from geoapps.drivers.magnetic_scalar.params import MagneticScalarParams

    np.random.seed(0)
    inducing_field = (50000.0, 90.0, 0.0)
    # Run the forward
    geoh5 = setup_inversion_workspace(
        tmp_path,
        background=0.0,
        anomaly=0.05,
        refinement=refinement,
        n_electrodes=n_grid_points,
        n_lines=n_grid_points,
        flatten=False,
    )
    model = geoh5.get_entity("model")[0]
    params = MagneticScalarParams(
        forward_only=True,
        geoh5=geoh5,
        mesh=model.parent.uid,
        topography_object=geoh5.get_entity("topography")[0].uid,
        inducing_field_strength=inducing_field[0],
        inducing_field_inclination=inducing_field[1],
        inducing_field_declination=inducing_field[2],
        resolution=0.0,
        z_from_topo=False,
        data_object=geoh5.get_entity("survey")[0].uid,
        starting_model_object=model.parent.uid,
        starting_model=model.uid,
    )
    params.workpath = tmp_path

    fwr_driver = MagneticScalarDriver(params)
    fwr_driver.run()
    geoh5 = Workspace(geoh5.h5file)
    tmi = geoh5.get_entity("Iteration_0_tmi")[0]
    # Run the inverse
    np.random.seed(0)
    params = MagneticScalarParams(
        geoh5=geoh5,
        mesh=geoh5.get_entity("mesh")[0].uid,
        topography_object=geoh5.get_entity("topography")[0].uid,
        inducing_field_strength=inducing_field[0],
        inducing_field_inclination=inducing_field[1],
        inducing_field_declination=inducing_field[2],
        resolution=0.0,
        data_object=tmi.parent.uid,
        starting_model=1e-4,
        s_norm=0.0,
        x_norm=0.0,
        y_norm=0.0,
        z_norm=0.0,
        gradient_type="components",
        lower_bound=0.0,
        tmi_channel_bool=True,
        z_from_topo=False,
        tmi_channel=tmi.uid,
        tmi_uncertainty=4.0,
        max_iterations=max_iterations,
        initial_beta_ratio=1e0,
    )
    params.workpath = tmp_path

    driver = MagneticScalarDriver(params)
    driver.run()
    run_ws = Workspace(driver.params.geoh5.h5file)
    output = get_inversion_output(
        driver.params.geoh5.h5file, driver.params.ga_group.uid
    )
    if pytest:
        np.testing.assert_almost_equal(
            np.linalg.norm(tmi.values),
            target_susceptibility_run["data_norm"],
            decimal=3,
        )
        np.testing.assert_almost_equal(
            output["phi_m"][1], target_susceptibility_run["phi_m"]
        )
        np.testing.assert_almost_equal(
            output["phi_d"][1], target_susceptibility_run["phi_d"]
        )

        nan_ind = np.isnan(run_ws.get_entity("Iteration_0_model")[0].values)
        inactive_ind = run_ws.get_entity("active_cells")[0].values == 0
        assert np.all(nan_ind == inactive_ind)
    else:
        return fwr_driver.starting_model, driver.inverse_problem.model


target_magnetic_vector_run = {
    "data_norm": 8.943476,
    "phi_d": 0.006804,
    "phi_m": 4.679e-6,
}


def test_magnetic_vector_run(
    tmp_path,
    n_grid_points=2,
    max_iterations=1,
    pytest=True,
    refinement=(2,),
):
    from geoapps.drivers.magnetic_vector.magnetic_vector_inversion import (
        MagneticVectorDriver,
    )
    from geoapps.drivers.magnetic_vector.params import MagneticVectorParams

    np.random.seed(0)
    inducing_field = (50000.0, 90.0, 0.0)
    # Run the forward
    geoh5 = setup_inversion_workspace(
        tmp_path,
        background=0.0,
        anomaly=0.05,
        refinement=refinement,
        n_electrodes=n_grid_points,
        n_lines=n_grid_points,
    )
    model = geoh5.get_entity("model")[0]
    params = MagneticVectorParams(
        forward_only=True,
        geoh5=geoh5,
        mesh=model.parent.uid,
        topography_object=geoh5.get_entity("topography")[0].uid,
        inducing_field_strength=inducing_field[0],
        inducing_field_inclination=inducing_field[1],
        inducing_field_declination=inducing_field[2],
        resolution=0.0,
        z_from_topo=False,
        data_object=geoh5.get_entity("survey")[0].uid,
        starting_model_object=model.parent.uid,
        starting_model=model.uid,
        starting_inclination=45,
        starting_declination=270,
    )
    fwr_driver = MagneticVectorDriver(params)
    fwr_driver.run()
    geoh5 = Workspace(geoh5.h5file)
    tmi = geoh5.get_entity("Iteration_0_tmi")[0]
    # Run the inverse
    params = MagneticVectorParams(
        geoh5=geoh5,
        mesh=geoh5.get_entity("mesh")[0].uid,
        topography_object=geoh5.get_entity("topography")[0].uid,
        inducing_field_strength=inducing_field[0],
        inducing_field_inclination=inducing_field[1],
        inducing_field_declination=inducing_field[2],
        resolution=0.0,
        data_object=tmi.parent.uid,
        starting_model=1e-4,
        s_norm=0.0,
        x_norm=1.0,
        y_norm=1.0,
        z_norm=1.0,
        gradient_type="components",
        tmi_channel_bool=True,
        z_from_topo=False,
        tmi_channel=tmi.uid,
        tmi_uncertainty=4.0,
        max_iterations=max_iterations,
        initial_beta_ratio=1e1,
        prctile=100,
    )
    driver = MagneticVectorDriver(params)
    driver.run()
    run_ws = Workspace(driver.params.geoh5.h5file)
    # Re-open the workspace and get iterations
    output = get_inversion_output(
        driver.params.geoh5.h5file, driver.params.ga_group.uid
    )
    if pytest:
        np.testing.assert_almost_equal(
            output["phi_m"][1], target_magnetic_vector_run["phi_m"]
        )
        np.testing.assert_almost_equal(
            output["phi_d"][1], target_magnetic_vector_run["phi_d"]
        )
        np.testing.assert_almost_equal(
            np.linalg.norm(tmi.values),
            target_magnetic_vector_run["data_norm"],
            decimal=3,
        )

        nan_ind = np.isnan(run_ws.get_entity("Iteration_0_amplitude_model")[0].values)
        inactive_ind = run_ws.get_entity("active_cells")[0].values == 0
        assert np.all(nan_ind == inactive_ind)
    else:
        return fwr_driver.starting_model, utils.spherical2cartesian(
            driver.inverse_problem.model
        )


if __name__ == "__main__":
    # Full run
    m_start, m_rec = test_susceptibility_run(
        "./", n_grid_points=20, max_iterations=30, pytest=False, refinement=(4, 8)
    )
    residual = np.linalg.norm(m_rec - m_start) / np.linalg.norm(m_start) * 100.0
    assert (
        residual < 15.0
    ), f"Deviation from the true solution is {residual:.2f}%. Validate the solution!"
    print("Susceptibility model is within 15% of the answer. Well done you!")
    m_start, m_rec = test_magnetic_vector_run(
        "./", n_grid_points=20, max_iterations=30, pytest=False, refinement=(4, 8)
    )
    residual = np.linalg.norm(m_rec - m_start) / np.linalg.norm(m_start) * 100.0
    assert (
        residual < 50.0
    ), f"Deviation from the true solution is {residual:.2f}%. Validate the solution!"
    print("MVI model is within 50% of the answer. Done!")
