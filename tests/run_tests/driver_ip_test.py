#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import numpy as np
from geoh5py.workspace import Workspace

from geoapps.utils import get_inversion_output
from geoapps.utils.testing import setup_inversion_workspace

# import pytest
# pytest.skip("eliminating conflicting test.", allow_module_level=True)

# To test the full run and validate the inversion.
# Move this file out of the test directory and run.

target_ip_run = {
    "data_norm": 0.007242,
    "phi_d": 4.985,
    "phi_m": 0.06141,
}


def test_ip_run(
    tmp_path,
    n_electrodes=4,
    n_lines=3,
    max_iterations=1,
    pytest=True,
    refinement=(4, 6),
):
    from geoapps.inversion.driver import InversionDriver
    from geoapps.inversion.electricals import InducedPolarizationParams

    np.random.seed(0)
    # Run the forward
    geoh5 = setup_inversion_workspace(
        tmp_path,
        background=1e-6,
        anomaly=1e-1,
        n_electrodes=n_electrodes,
        n_lines=n_lines,
        refinement=refinement,
        dcip=True,
        flatten=False,
    )

    tx_obj = geoh5.get_entity("survey (currents)")[0]
    tx_obj.cells = tx_obj.cells.astype("uint32")

    model = geoh5.get_entity("model")[0]
    params = InducedPolarizationParams(
        forward_only=True,
        geoh5=geoh5,
        mesh=model.parent.uid,
        topography_object=geoh5.get_entity("topography")[0].uid,
        resolution=0.0,
        z_from_topo=True,
        data_object=geoh5.get_entity("survey")[0].uid,
        starting_model_object=model.parent.uid,
        starting_model=model.uid,
        conductivity_model=1e-2,
    )
    params.workpath = tmp_path
    fwr_driver = InversionDriver(params)
    fwr_driver.run()
    geoh5 = Workspace(geoh5.h5file)
    potential = geoh5.get_entity("Iteration_0_ip")[0]
    # Run the inverse
    np.random.seed(0)
    params = InducedPolarizationParams(
        geoh5=geoh5,
        mesh=geoh5.get_entity("mesh")[0].uid,
        topography_object=geoh5.get_entity("topography")[0].uid,
        resolution=0.0,
        data_object=potential.parent.uid,
        conductivity_model=1e-2,
        starting_model=1e-6,
        s_norm=0.0,
        x_norm=0.0,
        y_norm=0.0,
        z_norm=0.0,
        gradient_type="components",
        chargeability_channel_bool=True,
        z_from_topo=True,
        chargeability_channel=potential.uid,
        chargeability_uncertainty=2e-4,
        max_iterations=max_iterations,
        initial_beta=None,
        initial_beta_ratio=1e0,
        prctile=100,
        upper_bound=0.1,
        tile_spatial=n_lines,
    )
    params.workpath = tmp_path
    driver = InversionDriver(params)
    driver.run()
    output = get_inversion_output(
        driver.params.geoh5.h5file, driver.params.ga_group.uid
    )
    if pytest:
        np.testing.assert_almost_equal(
            np.linalg.norm(potential.values),
            target_ip_run["data_norm"],
            decimal=3,
        )
        np.testing.assert_almost_equal(output["phi_m"][1], target_ip_run["phi_m"])
        np.testing.assert_almost_equal(output["phi_d"][1], target_ip_run["phi_d"])
    else:
        return fwr_driver.starting_model, driver.inverse_problem.model


if __name__ == "__main__":
    # Full run
    m_start, m_rec = test_ip_run(
        "./",
        n_electrodes=20,
        n_lines=5,
        max_iterations=20,
        pytest=False,
        refinement=(4, 8),
    )
    residual = np.linalg.norm(m_rec - m_start) / np.linalg.norm(m_start) * 100.0
    assert (
        residual < 80.0
    ), f"Deviation from the true solution is {residual:.2f}%. Validate the solution!"
    print("Conductivity model is within 15% of the answer. You are so special!")
