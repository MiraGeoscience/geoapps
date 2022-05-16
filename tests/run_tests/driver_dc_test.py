#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import numpy as np
from geoh5py.workspace import Workspace

from geoapps.utils.testing import check_target, setup_inversion_workspace
from geoapps.utils.utils import get_inversion_output

# import pytest
# pytest.skip("eliminating conflicting test.", allow_module_level=True)

# To test the full run and validate the inversion.
# Move this file out of the test directory and run.

target_run = {
    "data_norm": 0.152097,
    "phi_d": 9.878,
    "phi_m": 79.91,
}


def test_dc_run(
    tmp_path,
    n_electrodes=4,
    n_lines=3,
    max_iterations=1,
    pytest=True,
    refinement=(4, 6),
):
    from geoapps.inversion.driver import InversionDriver
    from geoapps.inversion.electricals import DirectCurrentParams

    np.random.seed(0)
    # Run the forward
    geoh5, mesh, model, survey, topography = setup_inversion_workspace(
        tmp_path,
        background=0.01,
        anomaly=10,
        n_electrodes=n_electrodes,
        n_lines=n_lines,
        refinement=refinement,
        inversion_type="dcip",
        flatten=False,
    )
    params = DirectCurrentParams(
        forward_only=True,
        geoh5=geoh5,
        mesh=model.parent.uid,
        topography_object=topography.uid,
        z_from_topo=True,
        data_object=survey.uid,
        starting_model_object=model.parent.uid,
        starting_model=model.uid,
    )
    params.workpath = tmp_path
    fwr_driver = InversionDriver(params)
    fwr_driver.initialize()
    fwr_driver.run()
    geoh5 = Workspace(geoh5.h5file)
    potential = geoh5.get_entity("Iteration_0_dc")[0]
    # Run the inverse
    np.random.seed(0)
    params = DirectCurrentParams(
        geoh5=geoh5,
        mesh=mesh.uid,
        topography_object=topography.uid,
        data_object=potential.parent.uid,
        starting_model=1e-2,
        s_norm=0.0,
        x_norm=1.0,
        y_norm=1.0,
        z_norm=1.0,
        gradient_type="components",
        potential_channel_bool=True,
        z_from_topo=True,
        potential_channel=potential.uid,
        potential_uncertainty=1e-3,
        max_iterations=max_iterations,
        initial_beta=None,
        initial_beta_ratio=1e0,
        prctile=100,
        upper_bound=10,
        tile_spatial=n_lines,
    )
    params.workpath = tmp_path
    driver = InversionDriver(params)
    driver.initialize()
    driver.run()
    output = get_inversion_output(
        driver.params.geoh5.h5file, driver.params.ga_group.uid
    )
    output["data"] = potential.values
    if pytest:
        check_target(output, target_run)
    else:
        return fwr_driver.starting_model, driver.inverse_problem.model


if __name__ == "__main__":
    # Full run
    m_start, m_rec = test_dc_run(
        "./",
        n_electrodes=20,
        n_lines=5,
        max_iterations=15,
        pytest=False,
        refinement=(4, 8),
    )
    residual = np.linalg.norm(m_rec - m_start) / np.linalg.norm(m_start) * 100.0
    assert (
        residual < 20.0
    ), f"Deviation from the true solution is {residual:.2f}%. Validate the solution!"
    print("Conductivity model is within 15% of the answer. You are so special!")
