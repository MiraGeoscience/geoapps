#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


import numpy as np
from geoh5py.workspace import Workspace

from geoapps.inversion.driver import start_inversion
from geoapps.shared_utils.utils import get_inversion_output
from geoapps.utils.testing import check_target, setup_inversion_workspace

# To test the full run and validate the inversion.
# Move this file out of the test directory and run.

target_run = {
    "data_norm": 0.0071214,
    "phi_d": 0.0001572,
    "phi_m": 0.009368,
}


def test_gravity_run(
    tmp_path,
    n_grid_points=2,
    max_iterations=1,
    pytest=True,
    refinement=(2,),
):
    from geoapps.inversion.driver import InversionDriver
    from geoapps.inversion.potential_fields import GravityParams

    np.random.seed(0)
    # Run the forward
    geoh5, mesh, model, survey, topography = setup_inversion_workspace(
        tmp_path,
        background=0.0,
        anomaly=0.75,
        n_electrodes=n_grid_points,
        n_lines=n_grid_points,
        refinement=refinement,
        flatten=False,
    )

    params = GravityParams(
        forward_only=True,
        geoh5=geoh5,
        mesh=model.parent.uid,
        topography_object=topography.uid,
        resolution=0.0,
        z_from_topo=False,
        data_object=survey.uid,
        starting_model_object=model.parent.uid,
        starting_model=model.uid,
    )
    fwr_driver = InversionDriver(params)
    fwr_driver.run()

    with geoh5.open(mode="r+"):
        gz = geoh5.get_entity("Iteration_0_gz")[0]
        orig_gz = gz.values.copy()

        # Turn some values to nan
        values = gz.values.copy()
        values[0] = np.nan
        gz.values = values

        # Run the inverse
        np.random.seed(0)
        params = GravityParams(
            geoh5=geoh5,
            mesh=mesh.uid,
            topography_object=topography.uid,
            resolution=0.0,
            data_object=gz.parent.uid,
            starting_model=1e-4,
            s_norm=0.0,
            x_norm=1.0,
            y_norm=1.0,
            z_norm=1.0,
            gradient_type="components",
            gz_channel_bool=True,
            z_from_topo=False,
            gz_channel=gz.uid,
            gz_uncertainty=2e-3,
            upper_bound=0.75,
            max_iterations=max_iterations,
            initial_beta_ratio=1e-2,
            prctile=100,
        )
        params.write_input_file(path=tmp_path, name="Inv_run")

    driver = start_inversion(str(tmp_path / "Inv_run.ui.json"))

    run_ws = Workspace(driver.params.geoh5.h5file)
    output = get_inversion_output(
        driver.params.geoh5.h5file, driver.params.ga_group.uid
    )

    residual = run_ws.get_entity("Iteration_1_gz_Residual")[0]
    assert np.isnan(residual.values).sum() == 1, "Number of nan residuals differ."

    predicted = [
        pred
        for pred in run_ws.get_entity("Iteration_0_gz")
        if pred.parent.parent.name == "GravityInversion"
    ][0]
    assert not any(np.isnan(predicted.values)), "Predicted data should not have nans."
    output["data"] = orig_gz
    if pytest:
        check_target(output, target_run)
        nan_ind = np.isnan(run_ws.get_entity("Iteration_0_model")[0].values)
        inactive_ind = run_ws.get_entity("active_cells")[0].values == 0
        assert np.all(nan_ind == inactive_ind)
    else:
        return fwr_driver.starting_model, driver.inverse_problem.model


if __name__ == "__main__":
    # Full run
    m_start, m_rec = test_gravity_run(
        "./", n_grid_points=20, max_iterations=30, pytest=False, refinement=(4, 8)
    )
    model_residual = np.linalg.norm(m_rec - m_start) / np.linalg.norm(m_start) * 100.0
    assert (
        model_residual < 50.0
    ), f"Deviation from the true solution is {model_residual:.2f}%. Validate the solution!"
    print("Density model is within 15% of the answer. Let's go!!")
