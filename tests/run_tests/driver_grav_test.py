#  Copyright (c) 2024 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from pathlib import Path

import numpy as np
from geoh5py.workspace import Workspace
from simpeg_drivers.components import InversionData
from simpeg_drivers.potential_fields import GravityParams
from simpeg_drivers.potential_fields.gravity.driver import GravityDriver

from geoapps.shared_utils.utils import get_inversion_output
from geoapps.utils.testing import check_target, setup_inversion_workspace

# To test the full run and validate the inversion.
# Move this file out of the test directory and run.

target_run = {"data_norm": 0.0011829136269148985, "phi_d": 4.475e-05, "phi_m": 0.00144}


def test_gravity_fwr_run(
    tmp_path: Path,
    n_grid_points=2,
    refinement=(2,),
):
    np.random.seed(0)
    # Run the forward
    geoh5, _, model, survey, topography = setup_inversion_workspace(
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
        geoh5=Path(geoh5.h5file),
        mesh=model.parent.uid,
        topography_object=topography.uid,
        resolution=0.0,
        z_from_topo=False,
        data_object=survey.uid,
        starting_model=model.uid,
    )
    fwr_driver = GravityDriver(params)
    fwr_driver.run()


def test_gravity_run(
    tmp_path: Path,
    max_iterations=1,
    pytest=True,
):
    workpath = tmp_path / "inversion_test.ui.geoh5"
    if pytest:
        workpath = tmp_path.parent / "test_gravity_fwr_run0" / "inversion_test.ui.geoh5"

    with Workspace(workpath) as geoh5:
        gz = geoh5.get_entity("Iteration_0_gz")[0]
        mesh = geoh5.get_entity("mesh")[0]
        topography = geoh5.get_entity("topography")[0]

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
            reference_model=0.0,
            s_norm=0.0,
            x_norm=0.0,
            y_norm=0.0,
            z_norm=0.0,
            gradient_type="components",
            gz_channel_bool=True,
            z_from_topo=False,
            gz_channel=gz.uid,
            gz_uncertainty=2e-3,
            lower_bound=0.0,
            max_global_iterations=max_iterations,
            initial_beta_ratio=1e-2,
            prctile=100,
            store_sensitivities="ram",
        )
        params.write_input_file(path=tmp_path, name="Inv_run")

        data = InversionData(geoh5, params)
        survey = data.create_survey()

        assert survey[0].dobs[0] == survey[0].dummy

    driver = GravityDriver.start(str(tmp_path / "Inv_run.ui.json"))

    assert driver.params.data_object.uid != gz.parent.uid
    assert driver.models.upper_bound is np.inf

    with Workspace(driver.params.geoh5.h5file) as run_ws:
        output = get_inversion_output(
            driver.params.geoh5.h5file, driver.params.out_group.uid
        )

        assert np.array([o is not np.nan for o in output["phi_d"]]).any()
        assert np.array([o is not np.nan for o in output["phi_m"]]).any()

        residual = run_ws.get_entity("Iteration_1_gz_Residual")[0]
        assert np.isnan(residual.values).sum() == 1, "Number of nan residuals differ."

        predicted = [
            pred
            for pred in run_ws.get_entity("Iteration_0_gz")
            if pred.parent.parent.name == "Gravity Inversion"
        ][0]

        assert not any(
            np.isnan(predicted.values)
        ), "Predicted data should not have nans."

        output["data"] = predicted.values

        assert len(run_ws.get_entity("SimPEG.log")) == 2

        if pytest:
            check_target(output, target_run)
            nan_ind = np.isnan(run_ws.get_entity("Iteration_0_model")[0].values)
            inactive_ind = run_ws.get_entity("active_cells")[0].values == 0
            assert np.all(nan_ind == inactive_ind)


if __name__ == "__main__":
    # Full run
    test_gravity_fwr_run(
        Path("./"),
        n_grid_points=20,
        refinement=(4, 8),
    )

    test_gravity_run(
        Path("./"),
        max_iterations=15,
        pytest=False,
    )
