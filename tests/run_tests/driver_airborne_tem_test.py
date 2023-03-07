#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import os

import numpy as np
from geoh5py.workspace import Workspace

from scipy.interpolate import interp1d

from geoapps.inversion.airborne_electromagnetics.time_domain import (
    TimeDomainElectromagneticsParams,
)
from geoapps.inversion.airborne_electromagnetics.time_domain.driver import (
    TimeDomainElectromagneticsDriver,
)
from geoapps.shared_utils.utils import get_inversion_output
from geoapps.utils.testing import check_target, setup_inversion_workspace

# To test the full run and validate the inversion.
# Move this file out of the test directory and run.

target_run = {
    "data_norm": 0.00877,
    "phi_d": 2.396,
    "phi_m": 0.3094,
}

np.random.seed(0)


def test_airborne_tem_fwr_run(
    tmp_path,
    n_grid_points=5,
    refinement=(2,),
):
    # Run the forward
    geoh5, _, model, survey, topography = setup_inversion_workspace(
        tmp_path,
        background=0.001,
        anomaly=1.0,
        n_electrodes=n_grid_points,
        n_lines=n_grid_points,
        refinement=refinement,
        inversion_type="airborne_tem",
        drape_height=15.0,
        flatten=True,
    )
    params = TimeDomainElectromagneticsParams(
        forward_only=True,
        geoh5=geoh5,
        mesh=model.parent.uid,
        topography_object=topography.uid,
        resolution=0.0,
        z_from_topo=False,
        data_object=survey.uid,
        starting_model=model.uid,
        conductivity_model=1e-2,
        x_channel_bool=True,
        y_channel_bool=True,
        z_channel_bool=True,
    )
    params.workpath = tmp_path
    fwr_driver = TimeDomainElectromagneticsDriver(params, warmstart=False)
    fwr_driver.run()

    return fwr_driver.starting_model


def test_airborne_tem_run(tmp_path, max_iterations=1, pytest=True):
    workpath = os.path.join(tmp_path, "inversion_test.geoh5")
    if pytest:
        workpath = str(tmp_path / "../test_airborne_tem_fwr_run0/inversion_test.geoh5")

    with Workspace(workpath) as geoh5:
        survey = geoh5.get_entity("Airborne_rx")[0]
        mesh = geoh5.get_entity("mesh")[0]
        topography = geoh5.get_entity("topography")[0]

        data = {}
        uncertainties = {}
        components = {
            "z": "dBzdt",
        }

        for comp, cname in components.items():
            data[cname] = []
            uncertainties[f"{cname} uncertainties"] = []
            for tt, time in enumerate(survey.channels):
                data_entity = geoh5.get_entity(f"Iteration_0_{comp}_{time:.2e}")[
                    0
                ].copy(parent=survey)
                data[cname].append(data_entity)


                uncert = survey.add_data(
                    {
                        f"uncertainty_{comp}_{time:.2e}": {
                            "values": np.ones_like(data_entity.values) * (np.median(np.abs(data_entity.values)))
                        }
                    }

                )
                uncertainties[f"{cname} uncertainties"].append(uncert)
                # uncertainties[f"{cname} uncertainties"][freq] = {"values": u.copy(parent=survey)}

        survey.add_components_data(data)
        survey.add_components_data(uncertainties)

        data_kwargs = {}
        for i, comp in enumerate(components):
            data_kwargs[f"{comp}_channel"] = survey.find_or_create_property_group(name=f"Iteration_0_{comp}")
            data_kwargs[f"{comp}_uncertainty"] = survey.find_or_create_property_group(name=f"dB{comp}dt uncertainties")

        # orig_dBzdt = geoh5.get_entity("Iteration_0_z_1.00e-05")[0].values

        # Run the inverse
        np.random.seed(0)
        params = TimeDomainElectromagneticsParams(
            geoh5=geoh5,
            mesh=mesh.uid,
            topography_object=topography.uid,
            resolution=0.0,
            data_object=survey.uid,
            starting_model=1e-3,
            reference_model=1e-3,
            chi_factor=0.001,
            s_norm=2.0,
            x_norm=2.0,
            y_norm=2.0,
            z_norm=2.0,
            alpha_s=0.0,
            gradient_type="total",
            z_from_topo=False,
            lower_bound=2e-6,
            upper_bound=1e2,
            max_global_iterations=max_iterations,
            initial_beta_ratio=1e2,
            cooling_rate=3,
            sens_wts_threshold=1.0,
            prctile=90,
            store_sensitivities="ram",
            **data_kwargs,
        )
        params.write_input_file(path=tmp_path, name="Inv_run")
        driver = TimeDomainElectromagneticsDriver.start(
            os.path.join(tmp_path, "Inv_run.ui.json")
        )

    with geoh5.open() as run_ws:
        output = get_inversion_output(
            driver.params.geoh5.h5file, driver.params.ga_group.uid
        )
        # output["data"] = orig_dBzdt
        if pytest:
            check_target(output, target_run, tolerance=0.5)
            nan_ind = np.isnan(run_ws.get_entity("Iteration_0_model")[0].values)
            inactive_ind = run_ws.get_entity("active_cells")[0].values == 0
            assert np.all(nan_ind == inactive_ind)
        else:
            return driver.inverse_problem.model


if __name__ == "__main__":
    # Full run
    mstart = test_airborne_tem_fwr_run("./", n_grid_points=8, refinement=(4, 8))

    m_rec = test_airborne_tem_run(
        "./",
        max_iterations=15,
        pytest=False,
    )

    residual = np.linalg.norm(m_rec - mstart) / np.linalg.norm(mstart) * 100.0
    assert (
        residual < 50.0
    ), f"Deviation from the true solution is {residual:.2f}%. Validate the solution!"
    print("Conductivity model is within 50% of the answer. Let's go!!")
