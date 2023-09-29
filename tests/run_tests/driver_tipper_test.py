#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from pathlib import Path

import numpy as np
from geoh5py.workspace import Workspace

from geoapps.inversion.natural_sources import TipperParams
from geoapps.inversion.natural_sources.tipper.driver import TipperDriver
from geoapps.shared_utils.utils import get_inversion_output
from geoapps.utils.testing import check_target, setup_inversion_workspace

# To test the full run and validate the inversion.
# Move this file out of the test directory and run.

target_run = {"data_norm": 0.0020959218368283884, "phi_d": 0.123, "phi_m": 3632}

np.random.seed(0)


def test_tipper_fwr_run(
    tmp_path: Path,
    n_grid_points=2,
    refinement=(2,),
):
    # Run the forward
    geoh5, _, model, survey, topography = setup_inversion_workspace(
        tmp_path,
        background=1e-3,
        anomaly=1.0,
        n_electrodes=n_grid_points,
        n_lines=n_grid_points,
        refinement=refinement,
        inversion_type="tipper",
        drape_height=15.0,
        flatten=False,
    )
    params = TipperParams(
        forward_only=True,
        geoh5=geoh5,
        mesh=model.parent.uid,
        topography_object=topography.uid,
        resolution=0.0,
        z_from_topo=False,
        data_object=survey.uid,
        starting_model=model.uid,
        conductivity_model=1e-3,
        txz_real_channel_bool=True,
        txz_imag_channel_bool=True,
        tyz_real_channel_bool=True,
        tyz_imag_channel_bool=True,
    )
    params.workpath = tmp_path
    fwr_driver = TipperDriver(params)
    fwr_driver.run()


def test_tipper_run(tmp_path: Path, max_iterations=1, pytest=True):
    workpath = tmp_path / "inversion_test.ui.geoh5"
    if pytest:
        workpath = tmp_path.parent / "test_tipper_fwr_run0" / "inversion_test.ui.geoh5"

    with Workspace(workpath) as geoh5:
        survey = geoh5.get_entity("survey")[0]
        mesh = geoh5.get_entity("mesh")[0]
        topography = geoh5.get_entity("topography")[0]

        data = {}
        uncertainties = {}
        components = {
            "txz_real": "Txz (real)",
            "txz_imag": "Txz (imag)",
            "tyz_real": "Tyz (real)",
            "tyz_imag": "Tyz (imag)",
        }

        for comp, cname in components.items():
            data[cname] = []
            uncertainties[f"{cname} uncertainties"] = []
            for ind in range(len(survey.channels)):
                data_entity = geoh5.get_entity(f"Iteration_0_{comp}_[{ind}]")[0].copy(
                    parent=survey
                )
                data[cname].append(data_entity)

                uncert = survey.add_data(
                    {
                        f"uncertainty_{comp}_[{ind}]": {
                            "values": np.ones_like(data_entity.values)
                            * np.percentile(np.abs(data_entity.values), 5)
                        }
                    }
                )
                uncertainties[f"{cname} uncertainties"].append(uncert)

        data_groups = survey.add_components_data(data)
        uncert_groups = survey.add_components_data(uncertainties)

        data_kwargs = {}
        for comp, data_group, uncert_group in zip(
            components, data_groups, uncert_groups
        ):
            data_kwargs[f"{comp}_channel"] = data_group.uid
            data_kwargs[f"{comp}_uncertainty"] = uncert_group.uid

        orig_tyz_real_1 = geoh5.get_entity("Iteration_0_tyz_real_[0]")[0].values

        # Run the inverse
        np.random.seed(0)
        params = TipperParams(
            geoh5=geoh5,
            mesh=mesh.uid,
            topography_object=topography.uid,
            resolution=0.0,
            data_object=survey.uid,
            starting_model=0.001,
            reference_model=0.001,
            conductivity_model=1e-3,
            s_norm=1.0,
            x_norm=1.0,
            y_norm=1.0,
            z_norm=1.0,
            alpha_s=1.0,
            gradient_type="components",
            z_from_topo=False,
            upper_bound=0.75,
            max_global_iterations=max_iterations,
            initial_beta_ratio=1e2,
            coolingRate=2,
            prctile=100,
            chi_factor=0.1,
            store_sensitivities="ram",
            **data_kwargs,
        )
        params.write_input_file(path=tmp_path, name="Inv_run")
        driver = TipperDriver.start(str(tmp_path / "Inv_run.ui.json"))

    with geoh5.open() as run_ws:
        output = get_inversion_output(
            driver.params.geoh5.h5file, driver.params.out_group.uid
        )
        output["data"] = orig_tyz_real_1
        if pytest:
            check_target(output, target_run, tolerance=0.5)
            nan_ind = np.isnan(run_ws.get_entity("Iteration_0_model")[0].values)
            inactive_ind = run_ws.get_entity("active_cells")[0].values == 0
            assert np.all(nan_ind == inactive_ind)


if __name__ == "__main__":
    # Full run
    test_tipper_fwr_run(Path("./"), n_grid_points=8, refinement=(4, 4))
    test_tipper_run(
        Path("./"),
        max_iterations=15,
        pytest=False,
    )
