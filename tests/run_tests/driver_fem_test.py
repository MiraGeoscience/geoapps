# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024 Mira Geoscience Ltd.                                     '
#                                                                              '
#  This file is part of geoapps.                                               '
#                                                                              '
#  geoapps is distributed under the terms and conditions of the MIT License    '
#  (see LICENSE file at the root of this source code package).                 '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# pylint: disable=too-many-locals

from __future__ import annotations

from pathlib import Path

import numpy as np
from geoh5py import Workspace
from geoh5py.groups import RootGroup

from geoapps.inversion.electromagnetics.frequency_domain.driver import (
    FrequencyDomainElectromagneticsDriver,
)
from geoapps.inversion.electromagnetics.frequency_domain.params import (
    FrequencyDomainElectromagneticsParams,
)
from geoapps.shared_utils.utils import get_inversion_output
from geoapps.utils.testing import check_target, setup_inversion_workspace

# To test the full run and validate the inversion.
# Move this file out of the test directory and run.

target_run = {"data_norm": 47.522882323952054, "phi_d": 364.3, "phi_m": 443.3}


def test_fem_fwr_run(
    tmp_path: Path,
    n_grid_points=3,
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
        drape_height=15.0,
        padding_distance=400,
        inversion_type="fem",
        flatten=True,
    )
    params = FrequencyDomainElectromagneticsParams(
        forward_only=True,
        geoh5=geoh5,
        mesh=model.parent.uid,
        topography_object=topography.uid,
        resolution=0.0,
        z_from_topo=False,
        data_object=survey.uid,
        starting_model=model.uid,
        z_real_channel_bool=True,
        z_imag_channel_bool=True,
    )
    params.workpath = tmp_path
    fwr_driver = FrequencyDomainElectromagneticsDriver(params)
    fwr_driver.run()
    geoh5.close()


def test_fem_run(tmp_path: Path, max_iterations=1, pytest=True):
    workpath = tmp_path / "inversion_test.ui.geoh5"
    if pytest:
        workpath = tmp_path.parent / "test_fem_fwr_run0" / "inversion_test.ui.geoh5"

    with Workspace(workpath) as geoh5:
        survey = [
            s
            for s in geoh5.get_entity("Airborne_rx")
            if isinstance(s.parent, RootGroup)
        ][0]
        mesh = geoh5.get_entity("mesh")[0]
        topography = geoh5.get_entity("topography")[0]
        data = {}
        uncertainties = {}
        components = {
            "z_real": "z_real",
            "z_imag": "z_imag",
        }

        for comp, cname in components.items():
            data[cname] = []
            uncertainties[f"{cname} uncertainties"] = []
            for ind, freq in enumerate(survey.channels):
                data_entity = geoh5.get_entity(f"Iteration_0_{comp}_[{ind}]")[0].copy(
                    parent=survey
                )
                data[cname].append(data_entity)
                abs_val = np.abs(data_entity.values)
                uncert = survey.add_data(
                    {
                        f"uncertainty_{comp}_[{ind}]": {
                            "values": np.ones_like(abs_val) * freq / 200.0
                        }
                    }
                )
                uncertainties[f"{cname} uncertainties"].append(
                    uncert.copy(parent=survey)
                )

        data_groups = survey.add_components_data(data)
        uncert_groups = survey.add_components_data(uncertainties)

        data_kwargs = {}
        for comp, data_group, uncert_group in zip(
            components, data_groups, uncert_groups
        ):
            data_kwargs[f"{comp}_channel"] = data_group.uid
            data_kwargs[f"{comp}_uncertainty"] = uncert_group.uid

        orig_z_real_1 = geoh5.get_entity("Iteration_0_z_real_[0]")[0].values

        # Run the inverse
        params = FrequencyDomainElectromagneticsParams(
            geoh5=geoh5,
            mesh=mesh.uid,
            topography_object=topography.uid,
            resolution=0.0,
            data_object=survey.uid,
            starting_model=1e-3,
            reference_model=1e-3,
            alpha_s=0.0,
            s_norm=0.0,
            x_norm=0.0,
            y_norm=0.0,
            z_norm=0.0,
            gradient_type="components",
            z_from_topo=False,
            upper_bound=0.75,
            max_global_iterations=max_iterations,
            initial_beta_ratio=1e1,
            prctile=100,
            coolingRate=3,
            chi_factor=0.25,
            store_sensitivities="ram",
            sens_wts_threshold=1.0,
            **data_kwargs,
        )
        params.write_input_file(path=tmp_path, name="Inv_run")
        driver = FrequencyDomainElectromagneticsDriver(params)
        driver.run()

    with geoh5.open() as run_ws:
        output = get_inversion_output(
            driver.params.geoh5.h5file, driver.params.out_group.uid
        )
        output["data"] = orig_z_real_1

        assert (
            run_ws.get_entity("Iteration_1_z_imag_[1]")[0].entity_type.uid
            == run_ws.get_entity("Observed_z_imag_[1]")[0].entity_type.uid
        )

        if pytest:
            check_target(output, target_run, tolerance=0.5)
            nan_ind = np.isnan(run_ws.get_entity("Iteration_0_model")[0].values)
            inactive_ind = run_ws.get_entity("active_cells")[0].values == 0
            assert np.all(nan_ind == inactive_ind)


if __name__ == "__main__":
    # Full run
    test_fem_fwr_run(Path("./"), n_grid_points=5, refinement=(4, 4, 4))
    test_fem_run(
        Path("./"),
        max_iterations=15,
        pytest=False,
    )
