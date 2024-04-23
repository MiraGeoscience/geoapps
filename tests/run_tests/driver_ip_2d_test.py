# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024 Mira Geoscience Ltd.                                     '
#                                                                              '
#  This file is part of geoapps.                                               '
#                                                                              '
#  geoapps is distributed under the terms and conditions of the MIT License    '
#  (see LICENSE file at the root of this source code package).                 '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from __future__ import annotations

from pathlib import Path

import numpy as np
from geoh5py.workspace import Workspace

from geoapps.inversion.electricals.induced_polarization.two_dimensions import (
    InducedPolarization2DParams,
)
from geoapps.inversion.electricals.induced_polarization.two_dimensions.driver import (
    InducedPolarization2DDriver,
)
from geoapps.shared_utils.utils import get_inversion_output
from geoapps.utils.testing import check_target, setup_inversion_workspace

# To test the full run and validate the inversion.
# Move this file out of the test directory and run.

target_run = {"data_norm": 0.09141, "phi_d": 8981, "phi_m": 0.1124}


def test_ip_2d_fwr_run(
    tmp_path: Path,
    n_electrodes=10,
    n_lines=3,
    refinement=(4, 6),
):
    # Run the forward
    geoh5, _, model, survey, topography = setup_inversion_workspace(
        tmp_path,
        background=1e-6,
        anomaly=1e-1,
        n_electrodes=n_electrodes,
        n_lines=n_lines,
        refinement=refinement,
        inversion_type="dcip_2d",
        flatten=False,
        drape_height=0.0,
    )
    params = InducedPolarization2DParams(
        forward_only=True,
        geoh5=geoh5,
        mesh=model.parent.uid,
        topography_object=topography.uid,
        z_from_topo=True,
        data_object=survey.uid,
        starting_model=model.uid,
        conductivity_model=1e-2,
        line_object=geoh5.get_entity("line_ids")[0].uid,
        line_id=101,
    )
    params.workpath = tmp_path
    fwr_driver = InducedPolarization2DDriver(params)
    fwr_driver.run()


def test_ip_2d_run(
    tmp_path: Path,
    max_iterations=1,
    pytest=True,
):
    workpath = tmp_path / "inversion_test.ui.geoh5"
    if pytest:
        workpath = tmp_path.parent / "test_ip_2d_fwr_run0" / "inversion_test.ui.geoh5"

    with Workspace(workpath) as geoh5:
        chargeability = geoh5.get_entity("Iteration_0_ip")[0]
        mesh = geoh5.get_entity("Models")[0]
        topography = geoh5.get_entity("topography")[0]

        # Run the inverse
        params = InducedPolarization2DParams(
            geoh5=geoh5,
            mesh=mesh.uid,
            topography_object=topography.uid,
            data_object=chargeability.parent.uid,
            chargeability_channel=chargeability.uid,
            chargeability_uncertainty=2e-4,
            line_object=geoh5.get_entity("line_ids")[0].uid,
            line_id=101,
            starting_model=1e-6,
            reference_model=1e-6,
            conductivity_model=1e-2,
            s_norm=0.0,
            x_norm=0.0,
            z_norm=0.0,
            gradient_type="components",
            chargeability_channel_bool=True,
            z_from_topo=True,
            max_global_iterations=max_iterations,
            initial_beta=None,
            initial_beta_ratio=1e0,
            prctile=100,
            upper_bound=0.1,
            store_sensitivities="ram",
            coolingRate=1,
        )
        params.write_input_file(path=tmp_path, name="Inv_run")

    driver = InducedPolarization2DDriver.start(str(tmp_path / "Inv_run.ui.json"))

    output = get_inversion_output(
        driver.params.geoh5.h5file, driver.params.out_group.uid
    )
    if geoh5.open():
        output["data"] = chargeability.values[np.isfinite(chargeability.values)]
    if pytest:
        check_target(output, target_run)


if __name__ == "__main__":
    # Full run
    test_ip_2d_fwr_run(
        Path("./"),
        n_electrodes=20,
        n_lines=3,
        refinement=(4, 8),
    )
    test_ip_2d_run(
        Path("./"),
        max_iterations=20,
        pytest=False,
    )
