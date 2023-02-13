#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).
import json
import os

import numpy as np
from geoh5py.data import FilenameData
from geoh5py.groups import SimPEGGroup
from geoh5py.workspace import Workspace

from geoapps.inversion.electricals.induced_polarization.pseudo_three_dimensions.driver import (
    InducedPolarizationPseudo3DDriver,
)
from geoapps.inversion.electricals.induced_polarization.pseudo_three_dimensions.params import (
    InducedPolarizationPseudo3DParams,
)
from geoapps.shared_utils.utils import get_inversion_output
from geoapps.utils.surveys import survey_lines
from geoapps.utils.testing import check_target, setup_inversion_workspace

# To test the full run and validate the inversion.
# Move this file out of the test directory and run.

target_run = {
    "data_norm": 0.05971,
    "phi_d": 1974,
    "phi_m": 0.02411,
}

np.random.seed(0)


def test_ip_p3d_fwr_run(
    tmp_path,
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
        inversion_type="dcip",
        drape_height=0.0,
        flatten=False,
    )

    _ = survey_lines(survey, [-100, -100], save="line_ids")
    params = InducedPolarizationPseudo3DParams(
        forward_only=True,
        geoh5=geoh5,
        mesh=model.parent.uid,
        u_cell_size=5.0,
        v_cell_size=5.0,
        depth_core=100.0,
        expansion_factor=1.1,
        padding_distance=100.0,
        topography_object=topography.uid,
        z_from_topo=True,
        data_object=survey.uid,
        conductivity_model=1e-2,
        starting_model=model.uid,
        line_object=geoh5.get_entity("line_ids")[0].uid,
        cleanup=True,
    )
    params.workpath = tmp_path
    fwr_driver = InducedPolarizationPseudo3DDriver(params)
    fwr_driver.run()

    drape_model = geoh5.get_entity("Line 2")[0]
    starting_model = [c for c in drape_model.children if c.name == "starting_model"][
        0
    ].values

    return starting_model


def test_ip_p3d_run(
    tmp_path,
    max_iterations=1,
    pytest=True,
):
    workpath = os.path.join(tmp_path, "inversion_test.geoh5")
    if pytest:
        workpath = os.path.abspath(
            tmp_path / "../test_ip_p3d_fwr_run0/inversion_test.geoh5"
        )

    with Workspace(workpath) as geoh5:
        chargeability = geoh5.get_entity("Iteration_0_ip")[0]
        models = geoh5.get_entity("Models")[0]
        mesh = models.get_entity("mesh")[0]  # Finds the octree mesh
        topography = geoh5.get_entity("topography")[0]
        _ = survey_lines(chargeability.parent, [-100, 100], save="line_IDs")

        # Run the inverse
        np.random.seed(0)
        params = InducedPolarizationPseudo3DParams(
            geoh5=geoh5,
            mesh=mesh.uid,
            u_cell_size=5.0,
            v_cell_size=5.0,
            depth_core=100.0,
            expansion_factor=1.1,
            padding_distance=100.0,
            topography_object=topography.uid,
            data_object=chargeability.parent.uid,
            chargeability_channel=chargeability.uid,
            chargeability_uncertainty=2e-4,
            line_object=geoh5.get_entity("line_IDs")[0].uid,
            conductivity_model=1e-2,
            starting_model=1e-6,
            reference_model=1e-6,
            s_norm=0.0,
            x_norm=0.0,
            y_norm=0.0,
            z_norm=0.0,
            gradient_type="components",
            chargeability_channel_bool=True,
            z_from_topo=True,
            max_global_iterations=max_iterations,
            initial_beta=None,
            initial_beta_ratio=1e0,
            prctile=100,
            upper_bound=0.1,
            coolingRate=1,
            cleanup=False,
        )
        params.write_input_file(path=tmp_path, name="Inv_run")

    driver = InducedPolarizationPseudo3DDriver.start(
        os.path.join(tmp_path, "Inv_run.ui.json")
    )

    basepath = os.path.dirname(workpath)
    with open(os.path.join(basepath, "lookup.json"), encoding="utf8") as f:
        lookup = json.load(f)
        middle_line_id = [k for k, v in lookup.items() if v["line_id"] == 2][0]

    with Workspace(
        os.path.join(basepath, f"{middle_line_id}.ui.geoh5"), mode="r"
    ) as workspace:
        middle_inversion_group = [
            k for k in workspace.groups if isinstance(k, SimPEGGroup)
        ][0]
        filedata = [
            k for k in middle_inversion_group.children if isinstance(k, FilenameData)
        ][0]

        with driver.pseudo3d_params.ga_group.workspace.open(mode="r+"):
            filedata.copy(parent=driver.pseudo3d_params.ga_group)

    output = get_inversion_output(
        driver.pseudo3d_params.geoh5.h5file, driver.pseudo3d_params.ga_group.uid
    )
    if geoh5.open():
        output["data"] = chargeability.values
    if pytest:
        check_target(output, target_run)
    else:
        return geoh5.get_entity("Iteration_1_model")[0].values


if __name__ == "__main__":
    # Full run
    m_start = test_ip_p3d_fwr_run(
        "./",
        n_electrodes=20,
        n_lines=3,
        refinement=(4, 8),
    )

    m_rec = test_ip_p3d_run(
        "./",
        max_iterations=20,
        pytest=False,
    )
    residual = np.linalg.norm(m_rec - m_start) / np.linalg.norm(m_start) * 100.0
    assert (
        residual < 20.0
    ), f"Deviation from the true solution is {residual:.2f}%. Validate the solution!"
    print("Conductivity model is within 20% of the answer. You are so special!")
