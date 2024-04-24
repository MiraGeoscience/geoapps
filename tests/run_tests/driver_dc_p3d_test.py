#  Copyright (c) 2024 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from geoh5py.groups import SimPEGGroup
from geoh5py.workspace import Workspace
from simpeg_drivers.electricals.direct_current.pseudo_three_dimensions.driver import (
    DirectCurrentPseudo3DDriver,
)
from simpeg_drivers.electricals.direct_current.pseudo_three_dimensions.params import (
    DirectCurrentPseudo3DParams,
)

from geoapps.shared_utils.utils import get_inversion_output
from geoapps.utils.testing import check_target, setup_inversion_workspace

# To test the full run and validate the inversion.
# Move this file out of the test directory and run.

target_run = {"data_norm": 1.099, "phi_d": 4150, "phi_m": 0.7511}

np.random.seed(0)


def test_dc_p3d_fwr_run(
    tmp_path: Path,
    n_electrodes=10,
    n_lines=3,
    refinement=(4, 6),
):
    # Run the forward
    geoh5, _, model, survey, topography = setup_inversion_workspace(
        tmp_path,
        background=0.01,
        anomaly=10,
        n_electrodes=n_electrodes,
        n_lines=n_lines,
        refinement=refinement,
        inversion_type="dcip",
        drape_height=0.0,
        flatten=False,
    )
    params = DirectCurrentPseudo3DParams(
        forward_only=True,
        geoh5=geoh5,
        mesh=model.parent.uid,
        u_cell_size=5.0,
        v_cell_size=5.0,
        depth_core=100.0,
        expansion_factor=1.1,
        padding_distance=100.0,
        topography_object=topography.uid,
        z_from_topo=False,
        data_object=survey.uid,
        starting_model=model.uid,
        line_object=geoh5.get_entity("line_ids")[0].uid,
        cleanup=True,
    )
    params.workpath = tmp_path
    fwr_driver = DirectCurrentPseudo3DDriver(params)
    fwr_driver.run()


def test_dc_p3d_run(
    tmp_path: Path,
    max_iterations=1,
    pytest=True,
):
    workpath = tmp_path / "inversion_test.ui.geoh5"
    if pytest:
        workpath = tmp_path.parent / "test_dc_p3d_fwr_run0" / "inversion_test.ui.geoh5"

    with Workspace(workpath) as geoh5:
        potential = geoh5.get_entity("Iteration_0_dc")[0]
        out_group = geoh5.get_entity("Line 1")[0].parent
        mesh = out_group.get_entity("mesh")[0]  # Finds the octree mesh
        topography = geoh5.get_entity("topography")[0]

        # Run the inverse
        np.random.seed(0)
        params = DirectCurrentPseudo3DParams(
            geoh5=geoh5,
            mesh=mesh.uid,
            u_cell_size=5.0,
            v_cell_size=5.0,
            depth_core=100.0,
            expansion_factor=1.1,
            padding_distance=100.0,
            topography_object=topography.uid,
            data_object=potential.parent.uid,
            potential_channel=potential.uid,
            potential_uncertainty=1e-3,
            line_object=geoh5.get_entity("line_ids")[0].uid,
            starting_model=1e-2,
            reference_model=1e-2,
            s_norm=0.0,
            x_norm=1.0,
            y_norm=1.0,
            z_norm=1.0,
            gradient_type="components",
            potential_channel_bool=True,
            z_from_topo=False,
            max_global_iterations=max_iterations,
            initial_beta=None,
            initial_beta_ratio=10.0,
            prctile=100,
            upper_bound=10,
            coolingRate=1,
            cleanup=False,
        )
        params.write_input_file(path=tmp_path, name="Inv_run")

    driver = DirectCurrentPseudo3DDriver.start(str(tmp_path / "Inv_run.ui.json"))

    basepath = workpath.parent
    with open(basepath / "lookup.json", encoding="utf8") as f:
        lookup = json.load(f)
        middle_line_id = [k for k, v in lookup.items() if v["line_id"] == 101][0]

    with Workspace(basepath / f"{middle_line_id}.ui.geoh5", mode="r") as workspace:
        middle_inversion_group = [
            k for k in workspace.groups if isinstance(k, SimPEGGroup)
        ][0]
        filedata = middle_inversion_group.get_entity("SimPEG.out")[0]

        with driver.pseudo3d_params.out_group.workspace.open(mode="r+"):
            filedata.copy(parent=driver.pseudo3d_params.out_group)

    output = get_inversion_output(
        driver.pseudo3d_params.geoh5.h5file, driver.pseudo3d_params.out_group.uid
    )
    if geoh5.open():
        output["data"] = potential.values
    if pytest:
        check_target(output, target_run)


if __name__ == "__main__":
    # Full run
    test_dc_p3d_fwr_run(
        Path("./"),
        n_electrodes=20,
        n_lines=3,
        refinement=(4, 8),
    )
    test_dc_p3d_run(
        Path("./"),
        max_iterations=20,
        pytest=False,
    )
