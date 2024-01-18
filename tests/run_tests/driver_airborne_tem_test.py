#  Copyright (c) 2023-2024 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from pathlib import Path

import numpy as np
from geoh5py.groups import RootGroup
from geoh5py.workspace import Workspace

from geoapps.inversion.components import InversionData
from geoapps.inversion.electromagnetics.time_domain import (
    TimeDomainElectromagneticsParams,
)
from geoapps.inversion.electromagnetics.time_domain.driver import (
    TimeDomainElectromagneticsDriver,
)
from geoapps.shared_utils.utils import get_inversion_output
from geoapps.utils.testing import check_target, setup_inversion_workspace

# To test the full run and validate the inversion.
# Move this file out of the test directory and run.

target_run = {
    "data_norm": 2.81018e-10,
    "phi_d": 15400,
    "phi_m": 718.9,
}

np.random.seed(0)


def test_airborne_tem_fwr_run(
    tmp_path: Path,
    n_grid_points=3,
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
        drape_height=10.0,
        padding_distance=400.0,
        flatten=False,
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
        x_channel_bool=True,
        y_channel_bool=True,
        z_channel_bool=True,
    )
    params.workpath = tmp_path
    fwr_driver = TimeDomainElectromagneticsDriver(params)
    fwr_driver.run()


def setup_airborne_tem_run(tmp_path: Path, pytest=True):
    workpath = tmp_path / "inversion_test.ui.geoh5"
    if pytest:
        workpath = (
            tmp_path.parent / "test_airborne_tem_fwr_run0" / "inversion_test.ui.geoh5"
        )

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
        properties = {}
        components = {"z": "dBzdt"}

        for comp, cname in components.items():
            data[cname] = []
            uncertainties[f"{cname} uncertainties"] = []
            properties[comp] = []
            for ii, _ in enumerate(survey.channels):
                data_entity = geoh5.get_entity(f"Iteration_0_{comp}_[{ii}]")[0].copy(
                    parent=survey
                )
                data[cname].append(data_entity)
                properties[comp].append(data_entity.uid)

                uncert = survey.add_data(
                    {
                        f"uncertainty_{comp}_[{ii}]": {
                            "values": np.ones_like(data_entity.values)
                            * (np.median(np.abs(data_entity.values)))
                        }
                    }
                )
                uncertainties[f"{cname} uncertainties"].append(uncert)

        survey.add_components_data(data)
        survey.add_components_data(uncertainties)

        data_kwargs = {}
        for comp in components:
            data_kwargs[f"{comp}_channel"] = survey.find_or_create_property_group(
                name=f"Iteration_0_{comp}",
                properties=properties[comp],
            )
            data_kwargs[f"{comp}_uncertainty"] = survey.find_or_create_property_group(
                name=f"dB{comp}dt uncertainties"
            )

        orig_dBzdt = geoh5.get_entity("Iteration_0_z_[0]")[0].values

        return workpath, mesh, topography, survey, data_kwargs, orig_dBzdt


def test_airborne_tem_run(tmp_path: Path, max_iterations=1, pytest=True):
    (
        workpath,
        mesh,
        topography,
        survey,
        data_kwargs,
        _,
    ) = setup_airborne_tem_run(tmp_path, pytest)
    with Workspace(workpath) as geoh5:
        # Set some data as nan
        vals = geoh5.get_entity(survey.uid)[0].get_data("Iteration_0_z_[0]")[0].values
        vals[0] = np.nan
        geoh5.get_entity(survey.uid)[0].get_data("Iteration_0_z_[0]")[0].values = vals
        orig_dBzdt = vals

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
            chi_factor=1.0,
            s_norm=2.0,
            x_norm=2.0,
            y_norm=2.0,
            z_norm=2.0,
            alpha_s=1e-4,
            gradient_type="total",
            z_from_topo=False,
            lower_bound=2e-6,
            upper_bound=1e2,
            max_global_iterations=max_iterations,
            initial_beta_ratio=1e2,
            coolingRate=4,
            max_cg_iterations=200,
            prctile=5,
            store_sensitivities="ram",
            **data_kwargs,
        )
        params.write_input_file(path=tmp_path, name="Inv_run")

        data = InversionData(geoh5, params)
        survey = data.create_survey()

        assert survey[0].dobs[0] == survey[0].dummy

    driver = TimeDomainElectromagneticsDriver.start(str(tmp_path / "Inv_run.ui.json"))

    with geoh5.open() as run_ws:
        output = get_inversion_output(
            driver.params.geoh5.h5file, driver.params.out_group.uid
        )
        assert np.array([o is not np.nan for o in output["phi_d"]]).any()
        assert np.array([o is not np.nan for o in output["phi_m"]]).any()

        output["data"] = orig_dBzdt
        if pytest:
            check_target(output, target_run, tolerance=0.5)
            nan_ind = np.isnan(run_ws.get_entity("Iteration_0_model")[0].values)
            inactive_ind = run_ws.get_entity("active_cells")[0].values == 0
            assert np.all(nan_ind == inactive_ind)


if __name__ == "__main__":
    # Full run
    test_airborne_tem_fwr_run(Path("./"), n_grid_points=5, refinement=(0, 0, 4))
    test_airborne_tem_run(
        Path("./"),
        max_iterations=15,
        pytest=False,
    )
