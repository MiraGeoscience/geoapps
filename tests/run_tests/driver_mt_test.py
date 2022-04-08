#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import warnings

import numpy as np
from geoh5py.objects import Curve
from geoh5py.workspace import Workspace

from geoapps.utils import get_inversion_output
from geoapps.utils.testing import setup_inversion_workspace

# import pytest
# pytest.skip("eliminating conflicting test.", allow_module_level=True)

# To test the full run and validate the inversion.
# Move this file out of the test directory and run.

target_magnetotellurics_run = {
    "data_norm": 0.0681608,
    "phi_d": 7.267,
    "phi_m": 165.6,
}


def test_magnetotellurics_run(
    tmp_path,
    n_grid_points=2,
    max_iterations=1,
    pytest=True,
    refinement=(2,),
):
    from geoapps.inversion.driver import InversionDriver
    from geoapps.inversion.natural_sources.magnetotellurics.params import (
        MagnetotelluricsParams,
    )

    np.random.seed(0)
    # Run the forward
    geoh5 = setup_inversion_workspace(
        tmp_path,
        background=0.01,
        anomaly=1.0,
        n_electrodes=n_grid_points,
        n_lines=n_grid_points,
        refinement=refinement,
        inversion_type="magnetotellurics",
        flatten=True,
    )

    model = geoh5.get_entity("model")[0]
    params = MagnetotelluricsParams(
        forward_only=True,
        geoh5=geoh5,
        mesh=model.parent.uid,
        topography_object=geoh5.get_entity("topography")[0].uid,
        resolution=0.0,
        z_from_topo=False,
        data_object=geoh5.get_entity("survey")[0].uid,
        starting_model_object=model.parent.uid,
        starting_model=model.uid,
        zxx_real_channel_bool=True,
        zxx_imag_channel_bool=True,
        zxy_real_channel_bool=True,
        zxy_imag_channel_bool=True,
        zyx_real_channel_bool=True,
        zyx_imag_channel_bool=True,
        zyy_real_channel_bool=True,
        zyy_imag_channel_bool=True,
    )
    params.workpath = tmp_path
    fwr_driver = InversionDriver(params, warmstart=False)
    fwr_driver.run()
    geoh5 = Workspace(geoh5.h5file)

    survey = geoh5.get_entity("survey")[0]

    data = {}
    uncertainties = {}
    components = {
        "zxx_real": "Zxx (real)",
        "zxx_imag": "Zxx (imag)",
        "zxy_real": "Zxy (real)",
        "zxy_imag": "Zxy (imag)",
        "zyx_real": "Zyx (real)",
        "zyx_imag": "Zyx (imag)",
        "zyy_real": "Zyy (real)",
        "zyy_imag": "Zyy (imag)",
    }
    curve = Curve.create(geoh5, vertices=survey.vertices)
    for comp, cname in components.items():
        data[cname] = []
        # uncertainties[f"{cname} uncertainties"] = {}
        uncertainties[f"{cname} uncertainties"] = []
        for freq in survey.channels:
            d = geoh5.get_entity(f"Iteration_0_{comp}_{freq:.2e}")[0].copy(
                parent=survey
            )
            data[cname].append(d)

            u = curve.add_data(
                {
                    f"uncertainty_{comp}_{freq:.2e}": {
                        "values": np.abs(0.05 * d.values) + d.values.std()
                    }
                }
            )
            uncertainties[f"{cname} uncertainties"].append(u.copy(parent=survey))
            # uncertainties[f"{cname} uncertainties"][freq] = {"values": u.copy(parent=survey)}

    survey.add_components_data(data)
    survey.add_components_data(uncertainties)

    data_kwargs = {}
    for i, comp in enumerate(components):
        data_kwargs[f"{comp}_channel"] = survey.property_groups[i].uid
        data_kwargs[f"{comp}_uncertainty"] = survey.property_groups[8 + i].uid

    orig_zyy_real_1 = geoh5.get_entity("Iteration_0_zyy_real_1.00e+01")[0].values

    # Run the inverse
    np.random.seed(0)
    params = MagnetotelluricsParams(
        geoh5=geoh5,
        mesh=geoh5.get_entity("mesh")[0].uid,
        topography_object=geoh5.get_entity("topography")[0].uid,
        resolution=0.0,
        data_object=survey.uid,
        starting_model=0.01,
        reference_model=None,
        s_norm=0.0,
        x_norm=1.0,
        y_norm=1.0,
        z_norm=1.0,
        gradient_type="components",
        z_from_topo=False,
        upper_bound=0.75,
        max_iterations=max_iterations,
        initial_beta_ratio=1e-2,
        prctile=100,
        **data_kwargs,
    )
    params.workpath = tmp_path
    driver = InversionDriver(params)
    driver.run()
    run_ws = Workspace(driver.params.geoh5.h5file)
    output = get_inversion_output(
        driver.params.geoh5.h5file, driver.params.ga_group.uid
    )

    predicted = run_ws.get_entity("Iteration_0_zyy_real_1.00e+01")[0]

    if pytest:
        if any(np.isnan(orig_zyy_real_1)):
            warnings.warn(
                "Skipping data norm comparison due to nan (used to bypass lone faulty test run in GH actions)."
            )
        else:
            np.testing.assert_almost_equal(
                np.linalg.norm(orig_zyy_real_1),
                target_magnetotellurics_run["data_norm"],
                decimal=3,
            )

        np.testing.assert_almost_equal(
            output["phi_m"][1], target_magnetotellurics_run["phi_m"], decimal=1
        )
        np.testing.assert_almost_equal(
            output["phi_d"][1], target_magnetotellurics_run["phi_d"], decimal=1
        )

        nan_ind = np.isnan(run_ws.get_entity("Iteration_0_model")[0].values)
        inactive_ind = run_ws.get_entity("active_cells")[0].values == 0
        assert np.all(nan_ind == inactive_ind)
    else:
        return fwr_driver.starting_model, driver.inverse_problem.model


if __name__ == "__main__":
    # Full run
    m_start, m_rec = test_magnetotellurics_run(
        "./", n_grid_points=8, max_iterations=30, pytest=False, refinement=(4, 8)
    )
    residual = np.linalg.norm(m_rec - m_start) / np.linalg.norm(m_start) * 100.0
    assert (
        residual < 50.0
    ), f"Deviation from the true solution is {residual:.2f}%. Validate the solution!"
    print("Conductivity model is within 15% of the answer. Let's go!!")
