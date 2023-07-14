#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).
from pathlib import Path

import numpy as np
from geoh5py.groups import SimPEGGroup
from geoh5py.objects import Octree
from geoh5py.workspace import Workspace
from SimPEG.maps import IdentityMap

from geoapps.inversion.joint.joint_cross_gradient import JointCrossGradientParams
from geoapps.inversion.joint.joint_cross_gradient.driver import JointCrossGradientDriver
from geoapps.inversion.potential_fields import (
    GravityParams,
    MagneticScalarParams,
    MagneticVectorParams,
)
from geoapps.inversion.potential_fields.gravity.driver import GravityDriver
from geoapps.inversion.potential_fields.magnetic_scalar.driver import (
    MagneticScalarDriver,
)
from geoapps.inversion.potential_fields.magnetic_vector.driver import (
    MagneticVectorDriver,
)
from geoapps.shared_utils.utils import get_inversion_output
from geoapps.utils.testing import check_target, setup_inversion_workspace

# To test the full run and validate the inversion.
# Move this file out of the test directory and run.

target_run = {
    "data_norm": 51.20747,
    "phi_d": 2061,
    "phi_m": 0.02767,
}


def test_joint_cross_gradient_fwr_run(
    tmp_path,
    n_grid_points=4,
    refinement=(2,),
):
    np.random.seed(0)
    # Create local problem A
    geoh5, _, model, survey, topography = setup_inversion_workspace(
        tmp_path,
        background=0.0,
        anomaly=0.75,
        refinement=refinement,
        n_electrodes=n_grid_points,
        n_lines=n_grid_points,
    )
    params = GravityParams(
        forward_only=True,
        geoh5=geoh5,
        mesh=model.parent.uid,
        topography_object=topography.uid,
        resolution=0.0,
        z_from_topo=False,
        data_object=survey.uid,
        starting_model=model.uid,
    )
    fwr_driver_a = GravityDriver(params)

    _, _, model, survey, _ = setup_inversion_workspace(
        tmp_path,
        background=0.0,
        anomaly=0.05,
        refinement=refinement,
        n_electrodes=n_grid_points,
        n_lines=n_grid_points,
        flatten=False,
    )
    inducing_field = (50000.0, 90.0, 0.0)
    params = MagneticVectorParams(
        forward_only=True,
        geoh5=geoh5,
        mesh=model.parent.uid,
        topography_object=topography.uid,
        inducing_field_strength=inducing_field[0],
        inducing_field_inclination=inducing_field[1],
        inducing_field_declination=inducing_field[2],
        resolution=0.0,
        z_from_topo=False,
        data_object=survey.uid,
        starting_model=model.uid,
    )
    params.workpath = tmp_path
    fwr_driver_b = MagneticVectorDriver(params)

    # Force co-location of meshes
    fwr_driver_b.inversion_mesh.entity.origin = (
        fwr_driver_a.inversion_mesh.entity.origin
    )
    fwr_driver_b.workspace.update_attribute(
        fwr_driver_b.inversion_mesh.entity, "attributes"
    )
    fwr_driver_b.inversion_mesh._mesh = None  # pylint: disable=protected-access

    fwr_driver_a.run()
    fwr_driver_b.run()

    vector_model = fwr_driver_b.directives.save_directives[0].transforms[0](
        fwr_driver_b.models.starting
    )
    vector_model = (
        fwr_driver_b.directives.save_directives[0].transforms[1] * vector_model
    )

    geoh5.close()
    return np.r_[
        fwr_driver_a.directives.save_directives[0].transforms[0]
        * fwr_driver_a.models.starting,
        vector_model.flatten(),
    ]


def test_joint_cross_gradient_inv_run(
    tmp_path,
    max_iterations=1,
    pytest=True,
):
    workpath = tmp_path / "inversion_test.ui.geoh5"
    if pytest:
        workpath = (
            tmp_path.parent
            / "test_joint_cross_gradient_fwr_0"
            / "inversion_test.ui.geoh5"
        )

    with Workspace(workpath) as geoh5:
        topography = geoh5.get_entity("topography")[0]
        drivers = []
        orig_data = []

        for group_name in ["Gravity Forward", "Magnetic vector Forward"]:
            group = geoh5.get_entity(group_name)[0]

            if not isinstance(group, SimPEGGroup):
                continue

            for child in group.children:
                if isinstance(child, Octree):
                    mesh = child
                else:
                    survey = child

            data = survey.children[0]
            orig_data.append(data.values)

            if group.options["inversion_type"] == "gravity":
                params = GravityParams(
                    geoh5=geoh5,
                    mesh=mesh.uid,
                    topography_object=topography.uid,
                    data_object=survey.uid,
                    gz_channel=data.uid,
                    gz_uncertainty=1e-3,
                    starting_model=0.0,
                )
                drivers.append(GravityDriver(params))
            else:
                params = MagneticVectorParams(
                    geoh5=geoh5,
                    mesh=mesh.uid,
                    topography_object=topography.uid,
                    inducing_field_strength=group.options["inducing_field_strength"][
                        "value"
                    ],
                    inducing_field_inclination=group.options[
                        "inducing_field_inclination"
                    ]["value"],
                    inducing_field_declination=group.options[
                        "inducing_field_declination"
                    ]["value"],
                    data_object=survey.uid,
                    starting_model=1e-4,
                    reference_model=0.0,
                    tile_spatial=2,
                    tmi_channel=data.uid,
                    tmi_uncertainty=2.0,
                )
                drivers.append(MagneticVectorDriver(params))
                params = MagneticScalarParams(
                    geoh5=geoh5,
                    mesh=mesh.uid,
                    topography_object=topography.uid,
                    inducing_field_strength=group.options["inducing_field_strength"][
                        "value"
                    ],
                    inducing_field_inclination=group.options[
                        "inducing_field_inclination"
                    ]["value"],
                    inducing_field_declination=group.options[
                        "inducing_field_declination"
                    ]["value"],
                    data_object=survey.uid,
                    starting_model=1e-4,
                    reference_model=0.0,
                    alpha_s=0.0,
                    lower_bound=0.0,
                    tile_spatial=2,
                    tmi_channel=data.uid,
                    tmi_uncertainty=2.0,
                )
                drivers.append(MagneticScalarDriver(params))

        # Run the inverse
        np.random.seed(0)
        joint_params = JointCrossGradientParams(
            geoh5=geoh5,
            topography_object=topography.uid,
            group_a=drivers[0].params.out_group,
            group_b=drivers[1].params.out_group,
            group_c=drivers[2].params.out_group,
            max_global_iterations=max_iterations,
            initial_beta_ratio=1e0,
            cross_gradient_weight_a_b=1e3,
            s_norm=0.0,
            x_norm=0.0,
            y_norm=0.0,
            z_norm=0.0,
            gradient_type="components",
            prctile=100,
            store_sensitivities="ram",
        )

    driver = JointCrossGradientDriver(joint_params)
    driver.run()

    with Workspace(driver.params.geoh5.h5file):
        output = get_inversion_output(
            driver.params.geoh5.h5file, driver.params.out_group.uid
        )

        output["data"] = np.hstack(orig_data)
        if pytest:
            check_target(output, target_run)
        else:
            out_model = []
            for sub_driver in driver.drivers:
                save_directive = sub_driver.directives.directive_list[0]

                model = driver.inverse_problem.model
                for fun in save_directive.transforms:
                    if isinstance(fun, (IdentityMap, np.ndarray, float)):
                        model = fun * model
                    else:
                        model = fun(model)

                out_model.append(model.flatten())

            return np.hstack(out_model)


if __name__ == "__main__":
    # Full run
    m_start = test_joint_cross_gradient_fwr_run(
        Path("./"),
        n_grid_points=20,
        refinement=(4, 8),
    )

    m_rec = test_joint_cross_gradient_inv_run(
        Path("./"),
        max_iterations=20,
        pytest=False,
    )
    nC = int(m_rec.size / 4)
    model_residual = (
        np.nansum((m_rec[: 2 * nC] - m_start[: 2 * nC]) ** 2.0) ** 0.5
        / np.nansum(m_start[: 2 * nC] ** 2.0) ** 0.5
        * 100.0
    )
    assert (
        model_residual < 75.0
    ), f"Deviation from the true solution is {model_residual:.2f}%. Validate the solution!"
    print("Recovered model is within 15% of the answer. Let's go!!")
