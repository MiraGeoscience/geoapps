# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024 Mira Geoscience Ltd.                                     '
#                                                                              '
#  This file is part of geoapps.                                               '
#                                                                              '
#  geoapps is distributed under the terms and conditions of the MIT License    '
#  (see LICENSE file at the root of this source code package).                 '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from __future__ import annotations

import argparse
from pathlib import Path

from octree_creation_app.params import OctreeParams
from simpeg_drivers.electricals.direct_current.pseudo_three_dimensions.params import (
    DirectCurrentPseudo3DParams,
)
from simpeg_drivers.electricals.direct_current.three_dimensions import (
    DirectCurrent3DParams,
)
from simpeg_drivers.electricals.direct_current.two_dimensions import (
    DirectCurrent2DParams,
)
from simpeg_drivers.electricals.induced_polarization.pseudo_three_dimensions.params import (
    InducedPolarizationPseudo3DParams,
)
from simpeg_drivers.electricals.induced_polarization.three_dimensions import (
    InducedPolarization3DParams,
)
from simpeg_drivers.electricals.induced_polarization.two_dimensions import (
    InducedPolarization2DParams,
)
from simpeg_drivers.electromagnetics.frequency_domain import (
    FrequencyDomainElectromagneticsParams,
)
from simpeg_drivers.electromagnetics.time_domain import TimeDomainElectromagneticsParams
from simpeg_drivers.joint.joint_cross_gradient import JointCrossGradientParams
from simpeg_drivers.joint.joint_surveys import JointSurveysParams
from simpeg_drivers.natural_sources import MagnetotelluricsParams, TipperParams
from simpeg_drivers.potential_fields import (
    GravityParams,
    MagneticScalarParams,
    MagneticVectorParams,
)

from geoapps import assets_path
from geoapps.block_model_creation.params import BlockModelParams
from geoapps.clustering.params import ClusteringParams
from geoapps.interpolation.params import DataInterpolationParams
from geoapps.iso_surfaces.params import IsoSurfacesParams
from geoapps.peak_finder.params import PeakFinderParams
from geoapps.scatter_plot.params import ScatterPlotParams

active_data_channels = [
    "z_real_channel",
    "z_imag_channel",
    "zxx_real_channel",
    "zxx_imag_channel",
    "zxy_real_channel",
    "zxy_imag_channel",
    "zyx_real_channel",
    "zyx_imag_channel",
    "zyy_real_channel",
    "zyy_imag_channel",
    "txz_real_channel",
    "txz_imag_channel",
    "tyz_real_channel",
    "tyz_imag_channel",
    "gz_channel",
    "tmi_channel",
    "z_channel",
]


def write_default_uijson(path: str | Path, use_initializers=False):
    from simpeg_drivers.potential_fields.gravity.constants import (
        app_initializer as grav_init,
    )

    grav_init["geoh5"] = str(assets_path() / "FlinFlon.geoh5")
    grav_init = grav_init if use_initializers else {}

    from simpeg_drivers.potential_fields.magnetic_scalar.constants import (
        app_initializer as mag_init,
    )

    mag_init["geoh5"] = str(assets_path() / "FlinFlon.geoh5")
    mag_init = mag_init if use_initializers else {}

    from geoapps.inversion.potential_fields.magnetic_vector.constants import (
        app_initializer as mvi_init,
    )

    mvi_init["geoh5"] = str(assets_path() / "FlinFlon.geoh5")
    mvi_init = mvi_init if use_initializers else {}

    from geoapps.inversion.electricals.direct_current.three_dimensions.constants import (
        app_initializer as dc_3d_init,
    )

    dc_3d_init["geoh5"] = str(assets_path() / "FlinFlon_dcip.geoh5")
    dc_3d_init = dc_3d_init if use_initializers else {}

    from geoapps.inversion.electricals.direct_current.two_dimensions.constants import (
        app_initializer as dc_2d_init,
    )

    dc_2d_init["geoh5"] = str(assets_path() / "FlinFlon_dcip.geoh5")
    dc_2d_init = dc_2d_init if use_initializers else {}

    from geoapps.inversion.electricals.direct_current.pseudo_three_dimensions.constants import (
        app_initializer as dc_p3d_init,
    )

    dc_p3d_init["geoh5"] = str(assets_path() / "FlinFlon_dcip.geoh5")
    dc_p3d_init = dc_p3d_init if use_initializers else {}

    from geoapps.inversion.electricals.induced_polarization.three_dimensions.constants import (
        app_initializer as ip_3d_init,
    )

    ip_3d_init["geoh5"] = str(assets_path() / "FlinFlon_dcip.geoh5")
    ip_3d_init = ip_3d_init if use_initializers else {}

    from geoapps.inversion.electricals.induced_polarization.two_dimensions.constants import (
        app_initializer as ip_2d_init,
    )

    ip_2d_init["geoh5"] = str(assets_path() / "FlinFlon_dcip.geoh5")
    ip_2d_init = ip_2d_init if use_initializers else {}

    from geoapps.inversion.electricals.induced_polarization.pseudo_three_dimensions.constants import (
        app_initializer as ip_p3d_init,
    )

    ip_p3d_init["geoh5"] = str(assets_path() / "FlinFlon_dcip.geoh5")
    ip_p3d_init = ip_p3d_init if use_initializers else {}

    from geoapps.inversion.electromagnetics.frequency_domain.constants import (
        app_initializer as fem_init,
    )

    fem_init["geoh5"] = str(assets_path() / "FlinFlon_natural_sources.geoh5")
    fem_init = fem_init if use_initializers else {}

    from geoapps.inversion.electromagnetics.time_domain.constants import (
        app_initializer as tdem_init,
    )
    from geoapps.inversion.natural_sources.magnetotellurics.constants import (
        app_initializer as mt_init,
    )

    mt_init["geoh5"] = str(assets_path() / "FlinFlon_natural_sources.geoh5")
    mt_init = mt_init if use_initializers else {}

    from geoapps.inversion.natural_sources.tipper.constants import (
        app_initializer as tipper_init,
    )

    tipper_init["geoh5"] = str(assets_path() / "FlinFlon_natural_sources.geoh5")
    tipper_init = tipper_init if use_initializers else {}

    from octree_creation_app.constants import template_dict

    from geoapps.octree_creation.constants import app_initializer as oct_init

    if use_initializers:
        oct_init["geoh5"] = str(assets_path() / "FlinFlon.geoh5")
        # Add refinements
    else:
        oct_init = {}
        for label in "ABC":
            name = f"Refinement {label}"
            for label, form in template_dict.items():
                oct_init[f"{name} {label}"] = form["value"]

    from geoapps.scatter_plot.constants import app_initializer as scatter_init

    scatter_init["geoh5"] = str(assets_path() / "FlinFlon.geoh5")
    scatter_init = scatter_init if use_initializers else {}

    from geoapps.interpolation.constants import app_initializer as interp_init

    interp_init["geoh5"] = str(assets_path() / "FlinFlon.geoh5")
    interp_init = interp_init if use_initializers else {}

    from geoapps.block_model_creation.constants import app_initializer as block_init

    block_init["geoh5"] = str(assets_path() / "FlinFlon.geoh5")
    block_init = block_init if use_initializers else {}

    from geoapps.clustering.constants import app_initializer as cluster_init

    cluster_init["geoh5"] = str(assets_path() / "FlinFlon.geoh5")
    cluster_init = cluster_init if use_initializers else {}

    from geoapps.peak_finder.constants import app_initializer as peak_init

    peak_init["geoh5"] = str(assets_path() / "FlinFlon.geoh5")
    peak_init = peak_init if use_initializers else {}

    from geoapps.iso_surfaces.constants import app_initializer as iso_init

    iso_init["geoh5"] = str(assets_path() / "FlinFlon.geoh5")
    iso_init = iso_init if use_initializers else {}

    from geoapps.inversion.joint.joint_surveys.constants import (
        app_initializer as joint_surveys_init,
    )

    joint_surveys_init["geoh5"] = str(assets_path() / "FlinFlon.geoh5")
    joint_surveys_init = joint_surveys_init if use_initializers else {}

    from geoapps.inversion.joint.joint_cross_gradient.constants import (
        app_initializer as joint_cross_gradient_init,
    )

    joint_cross_gradient_init["geoh5"] = str(assets_path() / "FlinFlon.geoh5")
    joint_cross_gradient_init = joint_cross_gradient_init if use_initializers else {}

    filedict = {
        "gravity_inversion.ui.json": GravityParams(validate=False, **grav_init),
        "gravity_forward.ui.json": GravityParams(forward_only=True, validate=False),
        "magnetic_scalar_inversion.ui.json": MagneticScalarParams(
            validate=False, **mag_init
        ),
        "magnetic_scalar_forward.ui.json": MagneticScalarParams(
            forward_only=True, validate=False
        ),
        "magnetic_vector_inversion.ui.json": MagneticVectorParams(
            validate=False, **mvi_init
        ),
        "magnetic_vector_forward.ui.json": MagneticVectorParams(
            forward_only=True, validate=False
        ),
        "direct_current_inversion_2d.ui.json": DirectCurrent2DParams(
            validate=False, **dc_2d_init
        ),
        "direct_current_forward_2d.ui.json": DirectCurrent2DParams(
            forward_only=True, validate=False
        ),
        "direct_current_inversion_3d.ui.json": DirectCurrent3DParams(
            validate=False, **dc_3d_init
        ),
        "direct_current_forward_3d.ui.json": DirectCurrent3DParams(
            forward_only=True, validate=False
        ),
        "direct_current_inversion_pseudo3d.ui.json": DirectCurrentPseudo3DParams(
            validate=False, **dc_p3d_init
        ),
        "direct_current_forward_pseudo3d.ui.json": DirectCurrentPseudo3DParams(
            forward_only=True, validate=False
        ),
        "induced_polarization_inversion_2d.ui.json": InducedPolarization2DParams(
            validate=False, **ip_2d_init
        ),
        "induced_polarization_forward_2d.ui.json": InducedPolarization2DParams(
            forward_only=True, validate=False
        ),
        "induced_polarization_inversion_3d.ui.json": InducedPolarization3DParams(
            validate=False, **ip_3d_init
        ),
        "induced_polarization_forward_3d.ui.json": InducedPolarization3DParams(
            forward_only=True, validate=False
        ),
        "induced_polarization_inversion_pseudo3d.ui.json": InducedPolarizationPseudo3DParams(
            validate=False, **ip_p3d_init
        ),
        "induced_polarization_forward_pseudo3d.ui.json": InducedPolarizationPseudo3DParams(
            forward_only=True, validate=False
        ),
        "fem_inversion.ui.json": FrequencyDomainElectromagneticsParams(
            forward_only=False, validate=False, **fem_init
        ),
        "fem_forward.ui.json": FrequencyDomainElectromagneticsParams(
            forward_only=True, validate=False
        ),
        "tdem_inversion.ui.json": TimeDomainElectromagneticsParams(
            forward_only=False, validate=False, **tdem_init
        ),
        "tdem_forward.ui.json": TimeDomainElectromagneticsParams(
            forward_only=True, validate=False
        ),
        "magnetotellurics_inversion.ui.json": MagnetotelluricsParams(
            forward_only=False, validate=False, **mt_init
        ),
        "magnetotellurics_forward.ui.json": MagnetotelluricsParams(
            forward_only=True, validate=False
        ),
        "tipper_inversion.ui.json": TipperParams(
            forward_only=False, validate=False, **tipper_init
        ),
        "tipper_forward.ui.json": TipperParams(forward_only=True, validate=False),
        "joint_surveys_inversion.ui.json": JointSurveysParams(
            forward_only=False, validate=False, **joint_surveys_init
        ),
        "joint_cross_gradient_inversion.ui.json": JointCrossGradientParams(
            forward_only=False, validate=False, **joint_cross_gradient_init
        ),
        "octree_mesh.ui.json": OctreeParams(validate=False, **oct_init),
        "peak_finder.ui.json": PeakFinderParams(validate=False, **peak_init),
        "scatter.ui.json": ScatterPlotParams(validate=False, **scatter_init),
        "interpolation.ui.json": DataInterpolationParams(validate=False, **interp_init),
        "block_model_creation.ui.json": BlockModelParams(validate=False, **block_init),
        "cluster.ui.json": ClusteringParams(validate=False, **cluster_init),
        "iso_surfaces.ui.json": IsoSurfacesParams(validate=False, **iso_init),
    }

    for filename, params in filedict.items():
        validation_options = {
            "update_enabled": (True if params.geoh5 is not None else False)
        }
        params.input_file.validation_options = validation_options
        if hasattr(params, "forward_only"):
            if params.forward_only:
                for form in params.input_file.ui_json.values():
                    if isinstance(form, dict):
                        group = form.get("group", None)
                        if group == "Data":
                            form["group"] = "Survey"
                for param in [
                    "starting_model",
                    "starting_inclination",
                    "starting_declination",
                ]:
                    if param in params.input_file.ui_json:
                        form = params.input_file.ui_json[param]
                        form["label"] = (
                            form["label"].replace("Initial ", "").capitalize()
                        )
            elif params.data_object is None:
                for channel in active_data_channels:
                    form = params.input_file.ui_json.get(channel, None)
                    if form:
                        form["enabled"] = True

        params.write_input_file(name=filename, path=path, validate=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write defaulted ui.json files.")
    parser.add_argument(
        "path",
        type=Path,
        help="Path to folder where default ui.json files will be written.",
    )
    parser.add_argument(
        "--use_initializers",
        help="Write files initialized with FlinFlon values.",
        action="store_true",
    )
    args = parser.parse_args()
    write_default_uijson(args.path, args.use_initializers)
