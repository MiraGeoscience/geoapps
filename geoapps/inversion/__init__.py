#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

# isort: skip_file

from __future__ import annotations

from SimPEG import dask

from geoapps.inversion.params import InversionBaseParams  # isort: skip
from geoapps.inversion.constants import default_ui_json


DRIVER_MAP = {
    "direct current 3d": (
        "geoapps.inversion.electricals.direct_current.three_dimensions.driver",
        "DirectCurrent3DDriver",
    ),
    "direct current 2d": (
        "geoapps.inversion.electricals.direct_current.two_dimensions.driver",
        "DirectCurrent2DDriver",
    ),
    "direct current pseudo 3d": (
        "geoapps.inversion.electricals.direct_current.pseudo_three_dimensions.driver",
        "DirectCurrentPseudo3DDriver",
    ),
    "induced polarization 3d": (
        "geoapps.inversion.electricals.induced_polarization.three_dimensions.driver",
        "InducedPolarization3DDriver",
    ),
    "induced polarization 2d": (
        "geoapps.inversion.electricals.induced_polarization.two_dimensions.driver",
        "InducedPolarization2DDriver",
    ),
    "induced polarization pseudo 3d": (
        "geoapps.inversion.electricals.induced_polarization.pseudo_three_dimensions.driver",
        "InducedPolarizationPseudo3DDriver",
    ),
    "joint surveys": (
        "geoapps.inversion.joint.joint_surveys.driver",
        "JointSurveyDriver",
    ),
    "tdem": (
        "geoapps.inversion.electromagnetics.time_domain.driver",
        "TimeDomainElectromagneticsDriver",
    ),
    "magnetotellurics": (
        "geoapps.inversion.natural_sources.magnetotellurics.driver",
        "MagnetotelluricsDriver",
    ),
    "tipper": ("geoapps.inversion.natural_sources.tipper.driver", "TipperDriver"),
    "gravity": ("geoapps.inversion.potential_fields.gravity.driver", "GravityDriver"),
    "magnetic scalar": (
        "geoapps.inversion.potential_fields.magnetic_scalar.driver",
        "MagneticScalarDriver",
    ),
    "magnetic vector": (
        "geoapps.inversion.potential_fields.magnetic_vector.driver",
        "MagneticVectorDriver",
    ),
}
