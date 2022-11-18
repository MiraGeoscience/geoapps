#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

# isort: skip_file

from __future__ import annotations
from geoapps.inversion.params import InversionBaseParams  # isort: skip
from geoapps.inversion.constants import default_ui_json
from geoapps.inversion.electricals.direct_current.three_dimensions.driver import (
    DirectCurrent3DDriver,
)
from geoapps.inversion.electricals.direct_current.two_dimensions.driver import (
    DirectCurrent2DDriver,
)
from geoapps.inversion.electricals.induced_polarization.three_dimensions.driver import (
    InducedPolarization3DDriver,
)
from geoapps.inversion.electricals.induced_polarization.two_dimensions.driver import (
    InducedPolarization2DDriver,
)
from geoapps.inversion.natural_sources.magnetotellurics.driver import (
    MagnetotelluricsDriver,
)
from geoapps.inversion.natural_sources.tipper.driver import TipperDriver

from geoapps.inversion.potential_fields.gravity.driver import GravityDriver
from geoapps.inversion.potential_fields.magnetic_scalar.driver import (
    MagneticScalarDriver,
)
from geoapps.inversion.potential_fields.magnetic_vector.driver import (
    MagneticVectorDriver,
)

DRIVER_MAP = {
    "gravity": GravityDriver,
    "magnetic scalar": MagneticScalarDriver,
    "magnetic vector": MagneticVectorDriver,
    "direct current 3d": DirectCurrent3DDriver,
    "direct current 2d": DirectCurrent2DDriver,
    "induced polarization 3d": InducedPolarization3DDriver,
    "induced polarization 2d": InducedPolarization2DDriver,
    "induced polarization pseudo 3d": None,  # InducedPolarizationPseudo3DDriver,
    "magnetotellurics": MagnetotelluricsDriver,
    "tipper": TipperDriver,
}
