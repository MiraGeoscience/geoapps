#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


from geoapps.io.DirectCurrent import constants as direct_current_constants
from geoapps.io.Gravity import constants as gravity_constants
from geoapps.io.InducedPolarization import constants as induced_polarization_constants
from geoapps.io.MagneticScalar import constants as magnetic_scalar_constants
from geoapps.io.MagneticVector import constants as magnetic_vector_constants

constants = [
    gravity_constants,
    magnetic_scalar_constants,
    magnetic_vector_constants,
    direct_current_constants,
    induced_polarization_constants,
]


def test_deprecated_uijson_fields():
    deprecated_fields = ["default"]
    for c in constants:
        d_u_j = c.default_ui_json
        for k, v in d_u_j.items():
            if isinstance(v, dict):
                for f in deprecated_fields:
                    assert f not in v.keys()
