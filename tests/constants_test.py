#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


from geoapps.inversion.electricals.direct_current.three_dimensions import (
    constants as direct_current_constants,
)
from geoapps.inversion.electricals.induced_polarization.three_dimensions import (
    constants as induced_polarization_constants,
)
from geoapps.inversion.potential_fields.gravity import constants as gravity_constants
from geoapps.inversion.potential_fields.magnetic_scalar import (
    constants as magnetic_scalar_constants,
)
from geoapps.inversion.potential_fields.magnetic_vector import (
    constants as magnetic_vector_constants,
)

constants = [
    gravity_constants,
    magnetic_scalar_constants,
    magnetic_vector_constants,
    direct_current_constants,
    induced_polarization_constants,
]


def test_deprecated_uijson_fields():
    deprecated_fields = ["default"]
    for constant in constants:
        d_u_j = constant.default_ui_json
        for value in d_u_j.values():
            if isinstance(value, dict):
                for field in deprecated_fields:
                    assert field not in value.keys()
