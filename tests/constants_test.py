#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


from geoapps.inversion.electric.direct_current import (
    constants as direct_current_constants,
)
from geoapps.inversion.electric.induced_polarization import (
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
    for c in constants:
        d_u_j = c.default_ui_json
        for k, v in d_u_j.items():
            if isinstance(v, dict):
                for f in deprecated_fields:
                    assert f not in v.keys()
