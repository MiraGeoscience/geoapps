#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import numpy as np
import pytest

from geoapps.utils import rotate_xy


def test_rotation_xy():
    vec = np.c_[1, 0, 0]

    rot_vec = rotate_xy(vec, [0, 0], 45)

    assert (
        np.norm(np.cross(rot_vec, [0.7071, 0.7071, 0])) < 1e-8
    ), "Error on positive rotation about origin."

    rot_vec = rotate_xy(vec, [1, 1], -90)

    assert (
        np.norm(np.cross(rot_vec, [0, 1, 0])) < 1e-8
    ), "Error on negative rotation about point."
