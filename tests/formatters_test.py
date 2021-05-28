#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


import pytest

from geoapps.utils.formatters import string_name


def test_string_name():

    chars = "!@#$%^&*().,"
    value = "H!e(l@l#o.W$o%r^l&d*"
    assert (
        string_name(value, characters=chars) == "H_e_l_l_o_W_o_r_l_d_"
    ), "string_name validator failed"
