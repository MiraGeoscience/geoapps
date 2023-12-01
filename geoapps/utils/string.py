#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations


def string_2_list(string):
    """
    Convert a list of numbers separated by comma to a list of floats
    """
    return [float(val) for val in string.split(",") if len(val) > 0]
