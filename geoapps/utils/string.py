#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import numpy as np


def string_2_list(string):
    """
    Convert a list of numbers separated by comma to a list of floats
    """
    return [float(val) for val in string.split(",") if len(val) > 0]


def string_to_numeric(text: str) -> int | float | str:
    """Converts numeric string representation to int or string if possible."""
    try:
        text_as_float = float(text)
        text_as_int = int(text_as_float)
        return text_as_int if text_as_int == text_as_float else text_as_float
    except ValueError:
        return np.nan if text == "nan" else text
