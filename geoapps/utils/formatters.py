#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


def string_name(value: str, characters: str = ".") -> str:
    """
    Find and replace characters in a string with underscores '_'.

    :param value: String to be validate
    :param char: Characters to be replaced

    :return value: Re-formatted string
    """
    for char in list(characters):
        value = value.replace(char, "_")
    return value
