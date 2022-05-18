#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import os
import re


def existing_file_incrementer(filename):
    """Add an increment of the form (n) to the filename."""

    if os.path.exists(filename):
        root, file = filename.rsplit(os.path.sep, 1)
        name, extension = file.split(".", 1)
        incremented = re.search(r"\(([0-9]+)\)", name)
        if incremented is None:
            filename = os.path.join(root, f"{name} (1).{extension}")
        else:
            increment = incremented.group(0)
            name = name.replace(increment, f"({int(incremented.group(1)) + 1})")
            filename = os.path.join(root, f"{name}.{extension}")

    return filename
