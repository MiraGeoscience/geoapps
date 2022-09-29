#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import json
import os
import uuid

from geoapps.inversion.electricals.direct_current.pseudo_three_dimensions.driver import (
    DirectCurrentPseudo3DDriver,
)


def test_line_files(tmp_path):
    filepath = os.path.join(tmp_path, "lookup.json")
    file_lines = {str(uuid.uuid4()): {"line_id": k} for k in range(1, 3)}
    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(file_lines, file)
    line_files = DirectCurrentPseudo3DDriver.line_files(os.path.dirname(filepath))
    assert line_files == {v["line_id"]: k for k, v in file_lines.items()}
