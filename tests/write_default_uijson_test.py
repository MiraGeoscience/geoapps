# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024 Mira Geoscience Ltd.                                     '
#                                                                              '
#  This file is part of geoapps.                                               '
#                                                                              '
#  geoapps is distributed under the terms and conditions of the MIT License    '
#  (see LICENSE file at the root of this source code package).                 '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from __future__ import annotations

from pathlib import Path

from geoh5py.ui_json import InputFile

from geoapps.inversion.potential_fields import GravityParams
from geoapps.utils.write_default_uijson import write_default_uijson


def test_write_default_uijson(tmp_path: Path):
    write_default_uijson(tmp_path)
    filepath = tmp_path / "gravity_inversion.ui.json"
    assert filepath.is_file()
    ifile = InputFile.read_ui_json(filepath, validate=False)
    params = GravityParams(input_file=ifile, validate=False)
    assert params.gz_uncertainty == 1.0
