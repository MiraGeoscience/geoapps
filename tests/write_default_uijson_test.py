#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import os

from geoapps.io import InputFile
from geoapps.io.Gravity import GravityParams
from geoapps.io.write_default_uijson import write_default_uijson


def test_write_default_uijson(tmp_path):
    write_default_uijson(tmp_path)
    filepath = os.path.join(tmp_path, "gravity_inversion.ui.json")
    assert os.path.exists(filepath)
    ifile = InputFile(filepath)
    params = GravityParams(ifile, validate=False)
    assert params.gz_uncertainty == 1.0


def test_write_default_uijson_initializers(tmp_path):
    write_default_uijson(tmp_path, use_initializers=True)
    filepath = os.path.join(tmp_path, "gravity_inversion.ui.json")
    assert os.path.exists(filepath)
    ifile = InputFile(filepath)
    params = GravityParams(ifile, validate=False)
    assert params.gz_uncertainty == 0.05
