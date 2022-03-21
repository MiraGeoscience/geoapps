#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import os

from geoh5py.ui_json import InputFile

from geoapps.applications.simpeg_inversions.gravity.params import GravityParams
from geoapps.utils.write_default_uijson import write_default_uijson


def test_write_default_uijson(tmp_path):
    write_default_uijson(tmp_path)
    filepath = os.path.join(tmp_path, "gravity_inversion.ui.json")
    assert os.path.exists(filepath)
    ifile = InputFile.read_ui_json(filepath, validation_options={"disabled": True})
    params = GravityParams(input_file=ifile, validate=False)
    assert params.gz_uncertainty == 1.0
