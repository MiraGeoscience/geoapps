#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import os

from geoapps.io import write_default_uijson


def test_write_default_uijson(tmp_path):
    write_default_uijson(tmp_path)
    assert os.path.exists(os.path.join(tmp_path, "gravity_inversion.ui.json"))
