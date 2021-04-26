#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import os

from geoapps.io import InputFile
from geoapps.io.constants import default_ui_json

path = "./test.ui.json"
ifile = InputFile(path)
ifile.write_ui_json()


def test_ui_json_io(tmp_path):

    path = os.path.join(tmp_path, "test.ui.json")
    ifile = InputFile(path)
    ifile.write_ui_json()
    ifile.read_ui_json()
    ifile.data == default_ui_json
