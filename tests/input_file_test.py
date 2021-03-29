#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import json
import os

import pytest

from geoapps.io import InputFile


def test_create_work_path():
    input_file = InputFile("../assets/test.json")
    wpath = input_file.create_work_path()
    assert wpath == os.path.abspath("../assets") + os.path.sep


def test_load(tmp_path):
    fname = os.path.join(tmp_path, "test.json")
    input_dict = {"inversion_type": "mvi"}
    with open(fname, "w") as outfile:
        json.dump(input_dict, outfile)
    input_file = InputFile(fname)
    input_file.load()
    assert list(input_file.data.keys())[0] == "inversion_type"
    assert list(input_file.data.values())[0] == "mvi"


def test_filename_extension():
    with pytest.raises(IOError):
        InputFile("test.yaml")
