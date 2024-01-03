#  Copyright (c) 2024 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).
#
#  This file is part of my_app project.
#
#  All rights reserved.

from __future__ import annotations

import warnings

import pytest
from geoh5py.ui_json.constants import default_ui_json
from geoh5py.ui_json.input_file import InputFile
from semver import Version

import geoapps
from geoapps import assets_path
from geoapps.driver_base.driver import BaseDriver
from geoapps.driver_base.params import BaseParams


def test_version_is_consistent(pyproject: dict[str]):
    assert Version.parse(geoapps.__version__) == Version.parse(
        pyproject["tool"]["poetry"]["version"]
    )


def test_version_is_semver():
    assert Version.is_valid(geoapps.__version__)


def test_input_file_version(tmp_path):
    ui_json = default_ui_json.copy()
    ui_json["version"] = "0.0.1"
    params = BaseParams(
        geoh5=str(assets_path() / "FlinFlon.geoh5"),
        input_file=InputFile(ui_json=ui_json),
    )
    app_version = Version.parse(geoapps.__version__)
    version_in_file = app_version.replace(minor=app_version.minor + 1)
    params.version = str(version_in_file)
    params.write_input_file("test.ui.json", tmp_path)

    class TestDriver(BaseDriver):
        _validations = None

        def run(self):
            pass

    with pytest.warns(
        UserWarning, match=f"Input file version '{version_in_file}' is ahead"
    ):
        TestDriver.start(tmp_path / "test.ui.json")

    if app_version.minor > 0:
        version_in_file = app_version.replace(minor=app_version.minor - 1)
    else:
        version_in_file = app_version.replace(major=app_version.major - 1)

    params.version = str(version_in_file)
    params.write_input_file("test.ui.json", tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        TestDriver.start(tmp_path / "test.ui.json")
