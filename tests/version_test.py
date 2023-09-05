#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of my_app project.
#
#  All rights reserved.

from __future__ import annotations

import pytest
from semver import Version

import geoapps
from geoapps import assets_path
from geoapps.octree_creation.constants import app_initializer as oct_init
from geoapps.octree_creation.driver import OctreeDriver
from geoapps.octree_creation.params import OctreeParams


def test_version_is_consistent(pyproject: dict[str]):
    assert Version.parse(geoapps.__version__) == Version.parse(
        pyproject["tool"]["poetry"]["version"]
    )


def test_version_is_semver():
    assert Version.is_valid(geoapps.__version__)


def test_input_file_version(tmp_path):
    oct_init["geoh5"] = str(assets_path() / "FlinFlon.geoh5")
    oct_init["Refinement A levels"] = "0, 0"
    oct_init["Refinement B levels"] = "0, 0"
    oct_init["Refinement B type"] = "radial"

    params = OctreeParams(**oct_init)
    app_version = Version.parse(geoapps.__version__)

    version_in_file = app_version.replace(minor=app_version.minor + 1)
    params.version = str(version_in_file)
    params.write_input_file("test.ui.json", tmp_path)

    with pytest.warns(
        UserWarning, match=f"Input file version '{version_in_file}' is ahead"
    ):
        OctreeDriver.start(tmp_path / "test.ui.json")

    if app_version.minor > 0:
        version_in_file = app_version.replace(minor=app_version.minor - 1)
    else:
        version_in_file = app_version.replace(major=app_version.major - 1)

    params.version = str(version_in_file)
    params.write_input_file("test.ui.json", tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        OctreeDriver.start(tmp_path / "test.ui.json")
