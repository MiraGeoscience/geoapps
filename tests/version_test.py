#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of my_app project.
#
#  All rights reserved.

from __future__ import annotations

import re

import pytest

import geoapps
from geoapps import assets_path
from geoapps.octree_creation.constants import app_initializer as oct_init
from geoapps.octree_creation.driver import OctreeDriver
from geoapps.octree_creation.params import OctreeParams


def test_version_is_consistent(pyproject: dict[str]):
    assert geoapps.__version__ == pyproject["tool"]["poetry"]["version"]


def test_version_is_semver():
    semver_re = (
        r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
        r"(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
        r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
        r"(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
    )
    assert re.search(semver_re, geoapps.__version__) is not None


def test_input_file_version(tmp_path):
    oct_init["geoh5"] = str(assets_path() / "FlinFlon.geoh5")
    params = OctreeParams(**oct_init)
    params.version = "0.0.0"
    params.write_input_file("test.ui.json", tmp_path)

    with pytest.warns(
        UserWarning, match="Input file version '0.0.0' does not match the."
    ):
        OctreeDriver.start(tmp_path / "test.ui.json")
