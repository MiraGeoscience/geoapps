#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of my_app project.
#
#  All rights reserved.

from __future__ import annotations

import re
from pathlib import Path

import toml

import geoapps


def get_version():
    path = Path(__file__).resolve().parents[1] / "pyproject.toml"

    with open(str(path), encoding="utf-8") as file:
        pyproject = toml.loads(file.read())

    return pyproject["tool"]["poetry"]["version"]


def test_version_is_consistent():
    assert geoapps.__version__ == get_version()


def test_version_is_semver():
    semver_re = (
        r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
        r"(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
        r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
        r"(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
    )
    assert re.search(semver_re, geoapps.__version__) is not None
