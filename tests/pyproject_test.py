#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations


def test_pyproject_extra_apps_include_core(pyproject: dict[str]):
    """Test that the list of extra packages for "apps" include the list for "core"
    """

    extras = pyproject["tool"]["poetry"]["extras"]
    core_extras = extras["core"]
    apps_extras = extras["apps"]
    core_in_apps = set(core_extras).intersection(apps_extras)
    assert core_in_apps == set(core_extras)
