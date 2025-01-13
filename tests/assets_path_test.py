# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024-2025 Mira Geoscience Ltd.                                '
#                                                                              '
#  This file is part of geoapps.                                               '
#                                                                              '
#  geoapps is distributed under the terms and conditions of the MIT License    '
#  (see LICENSE file at the root of this source code package).                 '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from __future__ import annotations

from pathlib import Path

from geoapps import assets_path


def test_assets_directory_exist():
    assert assets_path().is_dir()


def test_assets_directory_from_env(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GEOAPPS_ASSETS_DIR", str(tmp_path.absolute()))

    assert assets_path().is_dir()
    assert tmp_path == assets_path()


def test_assets_directory_from_wrong_env(tmp_path: Path, monkeypatch):
    non_existing_path = tmp_path / "wrong"
    monkeypatch.setenv("GEOAPPS_ASSETS_DIR", str(non_existing_path.absolute()))

    assert non_existing_path.is_dir() is False
    assert assets_path().is_dir() is True
    assert non_existing_path != assets_path()


def test_uijson_files_exists():
    assert (assets_path() / "uijson").is_dir()
    assert list((assets_path() / "uijson").iterdir())[0].is_file()
