#!/usr/bin/env python3

#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

"""
Creates cross-platform lock files for each python version and per-platform conda environment files.

Cross-platform lock files are created at the root of the project.
Per-platform conda environment files with and without dev dependencies, are placed under the `environments` sub-folder.

Usage: from the conda base environment, at the root of the project:
> python devtools/run_conda_lock.py

To prepare the conda base environment, see devtools/setup-conda-base.bat
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from add_url_tag_sha256 import patch_pyproject_toml

env_file_variables_section_ = """
variables:
  KMP_WARNINGS: 0
"""

_environments_folder = Path("environments")

_python_versions = ["3.10", "3.9"]


@contextmanager
def print_execution_time(name: str = "") -> Generator:
    from datetime import datetime

    start = datetime.now()
    try:
        yield
    finally:
        duration = datetime.now() - start
        message_prefix = f" {name} -" if name else ""
        print(f"--{message_prefix} execution time: {duration}")


def create_multi_platform_lock(py_ver: str, platform: str | None = None) -> None:
    print(f"# Creating multi-platform lock file for Python {py_ver} ...")
    platform_option = f"-p {platform}" if platform else ""
    with print_execution_time(f"conda-lock for {py_ver}"):
        subprocess.run(
            (
                "conda-lock lock --no-mamba --micromamba -f pyproject.toml"
                f" -f {_environments_folder}/env-python-{py_ver}.yml {platform_option}"
                f" --lockfile conda-py-{py_ver}-lock.yml"
            ),
            env=dict(os.environ, PYTHONUTF8="1"),
            shell=True,
            check=True,
            stderr=subprocess.STDOUT,
        )
        patch_absolute_path(Path(f"conda-py-{py_ver}-lock.yml"))


def per_platform_env(
    py_ver: str, extras: list[str] | None = None, dev=False, suffix=""
) -> None:
    if extras is None:
        extras = []
    print(
        f"# Creating per platform Conda env files for Python {py_ver} ({'WITH' if dev else 'NO'} dev dependencies) ... "
    )
    dev_dep_option = "--dev-dependencies" if dev else "--no-dev-dependencies"
    dev_suffix = "-dev" if dev else ""
    extras_option = " ".join(f"--extras {i}" for i in extras) if extras else ""
    subprocess.run(
        (
            f"conda-lock render {dev_dep_option} {extras_option} -k env"
            f" --filename-template {_environments_folder}/conda-py-{py_ver}-{{platform}}{dev_suffix}{suffix}.lock conda-py-{py_ver}-lock.yml"
        ),
        env=dict(os.environ, PYTHONUTF8="1"),
        shell=True,
        check=True,
        stderr=subprocess.STDOUT,
    )


def finalize_per_platform_envs(py_ver: str, dev=False, suffix="") -> None:
    dev_suffix = "-dev" if dev else ""
    platform_glob = "*-64"
    for lock_env_file in _environments_folder.glob(
        f"conda-py-{py_ver}-{platform_glob}{dev_suffix}{suffix}.lock.yml"
    ):
        LockFilePatcher(lock_env_file).patch()


def patch_absolute_path(file: Path) -> None:
    """
    Patch the given file to remove reference with absolute file path.
    """

    abs_path_base = str(_environments_folder.absolute().parent) + os.sep

    with tempfile.TemporaryDirectory(dir=str(file.parent)) as tmpdirname:
        patched_file = Path(tmpdirname) / file.name
        with open(patched_file, mode="w", encoding="utf-8") as patched:
            with open(file, encoding="utf-8") as f:
                for line in f:
                    patched.write(
                        line.replace(abs_path_base, "").replace(
                            _environments_folder.name + os.sep,
                            _environments_folder.name + "/",
                        )
                    )
        os.replace(patched_file, file)


class LockFilePatcher:
    """
    Patch the given file to remove hash information on pip dependency if any hash is missing.

    As soon as one hash is specified, pip requires all hashes to be specified.
    """

    def __init__(self, lock_file: Path) -> None:
        self.lock_file = lock_file
        self.pip_section_re = re.compile(r"^\s*- pip:\s*$")
        self.sha_re = re.compile(r"(.*)(\s--hash|#sha256)=\S*")

    def add_variables_section(self):
        """
        Add the variables section to the lock file.
        """

        with open(self.lock_file, mode="a", encoding="utf-8") as f:
            f.write(env_file_variables_section_)

    def patch_none_hash(self) -> None:
        """
        Patch the lock file to remove --hash=md5:None and #sha25=None

        - pip does not want hash with md5 (but accepts sha256 or others).
        - #sha256=None will conflict with the actual sha256
        """

        none_hash_re = re.compile(
            r"(.*)(?:\s--hash=(?:md5:|sha256:)|#sha256=)(?:None|)\s*$"
        )
        with tempfile.TemporaryDirectory(dir=str(self.lock_file.parent)) as tmpdirname:
            patched_file = Path(tmpdirname) / self.lock_file.name
            with open(patched_file, mode="w", encoding="utf-8") as patched, open(
                self.lock_file, encoding="utf-8"
            ) as f:
                for line in f:
                    match = none_hash_re.match(line)
                    if not match:
                        patched.write(line)
                    else:
                        patched.write(f"{match[1]}\n")
            patched_file.replace(self.lock_file)

    def is_missing_pip_hash(self) -> bool:
        """
        Check if the lock file contains pip dependencies with missing hash.
        """

        pip_dependency_re = re.compile(r"^\s*- (\S+) (@|===) .*")
        with open(self.lock_file, encoding="utf-8") as file:
            while not self.pip_section_re.match(file.readline()):
                pass

            for line in file:
                if pip_dependency_re.match(line) and not self.sha_re.match(line):
                    return True
        return False

    def remove_pip_hashes(self) -> None:
        """
        Remove all hashes from the pip dependencies.
        """

        with tempfile.TemporaryDirectory(dir=str(self.lock_file.parent)) as tmpdirname:
            patched_file = Path(tmpdirname) / self.lock_file.name
            with open(patched_file, mode="w", encoding="utf-8") as patched, open(
                self.lock_file, encoding="utf-8"
            ) as f:
                for line in f:
                    patched_line = self.sha_re.sub(r"\1", line)
                    patched.write(patched_line)
            patched_file.replace(self.lock_file)

    def patch(self, force_no_pip_hash=False) -> None:
        """
        Apply all patches to the lock file.
        """

        self.patch_none_hash()
        if force_no_pip_hash or self.is_missing_pip_hash():
            self.remove_pip_hashes()
        self.add_variables_section()


def config_conda() -> None:
    subprocess.run(
        "conda config --set channel_priority strict",
        shell=True,
        check=True,
        stderr=subprocess.STDOUT,
    )


def delete_multiplatform_lock_files() -> None:
    for f in Path().glob("*-lock.yml"):
        f.unlink()


def recreate_multiplatform_lock_files() -> None:
    delete_multiplatform_lock_files()

    # also delete per-platform lock files to make it obvious that
    # they must be cre-created after the multi-platform files were updated
    delete_per_platform_lock_files()

    with print_execution_time("create_multi_platform_lock"):
        for py_ver in _python_versions:
            create_multi_platform_lock(py_ver)


def delete_per_platform_lock_files() -> None:
    if _environments_folder.exists():
        for f in _environments_folder.glob("*.lock.yml"):
            f.unlink()


def recreate_per_platform_lock_files() -> None:
    delete_per_platform_lock_files()
    with print_execution_time("create_per_platform_lock"):
        for py_ver in _python_versions:
            per_platform_env(py_ver, ["core", "apps"], dev=False)
            finalize_per_platform_envs(py_ver, dev=False)
            per_platform_env(py_ver, ["core", "apps"], dev=True)
            finalize_per_platform_envs(py_ver, dev=True)


if __name__ == "__main__":
    assert _environments_folder.is_dir()

    config_conda()
    patch_pyproject_toml()

    recreate_multiplatform_lock_files()
    recreate_per_platform_lock_files()
