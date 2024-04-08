#!/usr/bin/env python3

#  Copyright (c) 2022-2024 Mira Geoscience Ltd.
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
import sys
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import networkx as nx  # type: ignore
import tomli as toml
from add_url_tag_sha256 import patch_pyproject_toml
from packaging.version import Version
from ruamel.yaml import YAML  # type: ignore

env_file_variables_section_ = """
variables:
  KMP_WARNINGS: 0
"""

_python_versions = ["3.10"]
_build_dev = True

_environments_folder = Path("environments").resolve()


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


def create_multi_platform_lock(py_ver: str, platforms: list[str] | None = None) -> Path:
    print(f"# Creating multi-platform lock file for Python {py_ver} ...")
    platform_option = " ".join(f"-p {p}" for p in platforms) if platforms else ""
    with print_execution_time(f"conda-lock for {py_ver}"):
        subprocess.run(
            (
                "conda-lock lock --no-mamba --micromamba -f pyproject.toml"
                f" -f {_environments_folder}/env-python-{py_ver}.yml {platform_option}"
                f" --lockfile py-{py_ver}.conda-lock.yml"
            ),
            env=dict(os.environ, PYTHONUTF8="1", CONDA_CHANNEL_PRIORITY="strict"),
            shell=True,
            check=True,
            stderr=subprocess.STDOUT,
        )
        file = Path(f"py-{py_ver}.conda-lock.yml")
        patch_absolute_path(file)
    return file


def per_platform_env(
    py_ver: str, extras: list[str] | None = None, dev=False, suffix=""
) -> None:
    if extras is None:
        extras = []
    print(
        f"# Creating per platform Conda env files for Python {py_ver} ({'WITH' if dev else 'NO'} dev dependencies) ... "
    )
    dev_dep_option = "--dev-dependencies" if dev else "--no-dev-dependencies"
    name_suffix = full_name_suffix(dev, suffix)
    extras_option = " ".join(f"--extras {i}" for i in extras) if extras else ""
    subprocess.run(
        (
            f"conda-lock render {dev_dep_option} {extras_option} -k env"
            f" --filename-template {_environments_folder}/py-{py_ver}-{{platform}}{name_suffix}.conda.lock py-{py_ver}.conda-lock.yml"
        ),
        env=dict(os.environ, PYTHONUTF8="1", CONDA_CHANNEL_PRIORITY="strict"),
        shell=True,
        check=True,
        stderr=subprocess.STDOUT,
    )


def finalize_per_platform_envs(py_ver: str, dev=False, suffix="") -> None:
    platform_glob = "*-64"
    name_suffix = full_name_suffix(dev, suffix)
    for lock_env_file in _environments_folder.glob(
        f"py-{py_ver}-{platform_glob}{name_suffix}.conda.lock.yml"
    ):
        LockFilePatcher(lock_env_file).patch()


def full_name_suffix(dev: bool, suffix: str) -> str:
    dev_suffix = "-dev" if dev else ""
    return f"{dev_suffix}{suffix}"


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
            # advance until the pip section
            for line in file:
                if self.pip_section_re.match(line):
                    break

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


def get_multiplatform_lock_files() -> list[Path]:
    return list(Path().glob("*.conda-lock.yml"))


def delete_multiplatform_lock_files() -> None:
    for f in get_multiplatform_lock_files():
        f.unlink()


def recreate_multiplatform_lock_files() -> list[Path]:
    """
    Delete and recreate the multi-platform lock files for each python version.
    """

    delete_multiplatform_lock_files()

    # also delete per-platform lock files to make it obvious that
    # they must be cre-created after the multi-platform files were updated
    delete_per_platform_lock_files()

    non_optional_deps = non_optional_dependencies()
    created_files: list[Path] = []
    with print_execution_time("create_multi_platform_lock"):
        for py_ver in _python_versions:
            file = create_multi_platform_lock(py_ver)
            created_files.append(file)
            remove_redundant_pip_from_lock_file(file)
            force_non_optional_packages(file, non_optional_deps)
    return created_files


def delete_per_platform_lock_files() -> None:
    if _environments_folder.exists():
        for f in _environments_folder.glob("*.lock.yml"):
            f.unlink()


def recreate_per_platform_lock_files(
    suffix_for_extras: dict[str, list[str]] = {}
) -> None:
    """
    Delete and recreate the per-platform lock files for each python version.

    :param suffix_for_extras: a dictionary with the suffix for each list extra.
        Creates a per-platform lock file for each extra list with the corresponding suffix.
        For example, to create "core", "apps", and "ui" per-platform lock files with
        their corresponding list of extras::

            {
                "core": [],  # no extra
                "apps": ["apps"],
                "ui": ["apps", "ui"],  # UI needs both apps and ui extras
            }
    """

    delete_per_platform_lock_files()
    if not suffix_for_extras:
        suffix_for_extras = {"": []}
    with print_execution_time("create_per_platform_lock"):
        for py_ver in _python_versions:
            for suffix, extras in suffix_for_extras.items():
                if suffix and not suffix.startswith("-"):
                    suffix = f"-{suffix}"
                per_platform_env(py_ver, extras, dev=False, suffix=suffix)
                finalize_per_platform_envs(py_ver, dev=False, suffix=suffix)
                if _build_dev:
                    per_platform_env(py_ver, extras, dev=True, suffix=suffix)
                    finalize_per_platform_envs(py_ver, dev=True, suffix=suffix)


def non_optional_dependencies() -> list[str]:
    """
    List the names of non-optional dependencies from pyproject.toml
    """

    pyproject_toml = Path("pyproject.toml")
    assert pyproject_toml.is_file()

    non_optional_packages: list[str] = []
    with open(pyproject_toml, "rb") as pyproject:
        content = toml.load(pyproject)
    for name, spec in content["tool"]["poetry"]["dependencies"].items():
        if isinstance(spec, str) or (
            isinstance(spec, dict) and not spec.get("optional", False)
        ):
            non_optional_packages.append(name)
    return non_optional_packages


def force_non_optional_packages(lock_file: Path, force_packages: list[str]) -> None:
    """
    Patch the multi-platform lock file to force some packages not to be optional.
    """

    if len(force_packages) == 0:
        return

    assert lock_file.is_file()
    print("## Force packages as non-optional: " + ", ".join(force_packages))

    yaml = YAML()
    yaml.width = 1200

    with open(lock_file, encoding="utf-8") as file:
        yaml_content = yaml.load(file)
        assert yaml_content is not None

    # collect packages from that list that are already in the lock file as optional
    packages_to_change = [
        package
        for package in yaml_content["package"]
        if package["name"] in force_packages and package["optional"]
    ]

    # change all packages in the dependency tree to be non-optional
    graph = build_dependency_tree(packages_to_change)
    for package in graph.nodes:
        for p in yaml_content["package"]:
            if p["name"] == package:
                p["optional"] = False
                p["category"] = "main"

    with open(lock_file, mode="w", encoding="utf-8") as file:
        yaml.dump(yaml_content, file)


def remove_redundant_pip_from_lock_file(lock_file: Path) -> None:
    """
    Remove pip packages that are also fetched from conda.

    If a package version is different between pip and conda, exit with an error.
    """

    assert lock_file.is_file()
    print("## Removing redundant pip packages ...")

    yaml = YAML()
    yaml.width = 1200

    with open(lock_file, encoding="utf-8") as file:
        yaml_content = yaml.load(file)
        assert yaml_content is not None

    packages = yaml_content["package"]
    remaining_pip_per_platform: dict[str, list[str]] = {}
    for platform in ["linux-64", "osx-64", "win-64"]:
        # parse the yml file
        pip_packages = [
            p for p in packages if p["manager"] == "pip" and p["platform"] == platform
        ]
        if len(pip_packages) == 0:
            continue

        conda_packages = {
            p["name"]: p["version"]
            for p in packages
            if p["manager"] == "conda" and p["platform"] == platform
        }
        redundant_pip_names = list_redundant_pip_packages(pip_packages, conda_packages)

        # these Qt libraries are irrelevant for Conda
        redundant_pip_names.append("pyqt5-qt5")
        redundant_pip_names.append("pyqtwebengine-qt5")

        graph = build_dependency_tree(pip_packages)
        graph = trim_dependency_tree(graph, redundant_pip_names)
        remaining_pip_names = list(graph.nodes)
        remaining_pip_packages = [
            p for p in pip_packages if p["name"] in remaining_pip_names
        ]
        assert (
            len(list_redundant_pip_packages(remaining_pip_packages, conda_packages))
            == 0
        ), "Could not eliminate all redundant pip packages (likely due mismatch on versions)"
        remaining_pip_per_platform[platform] = remaining_pip_names

    yaml_content["package"] = [
        p
        for p in packages
        if p["manager"] != "pip"
        or p["name"] in remaining_pip_per_platform[p["platform"]]
    ]
    with open(lock_file, mode="w", encoding="utf-8") as file:
        yaml.dump(yaml_content, file)


def list_redundant_pip_packages(
    pip_packages: list[dict], conda_packages: dict[str, str]
) -> list[str]:
    """
    Return the names of pip packages that are also listed as conda packages.

    If some packages are present for both pip and conda but with different versions, exit with an error.
    :param pip_packages: list of pip packages, where each package is a dict with its full description.
    :param conda_packages: list of conda packages as a dict with the package name as key and its version as value.
    :return: names of the pip packages that are also listed as conda packages.
    """

    redundant_pip_packages: list[str] = []
    for pip_package in pip_packages:
        package_name = pip_package["name"]
        version_str_from_conda = conda_packages.get(package_name, None)
        if version_str_from_conda is None:
            continue

        version_from_conda = Version(version_str_from_conda)
        version_from_pip = Version(pip_package["version"])
        has_non_compatible_versions = False
        if version_from_conda == pip_package["version"]:
            print(
                f"package {pip_package['name']} ({version_from_conda} {pip_package['platform']}) is fetched from pip and conda"
            )
            redundant_pip_packages.append(package_name)
        else:
            msg = (
                f"package {pip_package['name']} ({pip_package['platform']}) is fetched with a different version "
                f"from pip ({pip_package['version']}) and conda ({version_from_conda})"
            )
            if (
                version_from_pip <= version_from_conda
                and version_from_pip.major == version_from_conda.major
            ):
                print(msg + ": versions are expected compatible")
                redundant_pip_packages.append(package_name)
            else:
                has_non_compatible_versions = True
                print(msg + ": versions are **not compatible**")
        if has_non_compatible_versions:
            sys.exit(1)

    return redundant_pip_packages


def build_dependency_tree(packages: list[dict]) -> nx.DiGraph:
    graph = nx.DiGraph()
    for package in packages:
        graph.add_node(package["name"])
        for dependency in package["dependencies"].keys():
            graph.add_edge(package["name"], dependency)
    return graph


def trim_dependency_tree(
    graph: nx.DiGraph, packages_to_remove: list[str]
) -> nx.DiGraph:
    # Remove the specified packages
    for package in packages_to_remove:
        if package in graph:
            graph.remove_node(package)

    orphaned_nodes = [node for node, degree in graph.degree() if degree == 0]
    for node in orphaned_nodes:
        graph.remove_node(node)

    return graph


def main():
    assert _environments_folder.is_dir()

    patch_pyproject_toml()
    recreate_multiplatform_lock_files()

    extras_and_suffix = {
        "": ["core", "apps"],
    }
    recreate_per_platform_lock_files(extras_and_suffix)


if __name__ == "__main__":
    main()
