#!/usr/bin/env python3

# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2022-2024 Mira Geoscience Ltd.                                '
#                                                                              '
#  This file is part of geoapps.                                               '
#                                                                              '
#  geoapps is distributed under the terms and conditions of the MIT License    '
#  (see LICENSE file at the root of this source code package).                 '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

"""
Creates cross-platform lock files for each python version and per-platform conda environment files.

Cross-platform lock files are created at the root of the project.
Per-platform conda environment files with and without dev dependencies, are placed under the `environments` sub-folder.

Usage: from the conda base environment, at the root of the project:
> python devtools/run_conda_lock.py

To prepare the conda base environment, see devtools/setup-conda-base.bat
"""

from __future__ import annotations

import logging
import os
import re
import shutil
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

_ENV_FILE_VARIABLES_SECTION = """
variables:
  KMP_WARNINGS: 0
"""

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ENVIRONMENT_FILES_DIR = _PROJECT_ROOT / "environments"

_PYTHON_VERSIONS = ["3.10"]

_MAMBA_ENV_RUNNER = (
    r"%ProgramFiles%\Mira Geoscience\Geoscience ANALYST\CmdRunner\MambaEnvRunner.exe"
)

_logger = logging.getLogger(f"{__package__}.{Path(__file__).stem}")


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


def create_multi_platform_lock(
    py_ver: str, platforms: list[str] | None = None, include_dev: bool = False
) -> Path:
    print(f"# Creating multi-platform lock file for Python {py_ver} ...")
    platform_option = " ".join(f"-p {p}" for p in platforms) if platforms else ""
    output_lock_file = _PROJECT_ROOT / f"py-{py_ver}.conda-lock.yml"
    with print_execution_time(f"conda-lock for {py_ver}"):
        conda_lock_cmd = "conda-lock lock --no-mamba --micromamba -f pyproject.toml"
        conda_lock_cmd += " " + (
            "--dev-dependencies" if include_dev else "--no-dev-dependencies"
        )
        conda_lock_cmd += (
            f" -f {_ENVIRONMENT_FILES_DIR}/env-python-{py_ver}.yml {platform_option}"
            f" --lockfile {output_lock_file}"
        )
        subprocess.run(
            conda_lock_cmd,
            env=dict(os.environ, PYTHONUTF8="1", CONDA_CHANNEL_PRIORITY="strict"),
            shell=True,
            check=True,
            stderr=subprocess.STDOUT,
        )
        patch_absolute_path(output_lock_file)
    return output_lock_file


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

    patched_locked_env_file = (
        _PROJECT_ROOT / f"py-{py_ver}-patched{name_suffix}.conda-lock.yml"
    )
    patch_extra_as_non_optional(
        _PROJECT_ROOT / f"py-{py_ver}.conda-lock.yml", patched_locked_env_file, extras
    )

    subprocess.run(
        (
            f"conda-lock render {dev_dep_option} {extras_option} -k env"
            f" --filename-template {_ENVIRONMENT_FILES_DIR}/py-{py_ver}-{{platform}}{name_suffix}.conda.lock"
            f" {patched_locked_env_file}"
        ),
        env=dict(os.environ, PYTHONUTF8="1", CONDA_CHANNEL_PRIORITY="strict"),
        shell=True,
        check=True,
        stderr=subprocess.STDOUT,
    )
    patched_locked_env_file.unlink(missing_ok=True)


def patch_extra_as_non_optional(
    input_lock_file: Path, output_lock_file: Path, extra_deps: list[str]
) -> None:
    """Create a new multiplatform lock file from the input multiplatform lock file
    by forcing the extra packages and their dependencies as non-optional.

    This is to work-around a bug of conda-lock: when an indirect dependency
    is present in multiple extra groups, conda-lock assign it only to one group (aka category).
    Then, in per platform lock file create from the multi-platform lock, some dependencies would be lost.

    :param input_lock_file: the path to the original multiplatform lock file.
    :param output_lock_file: the path to the new multiplatform lock file to create.
    :param extra_deps: the list of extras dependencies to force as non-optional. It will also force recursively
        the dependencies of these extras as non-optional.
    """

    extra_deps = extra_dependencies(extra_deps)
    shutil.copy(input_lock_file, output_lock_file)
    force_non_optional_packages(output_lock_file, extra_deps)


def finalize_per_platform_envs(py_ver: str, dev=False, suffix="") -> None:
    platform_glob = "*-64"
    file_glob = per_platform_lock_file_name(py_ver, platform_glob, dev, suffix)
    for lock_env_file in _ENVIRONMENT_FILES_DIR.glob(file_glob):
        LockFilePatcher(lock_env_file).patch()


def per_platform_lock_file_name(
    py_ver: str, platform: str, dev=False, suffix=""
) -> str:
    name_suffix = full_name_suffix(dev, suffix)
    return f"py-{py_ver}-{platform}{name_suffix}.conda.lock.yml"


def full_name_suffix(dev: bool, suffix: str) -> str:
    dev_suffix = "-dev" if dev else ""
    if suffix and not suffix.startswith("-"):
        suffix = f"-{suffix}"
    return f"{dev_suffix}{suffix}"


def patch_absolute_path(file: Path) -> None:
    """
    Patch the given file to remove reference with absolute file path.
    """

    abs_path_base = str(_ENVIRONMENT_FILES_DIR.absolute().parent) + os.sep

    with tempfile.TemporaryDirectory(dir=str(file.parent)) as tmpdirname:
        patched_file = Path(tmpdirname) / file.name
        with open(patched_file, mode="w", encoding="utf-8") as patched:
            with open(file, encoding="utf-8") as f:
                for line in f:
                    patched.write(
                        line.replace(abs_path_base, "").replace(
                            _ENVIRONMENT_FILES_DIR.name + os.sep,
                            _ENVIRONMENT_FILES_DIR.name + "/",
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
            f.write(_ENV_FILE_VARIABLES_SECTION)

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
    return list(_ENVIRONMENT_FILES_DIR.glob("*.conda-lock.yml"))


def delete_multiplatform_lock_files() -> None:
    for f in get_multiplatform_lock_files():
        f.unlink()


def recreate_multiplatform_lock_files(include_dev: bool = True) -> list[Path]:
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
        for py_ver in _PYTHON_VERSIONS:
            file = create_multi_platform_lock(py_ver, include_dev=include_dev)
            created_files.append(file)
            remove_redundant_pip_from_lock_file(file)
            force_non_optional_packages(file, non_optional_deps)
    return created_files


def delete_per_platform_lock_files() -> None:
    if _ENVIRONMENT_FILES_DIR.exists():
        for f in _ENVIRONMENT_FILES_DIR.glob("*.lock.yml"):
            f.unlink()


def recreate_per_platform_lock_files(
    suffix_for_extras: dict[str, list[str]] | None = None, include_dev: bool = False
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
    :param include_dev: whether to include the dev dependencies.
    """

    delete_per_platform_lock_files()
    if not suffix_for_extras:
        suffix_for_extras = {"": []}
    with print_execution_time("create_per_platform_lock"):
        for py_ver in _PYTHON_VERSIONS:
            for suffix, extras in suffix_for_extras.items():
                if suffix and not suffix.startswith("-"):
                    suffix = f"-{suffix}"
                per_platform_env(py_ver, extras, dev=False, suffix=suffix)
                finalize_per_platform_envs(py_ver, dev=False, suffix=suffix)
                if include_dev:
                    per_platform_env(py_ver, extras, dev=True, suffix=suffix)
                    finalize_per_platform_envs(py_ver, dev=True, suffix=suffix)


def extra_dependencies(extra_names: list[str]) -> list[str]:
    """
    List the names of dependencies from pyproject.toml under
    any extra listed in `extra_names`.
    """

    pyproject_toml = _PROJECT_ROOT / "pyproject.toml"
    assert pyproject_toml.is_file()

    extra_packages: list[str] = []
    with open(pyproject_toml, "rb") as pyproject:
        content = toml.load(pyproject)
    extras_section = content["tool"]["poetry"].get("extras", {})
    for name, spec in extras_section.items():
        if name in extra_names and isinstance(spec, list):
            extra_packages.extend(spec)
    return extra_packages


def non_optional_dependencies() -> list[str]:
    """
    List the names of non-optional dependencies from pyproject.toml
    """

    pyproject_toml = _PROJECT_ROOT / "pyproject.toml"
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

    assert lock_file.is_file(), f"File not found: {lock_file}"
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

    assert lock_file.is_file(), f"File not found: {lock_file}"
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
        if version_from_conda == version_from_pip:
            _logger.info(
                f"package {pip_package['name']} ({version_from_conda} {pip_package['platform']})"
                " is fetched from pip and conda."
            )
            redundant_pip_packages.append(package_name)
        else:
            msg = (
                f"package {pip_package['name']} ({pip_package['platform']}) is fetched with a different version "
                f"from pip ({pip_package['version']}) and conda ({version_from_conda})."
            )
            if (
                version_from_pip <= version_from_conda
                and version_from_pip.major == version_from_conda.major
            ):
                _logger.warning(msg + ": versions are expected compatible.")
                redundant_pip_packages.append(package_name)
            else:
                has_non_compatible_versions = True
                _logger.critical(msg + ": versions are **not compatible**.")
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


def create_env_lock_files(suffix_for_extras: dict[str, list[str]], include_dev: bool):
    """
    :param suffix_for_extras: specify suffix for env lock file with their list of extras. See in pyproject.toml
        for the definition of the extras.
        For more details, see this parameter in the documentation of recreate_per_platform_lock_files()
    :param include_dev: whether to include the development dependencies in the locked env file.
    """
    assert _ENVIRONMENT_FILES_DIR.is_dir()

    patch_pyproject_toml()
    recreate_multiplatform_lock_files(include_dev)
    recreate_per_platform_lock_files(suffix_for_extras, include_dev)


def main():
    logging.basicConfig(level=logging.INFO)

    suffix_for_extras = {
        "": ["core", "apps"],
    }
    create_env_lock_files(suffix_for_extras, True)


if __name__ == "__main__":
    main()
