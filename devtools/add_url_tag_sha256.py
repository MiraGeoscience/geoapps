#!/usr/bin/env python3

#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

"""
Patch pyproject.toml with the computed sha256 for pip dependencies with a tag URL.

Usage: at the root of the project:
> python devtools/add_url_tag_sha256.py
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import warnings
from pathlib import Path
from urllib import request

_url_filename_re = re.compile(".*/([^/]*)")


def compute_sha256(url: str, base_name: str | None = None) -> str:
    filename_match = _url_filename_re.match(url)
    assert filename_match

    filename = filename_match[1]
    if base_name:
        filename = f"{base_name}-{filename}"
    with tempfile.TemporaryDirectory() as tmpdirname:
        copy = Path(tmpdirname) / filename
        print(f"# Fetching {url} ...")
        request.urlretrieve(url, str(copy))
        return (
            subprocess.check_output(["pip", "hash", "--algorithm", "sha256", copy])
            .decode("utf-8")
            .splitlines()[1]
            .split(":")[1]
        )


def patch_pyproject_toml() -> None:
    pyproject = Path("pyproject.toml")
    assert pyproject.is_file()

    if has_git_branches(pyproject):
        warnings.warn(
            f"{pyproject} contains git branches: not computing the sha256 for any git dependency."
        )
        return

    tag_url_re = re.compile(
        r"""^(\s*[^#]\S+\s*=\s*{\s*url\s*=\s*)"(.*/archive/refs/tags/.*)#sha256=\w*"(.*}.*)"""
    )
    pyproject_sha = Path("pyproject-sha.toml")
    with open(pyproject_sha, mode="w", encoding="utf-8") as patched:
        with open(pyproject, encoding="utf-8") as original:
            for line in original:
                match = tag_url_re.match(line)
                if not match:
                    patched.write(line)
                else:
                    line_start = match[1]
                    package_name = line_start.strip()
                    url = match[2]
                    line_end = match[3]
                    sha = compute_sha256(url, package_name)
                    patched_line = f"""{line_start}"{url}#sha256={sha}"{line_end}\n"""
                    patched.write(patched_line)

    pyproject_sha.replace(pyproject)


def has_git_branches(pyproject: Path) -> bool:
    branch_url_re = re.compile(
        r"""^(\s*[^#]\S+\s*=\s*{\s*url\s*=\s*)"(.*/archive/refs/heads/.*)\S*"(.*}.*)"""
    )
    with open(pyproject, encoding="utf-8") as f:
        for line in f:
            match = branch_url_re.match(line)
            if match:
                return True
    return False


if __name__ == "__main__":
    patch_pyproject_toml()
