#!/usr/bin/env python3

#  Copyright (c) 2022 Mira Geoscience Ltd.
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

import hashlib
import re
import tempfile
from pathlib import Path
from urllib import request

_url_filename_re = re.compile(".*/([^/]*)")


def computeSha256(url: str, base_name: str = None) -> str:
    filename_match = _url_filename_re.match(url)
    assert filename_match

    filename = filename_match[1]
    if base_name:
        filename = f"{base_name}-{filename}"
    with tempfile.TemporaryDirectory() as tmpdirname:
        copy = Path(tmpdirname) / filename
        print(f"# Fetching {url} ...")
        request.urlretrieve(url, str(copy))
        with open(copy, "rb") as f:
            f_byte = f.read()
            sha256 = hashlib.sha256(f_byte)
            return sha256.hexdigest()


def patchPyprojectToml():
    pyproject = Path("pyproject.toml")
    assert pyproject.is_file()

    tag_url_re = re.compile(
        r"""^(\s*\w*\s*=\s*{\s*url\s*=\s*)"(.*/archive/refs/tags/.*)#sha256=\w*"(.*}.*)"""
    )
    pyproject_sha = Path("pyproject-sha.toml")
    with open(pyproject_sha, "w") as patched:
        with open(pyproject) as input:
            for line in input:
                match = tag_url_re.match(line)
                if not match:
                    patched.write(line)
                else:
                    line_start = match[1]
                    package_name = line_start.strip()
                    url = match[2]
                    line_end = match[3]
                    sha = computeSha256(url, package_name)
                    patched_line = f"""{line_start}"{url}#sha256={sha}"{line_end}\n"""
                    patched.write(patched_line)

    pyproject_sha.replace(pyproject)


if __name__ == "__main__":
    patchPyprojectToml()
