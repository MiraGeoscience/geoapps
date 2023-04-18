#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import urllib
import json

def license_notice_from_pypi(package: str, version: str) -> str:
    pypi_api_url = f"https://pypi.org/pypi/{package}/{version}/json"
    with urllib.request.urlopen(pypi_api_url) as response:
        data = json.loads(response.read())
        return data["info"]["license"]



def get_package_list() -> dict:
    cmd = "poetry show --no-dev"

