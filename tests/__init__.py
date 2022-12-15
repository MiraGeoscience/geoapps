#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import os.path

import requests

PROJECT = "FlinFlon.geoh5"

URL = (
    "https://github.com/MiraGeoscience/geoapps/raw/release/0.9.1/assets/FlinFlon.geoh5"
)
# Todo - change this back to release branch
if not os.path.isfile("FlinFlon.geoh5"):
    r = requests.get(URL, timeout=5)
    with open(PROJECT, "wb") as file:
        file.write(r.content)

PROJECT_DCIP = "FlinFlon_dcip.geoh5"
URL = "https://github.com/MiraGeoscience/geoapps/raw/release/0.9.1/assets/FlinFlon_dcip.geoh5"

if not os.path.isfile("FlinFlon_dcip.geoh5"):
    r = requests.get(URL, timeout=5)
    with open(PROJECT_DCIP, "wb") as file:
        file.write(r.content)
