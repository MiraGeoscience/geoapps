#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import requests

PROJECT = "FlinFlon.geoh5"
URL = "https://github.com/MiraGeoscience/geoapps/raw/main/assets/FlinFlon.geoh5"
r = requests.get(URL)
with open(PROJECT, "wb") as file:
    file.write(r.content)

PROJECT_DCIP = "FlinFlon_dcip.geoh5"
URL = "https://github.com/MiraGeoscience/geoapps/raw/develop/assets/FlinFlon_dcip.geoh5"
r = requests.get(URL)
with open(PROJECT_DCIP, "wb") as file:
    file.write(r.content)
