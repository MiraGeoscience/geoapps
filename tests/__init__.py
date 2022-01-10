#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import requests

project = "FlinFlon.geoh5"
url = "https://github.com/MiraGeoscience/geoapps/raw/main/assets/FlinFlon.geoh5"
r = requests.get(url)
open(project, "wb").write(r.content)

project_dcip = "FlinFlon_dcip.geoh5"
url = "https://github.com/MiraGeoscience/geoapps/raw/develop/assets/FlinFlon_dcip.geoh5"
r = requests.get(url)
open(project_dcip, "wb").write(r.content)
