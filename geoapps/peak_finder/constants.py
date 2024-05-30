# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024 Mira Geoscience Ltd.                                     '
#                                                                              '
#  This file is part of geoapps.                                               '
#                                                                              '
#  geoapps is distributed under the terms and conditions of the MIT License    '
#  (see LICENSE file at the root of this source code package).                 '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from uuid import UUID

from geoapps import assets_path

app_initializer = {
    "geoh5": str(assets_path() / "FlinFlon.geoh5"),
    "objects": UUID("{bb208abb-dc1f-4820-9ea9-b8883e5ff2c6}"),
    "data": UUID("{b834a590-dea9-48cb-abe3-8c714bb0bb7c}"),
    "line_field": UUID("{90b1d710-8a0f-4f69-bd38-6c06c7a977ed}"),
}
