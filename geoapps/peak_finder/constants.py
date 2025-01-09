# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024-2025 Mira Geoscience Ltd.                                '
#                                                                              '
#  This file is part of geoapps.                                               '
#                                                                              '
#  geoapps is distributed under the terms and conditions of the MIT License    '
#  (see LICENSE file at the root of this source code package).                 '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from uuid import UUID

from geoapps import assets_path

app_initializer = {
    "geoh5": str(assets_path() / "FlinFlon_airborne_tem.geoh5"),
    "objects": UUID("{34698019-cde6-4b43-8d53-a040b25c989a}"),
    "group_a_data": UUID("{22a9cf91-5cff-42b5-8bbb-2f1c6a559204}"),
}
