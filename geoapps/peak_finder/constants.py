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
    "geoh5": str(assets_path() / "FlinFlon_tem.geoh5"),
    "objects": UUID("{4667bf5a-b639-4fd0-8e04-c0e555f59f0e}"),
    "group_a_data": UUID("{ca9b158a-9b39-409a-88d1-29b965c357a3}"),
}
