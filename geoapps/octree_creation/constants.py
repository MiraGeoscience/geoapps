# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024-2025 Mira Geoscience Ltd.                                '
#                                                                              '
#  This file is part of geoapps.                                               '
#                                                                              '
#  geoapps is distributed under the terms and conditions of the MIT License    '
#  (see LICENSE file at the root of this source code package).                 '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from __future__ import annotations

from uuid import UUID

from geoapps import assets_path


app_initializer = {
    "geoh5": str(assets_path() / "FlinFlon.geoh5"),
    "objects": UUID("{656acd40-25de-4865-814c-cb700f6ee51a}"),
    "Refinement A object": UUID("{656acd40-25de-4865-814c-cb700f6ee51a}"),
    "Refinement A levels": "4, 4, 4",
    "Refinement A horizon": False,
    "Refinement A distance": None,
    "Refinement B object": UUID("{ab3c2083-6ea8-4d31-9230-7aad3ec09525}"),
    "Refinement B levels": "0, 0, 4",
    "Refinement B horizon": True,
    "Refinement B distance": 1200.0,
}
