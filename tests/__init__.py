# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024 Mira Geoscience Ltd.                                     '
#                                                                              '
#  This file is part of geoapps.                                               '
#                                                                              '
#  geoapps is distributed under the terms and conditions of the MIT License    '
#  (see LICENSE file at the root of this source code package).                 '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from pathlib import Path

PROJECT = Path(__file__).parent.parent / "geoapps-assets" / "FlinFlon.geoh5"
PROJECT_DCIP = Path(__file__).parent.parent / "geoapps-assets" / "FlinFlon_dcip.geoh5"
PROJECT_TEM = (
    Path(__file__).parent.parent / "geoapps-assets" / "FlinFlon_airborne_tem.geoh5"
)
