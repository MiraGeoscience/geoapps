#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import subsurface as subs
from geoh5py.objects import Curve, Points, Surface
from geoh5py.shared import Entity


def to_subsurface(entity: Entity, data=None):
    """
    Convert a geoh5py object and data to a subsurface class
    """

    subs_obj = None
    TYPES = (Points, Curve, Surface)
    assert isinstance(
        entity, TYPES
    ), f"Conversion to subsurface only available for objects type {TYPES}"

    return subs_obj
