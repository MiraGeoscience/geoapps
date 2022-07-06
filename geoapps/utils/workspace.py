#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from uuid import UUID

from geoh5py.data import FloatData, IntegerData
from geoh5py.shared import Entity
from geoh5py.workspace import Workspace

from geoapps.utils.list import sorted_alphanumeric_list


def sorted_children_dict(
    object: UUID | Entity, workspace: Workspace = None
) -> dict[str, UUID]:
    """
    Uses natural sorting algorithm to order the keys of a dictionary containing
    children name/uid key/value pairs.

    If valid uuid entered calls get_entity.  Will return None if no object found
    in workspace for provided object

    :param object: geoh5py object containing children IntegerData, FloatData
        entities

    :return : sorted name/uid dictionary of children entities of object.

    """

    if isinstance(object, UUID):
        object = workspace.get_entity(object)[0]
        if not object:
            return None

    children_dict = {}
    for c in object.children:
        if not isinstance(c, (IntegerData, FloatData)):
            continue
        else:
            children_dict[c.name] = c.uid

    children_order = sorted_alphanumeric_list(list(children_dict.keys()))

    return {k: children_dict[k] for k in children_order}
