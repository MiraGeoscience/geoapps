# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024 Mira Geoscience Ltd.                                     '
#                                                                              '
#  This file is part of geoapps.                                               '
#                                                                              '
#  geoapps is distributed under the terms and conditions of the MIT License    '
#  (see LICENSE file at the root of this source code package).                 '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from __future__ import annotations

from uuid import UUID

from geoh5py.data import FloatData, IntegerData
from geoh5py.shared import Entity
from geoh5py.workspace import Workspace

from geoapps.utils.list import sorted_alphanumeric_list


def sorted_children_dict(
    entity: UUID | Entity, workspace: Workspace | None = None
) -> dict[str, UUID]:
    """
    Uses natural sorting algorithm to order the keys of a dictionary containing
    children name/uid key/value pairs.

    If valid uuid entered calls get_entity.  Will return None if no entity found
    in workspace for provided entity

    :param entity: geoh5py entity containing children IntegerData, FloatData
        entities

    :return : sorted name/uid dictionary of children entities of entity.

    """

    if isinstance(entity, UUID):
        entity = workspace.get_entity(entity)[0]
        if not entity:
            return None

    children_dict = {}
    for child in entity.children:
        if not isinstance(child, (IntegerData, FloatData)):
            continue

        children_dict[child.name] = child.uid

    children_order = sorted_alphanumeric_list(list(children_dict))

    return {k: children_dict[k] for k in children_order}
