#  Copyright (c) 2020 Mira Geoscience Ltd.
#
#  This file is part of geoh5py.
#
#  geoh5py is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  geoh5py is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with geoh5py.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from ..shared import EntityType

if TYPE_CHECKING:
    from .. import workspace
    from . import group  # noqa: F401


class GroupType(EntityType):
    _attribute_map = EntityType._attribute_map.copy()
    _attribute_map.update(
        {
            "Allow move contents": "allow_move_content",
            "Allow delete contents": "allow_delete_content",
        }
    )

    _allow_move_content = True
    _allow_delete_content = True

    def __init__(self, workspace: "workspace.Workspace", **kwargs):
        assert workspace is not None
        super().__init__(workspace, **kwargs)

        workspace._register_type(self)

    @staticmethod
    def _is_abstract() -> bool:
        return False

    @property
    def allow_move_content(self) -> bool:
        """
        :obj:`bool`: [True] Allow to move the group
        :obj:`geoh5py.shared.entity.Entity.children`.
        """
        return self._allow_move_content

    @allow_move_content.setter
    def allow_move_content(self, allow: bool):
        self._allow_move_content = bool(allow)

    @property
    def allow_delete_content(self) -> bool:
        """
        :obj:`bool`: [True] Allow to delete the group
        :obj:`geoh5py.shared.entity.Entity.children`.
        """
        return self._allow_delete_content

    @allow_delete_content.setter
    def allow_delete_content(self, allow: bool):
        self._allow_delete_content = bool(allow)

    @classmethod
    def find_or_create(
        cls, workspace: "workspace.Workspace", entity_class, **kwargs
    ) -> GroupType:
        """ Find or creates an EntityType with given UUID that matches the given
        Group implementation class.

        :param workspace: An active Workspace class
        :param entity_class: An Group implementation class.

        :return: A new instance of GroupType.
        """
        uid = uuid.uuid4()
        if getattr(entity_class, "default_type_uid", None) is not None:
            uid = entity_class.default_type_uid()

            if "ID" in list(kwargs.keys()):
                kwargs["ID"] = uid
            else:
                kwargs["uid"] = uid
        else:
            for key, val in kwargs.items():
                if key.lower() in ["id", "uid"]:
                    uid = uuid.UUID(val)

        entity_type = cls.find(workspace, uid)
        if entity_type is not None:
            return entity_type

        return cls(workspace, **kwargs)

    @staticmethod
    def create_custom(workspace: "workspace.Workspace", **kwargs) -> GroupType:
        """ Creates a new instance of GroupType for an unlisted custom Group type with a
        new auto-generated UUID.
        """
        return GroupType(workspace, **kwargs)
