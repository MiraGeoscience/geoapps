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

# pylint: skip-file

from typing import TYPE_CHECKING, List, Optional

from .. import interfaces
from ..workspace import Workspace

if TYPE_CHECKING:
    from ..interfaces.groups import Group as i_Group
    from ..interfaces.groups import GroupQuery as i_GroupQuery
    from ..interfaces.shared import Uuid as i_Uuid


class GroupsHandler:
    @staticmethod
    def get_root() -> Optional[i_Group]:
        root = Workspace.active().root

        if root is not None:
            root_entity = interfaces.shared.Entity(
                uid=interfaces.shared.Uuid(str(root.uid)),
                type_uid=interfaces.shared.Uuid(str(root.entity_type.uid)),
                name=root.name,
                visible=root.visible,
                allow_delete=root.allow_delete,
                allow_rename=root.allow_rename,
                is_public=root.public,
            )

            return interfaces.groups.Group(
                entity_=root_entity, allow_move=root.allow_move
            )
        return None

    def get_type(self, group_class: int) -> i_Uuid:
        # TODO
        pass

    def get_class(self, type_uid: i_Uuid) -> int:
        # TODO
        pass

    @staticmethod
    def get_all() -> List[i_Group]:
        Workspace.active().all_groups()
        # TODO
        return []

    def find(self, query: i_GroupQuery) -> List[i_Group]:
        # TODO
        pass

    def set_allow_move(self, groups: List[i_Uuid], allow: bool) -> None:
        # TODO
        pass

    def move_to_group(self, groups: List[i_Uuid], destination_group: i_Uuid) -> None:
        # TODO
        pass

    def create(self, type_uid: i_Uuid) -> i_Group:
        # TODO
        pass

    def set_public(self, entities: List[i_Uuid], is_public: bool) -> None:
        # TODO
        pass

    def set_visible(self, entities: List[i_Uuid], visible: bool) -> None:
        # TODO
        pass

    def set_allow_delete(self, entities: List[i_Uuid], allow: bool) -> None:
        # TODO
        pass

    def set_allow_rename(self, entities: List[i_Uuid], allow: bool) -> None:
        # TODO
        pass

    def rename(self, entities: i_Uuid, new_name: str) -> None:
        # TODO
        pass
