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

import uuid
from abc import abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from ..data import CommentsData, Data
from ..shared import Entity
from .group_type import GroupType

if TYPE_CHECKING:
    from .. import workspace


class Group(Entity):
    """ Base Group class """

    def __init__(self, group_type: GroupType, **kwargs):
        assert group_type is not None
        super().__init__(**kwargs)

        self._entity_type = group_type

    def add_comment(self, comment: str, author: str = None):
        """
        Add text comment to an object.

        :param comment: Text to be added as comment.
        :param author: Author's name or :obj:`~geoh5py.workspace.workspace.Worspace.contributors`
        """

        date = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        if author is None:
            author = ",".join(self.workspace.contributors)

        comment_dict = {"Author": author, "Date": date, "Text": comment}

        if self.comments is None:

            self.workspace.create_entity(
                Data,
                entity={
                    "name": "UserComments",
                    "values": [comment_dict],
                    "parent": self,
                    "association": "OBJECT",
                },
                entity_type={"primitive_type": "TEXT"},
            )

        else:
            self.comments.values = self.comments.values + [comment_dict]

    @property
    def comments(self):
        """
        Fetch a :obj:`~geoh5py.data.text_data.CommentsData` entity from children.
        """
        for child in self.children:
            if isinstance(child, CommentsData):
                return child

        return None

    @property
    def entity_type(self) -> GroupType:
        return self._entity_type

    @classmethod
    def find_or_create_type(
        cls, workspace: "workspace.Workspace", **kwargs
    ) -> GroupType:

        return GroupType.find_or_create(workspace, cls, **kwargs)

    @classmethod
    @abstractmethod
    def default_type_uid(cls) -> Optional[uuid.UUID]:
        ...
