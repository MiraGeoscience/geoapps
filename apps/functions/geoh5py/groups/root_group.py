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

from . import NoTypeGroup
from .group import GroupType


class RootGroup(NoTypeGroup):
    """The Root group of a workspace."""

    __ROOT_NAME = "Workspace"

    def __init__(self, group_type: GroupType, **kwargs):
        assert group_type is not None
        super().__init__(group_type, **kwargs)

        # Hard wired attributes
        self._parent = None
        self._allow_move = False
        self._allow_delete = False
        self._allow_rename = False
        self._name = self.__ROOT_NAME

    @property
    def parent(self):
        """
        Parental entity of root is always None
        """
        return self._parent

    @parent.setter
    def parent(self, _):
        self._parent = None
