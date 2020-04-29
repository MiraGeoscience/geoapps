from __future__ import annotations

from . import NoTypeGroup
from .group import GroupType


class RootGroup(NoTypeGroup):
    """ The type for the workspace root group."""

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
