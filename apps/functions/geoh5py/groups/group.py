import uuid
from abc import abstractmethod
from typing import TYPE_CHECKING, Optional

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
