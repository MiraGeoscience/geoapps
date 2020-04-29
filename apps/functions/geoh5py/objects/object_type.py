from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from ..shared import EntityType

if TYPE_CHECKING:
    from .. import workspace


class ObjectType(EntityType):
    def __init__(self, workspace: "workspace.Workspace", **kwargs):
        assert workspace is not None
        super().__init__(workspace, **kwargs)

        workspace._register_type(self)

    @staticmethod
    def _is_abstract() -> bool:
        return False

    @classmethod
    def find_or_create(
        cls, workspace: "workspace.Workspace", entity_class, **kwargs
    ) -> ObjectType:
        """ Find or creates an EntityType with given UUID that matches the given
        Group implementation class.

        It is expected to have a single instance of EntityType in the Workspace
        for each concrete Entity class.

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
    def create_custom(workspace: "workspace.Workspace") -> ObjectType:
        """ Creates a new instance of ObjectType for an unlisted custom Object type with a
        new auto-generated UUID.
        """
        return ObjectType(workspace)
