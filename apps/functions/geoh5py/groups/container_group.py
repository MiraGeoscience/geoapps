import uuid

from .group import Group, GroupType


class ContainerGroup(Group):
    """ The type for the basic Container group."""

    __TYPE_UID = uuid.UUID(
        fields=(0x61FBB4E8, 0xA480, 0x11E3, 0x8D, 0x5A, 0x2776BDF4F982)
    )

    _name = "Container"
    _description = "Container"

    def __init__(self, group_type: GroupType, **kwargs):
        assert group_type is not None
        super().__init__(group_type, **kwargs)

        group_type.workspace._register_group(self)

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID
