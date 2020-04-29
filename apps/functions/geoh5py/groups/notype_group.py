import uuid

from .group import Group, GroupType


class NoTypeGroup(Group):
    """ A group with no type."""

    __TYPE_UID = uuid.UUID(
        fields=(0xDD99B610, 0xBE92, 0x48C0, 0x87, 0x3C, 0x5B5946EA2840)
    )

    _name = "NoType"
    _description = "<Unknown>"

    def __init__(self, group_type: GroupType, **kwargs):
        assert group_type is not None
        super().__init__(group_type, **kwargs)

        group_type.workspace._register_group(self)

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID
