import uuid

from .group import Group, GroupType


class GiftoolsGroup(Group):
    """ The type for a GIFtools group."""

    __TYPE_UID = uuid.UUID(
        fields=(0x585B3218, 0xC24B, 0x41FE, 0xAD, 0x1F, 0x24D5E6E8348A)
    )

    _name = "GIFtools Project"
    _description = "GIFtools Project"

    def __init__(self, group_type: GroupType, **kwargs):
        assert group_type is not None
        super().__init__(group_type, **kwargs)

        group_type.workspace._register_group(self)

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID
