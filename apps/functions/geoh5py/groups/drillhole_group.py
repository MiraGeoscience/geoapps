import uuid

from .group import Group, GroupType


class DrillholeGroup(Group):
    """ The type for the group containing drillholes."""

    __TYPE_UID = uuid.UUID(
        fields=(0x825424FB, 0xC2C6, 0x4FEA, 0x9F, 0x2B, 0x6CD00023D393)
    )

    _name = "Drillholes"

    def __init__(self, group_type: GroupType, **kwargs):
        assert group_type is not None
        super().__init__(group_type, **kwargs)

        group_type.workspace._register_group(self)

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID
