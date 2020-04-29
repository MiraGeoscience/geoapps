import uuid
from typing import Optional

from .group import Group, GroupType


class CustomGroup(Group):
    """ A custom group, for an unlisted Group type.
    """

    _name = "Custom"
    _description = "Custom"

    def __init__(self, group_type: GroupType, **kwargs):
        assert group_type is not None
        super().__init__(group_type, **kwargs)

        group_type.workspace._register_group(self)

    @classmethod
    def default_type_uid(cls) -> Optional[uuid.UUID]:
        raise RuntimeError(f"No predefined static type UUID for {cls}.")
        # return None
