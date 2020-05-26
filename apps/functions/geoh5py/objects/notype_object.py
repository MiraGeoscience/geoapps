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

from .object_base import ObjectBase
from .object_type import ObjectType


class NoTypeObject(ObjectBase):
    """
    Generic Data object without a registered type
    """

    __TYPE_UID = uuid.UUID(
        fields=(0x849D2F3E, 0xA46E, 0x11E3, 0xB4, 0x01, 0x2776BDF4F982)
    )

    def __init__(self, object_type: ObjectType, **kwargs):
        super().__init__(object_type, **kwargs)

        if object_type.name == "None":
            self.entity_type.name = "Unknown"

        object_type.workspace._register_object(self)

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID
