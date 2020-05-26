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

from .primitive_type_enum import PrimitiveTypeEnum


class GeometricDataConstants:
    __X_TYPE_UID = uuid.UUID(
        fields=(0xE9E6B408, 0x4109, 0x4E42, 0xB6, 0xA8, 0x685C37A802EE)
    )
    __Y_TYPE_UID = uuid.UUID(
        fields=(0xF55B07BD, 0xD8A0, 0x4DFF, 0xBA, 0xE5, 0xC975D490D71C)
    )
    __Z_TYPE_UID = uuid.UUID(
        fields=(0xDBAFB885, 0x1531, 0x410C, 0xB1, 0x8E, 0x6AC9A40B4466)
    )

    @classmethod
    def x_datatype_uid(cls) -> uuid.UUID:
        return cls.__X_TYPE_UID

    @classmethod
    def y_datatype_uid(cls) -> uuid.UUID:
        return cls.__Y_TYPE_UID

    @classmethod
    def z_datatype_uid(cls) -> uuid.UUID:
        return cls.__Z_TYPE_UID

    @classmethod
    def primitive_type(cls) -> PrimitiveTypeEnum:
        return PrimitiveTypeEnum.GEOMETRIC
