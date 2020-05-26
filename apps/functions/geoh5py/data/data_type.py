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

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Dict, Optional, Type, cast

from ..shared import EntityType
from .color_map import ColorMap
from .geometric_data_constants import GeometricDataConstants
from .primitive_type_enum import PrimitiveTypeEnum

if TYPE_CHECKING:
    from .. import workspace
    from . import data  # noqa: F401


class DataType(EntityType):
    """
    DataType class
    """

    _attribute_map = EntityType._attribute_map.copy()
    _attribute_map.update(
        {
            "Hidden": "hidden",
            "Mapping": "mapping",
            "Number of bins": "number_of_bins",
            "Primitive type": "primitive_type",
            "Transparent no data": "transparent_no_data",
        }
    )

    _primitive_type: Optional[PrimitiveTypeEnum] = None
    _color_map: Optional[ColorMap] = None
    _units: Optional[str] = None
    _number_of_bins: int = 50
    _transparent_no_data = True
    _mapping: str = "equal_area"
    _hidden: bool = False

    def __init__(self, workspace: "workspace.Workspace", **kwargs):
        assert workspace is not None
        super().__init__(workspace, **kwargs)

        workspace._register_type(self)

    @staticmethod
    def _is_abstract() -> bool:
        return False

    @property
    def color_map(self) -> Optional[ColorMap]:
        r"""
        :obj:`~geoh5py.data.color_map.ColorMap`: Colormap used for plotting

        The colormap can be set from a :obj:`dict` of sorted values with
        corresponding RGBA color.

        .. code-block:: python

            color_map = {
                val_1: [r_1, g_1, b_1, a_1],
                ...,
                val_i: [r_i, g_i, b_i, a_i]
            }

        """
        return self._color_map

    @color_map.setter
    def color_map(self, color_map: Dict):
        assert "values" in list(color_map.keys()), "'color_map' must contain 'values'"
        self._color_map = ColorMap(**color_map)
        self.modified_attributes = "Color map"

    @property
    def units(self) -> Optional[str]:
        """
        :obj:`str`: Data units
        """
        return self._units

    @units.setter
    def units(self, unit: str):
        self._units = unit
        self.modified_attributes = "attributes"

    @property
    def number_of_bins(self) -> Optional[int]:
        """
        :obj:`int`: Number of bins used by the histogram [50]
        """
        return self._number_of_bins

    @number_of_bins.setter
    def number_of_bins(self, n_bins: int):
        self._number_of_bins = n_bins
        self.modified_attributes = "attributes"

    @property
    def transparent_no_data(self) -> bool:
        """
        :obj:`bool`: Use transparent for no-data-value [True]
        """
        return self._transparent_no_data

    @transparent_no_data.setter
    def transparent_no_data(self, value: bool):
        self._transparent_no_data = value
        self.modified_attributes = "attributes"

    @property
    def hidden(self) -> bool:
        """
        :obj:`bool`: Hidden data [False]
        """
        return self._hidden

    @hidden.setter
    def hidden(self, value: bool):
        self._hidden = value
        self.modified_attributes = "attributes"

    @property
    def mapping(self) -> str:
        """
        :obj:`str`: Color stretching type chosen from:
        'linear', ['equal_area'], 'logarithmic', 'cdf', 'missing'
        """
        return self._mapping

    @mapping.setter
    def mapping(self, value: str):
        mappings = ["linear", "equal_area", "logarithmic", "cdf", "missing"]
        assert (
            value in mappings
        ), f"Mapping {value} was provided but should be one of {mappings}"
        self._mapping = value
        self.modified_attributes = "attributes"

    @property
    def primitive_type(self) -> Optional[PrimitiveTypeEnum]:
        """
        :obj:`~geoh5py.data.primitive_type_enum.PrimitiveTypeEnum`
        """
        return self._primitive_type

    @primitive_type.setter
    def primitive_type(self, value):
        if isinstance(value, str):
            self._primitive_type = getattr(PrimitiveTypeEnum, value.upper())
        else:
            assert isinstance(
                value, PrimitiveTypeEnum
            ), f"Primitive type value must be of type {PrimitiveTypeEnum}"
            self._primitive_type = value

    @classmethod
    def create(
        cls, workspace: "workspace.Workspace", data_class: Type["data.Data"]
    ) -> DataType:
        """ Creates a new instance of :obj:`~geoh5py.data.data_type.DataType` with
        corresponding :obj:`~geoh5py.data.primitive_type_enum.PrimitiveTypeEnum`

        :param data_class: A :obj:`~geoh5py.data.data.Data` implementation class.

        :return: A new instance of :obj:`~geoh5py.data.data_type.DataType`.
        """
        uid = uuid.uuid4()
        primitive_type = data_class.primitive_type()
        return cls(workspace, uid=uid, primitive_type=primitive_type)

    @classmethod
    def find_or_create(cls, workspace: "workspace.Workspace", **kwargs) -> DataType:
        """ Find or creates an EntityType with given UUID that matches the given
        Group implementation class.

        :param workspace: An active Workspace class

        :return: A new instance of DataType.
        """
        uid = uuid.uuid4()

        for key, val in kwargs.items():
            if key.lower() in ["id", "uid"]:
                if isinstance(val, uuid.UUID):
                    uid = val
                else:
                    uid = uuid.UUID(val)

        entity_type = cls.find(workspace, uid)
        if entity_type is not None:
            return entity_type

        return cls(workspace, **kwargs)

    @classmethod
    def _for_geometric_data(
        cls, workspace: "workspace.Workspace", uid: uuid.UUID
    ) -> DataType:
        geom_primitive_type = GeometricDataConstants.primitive_type()
        data_type = cast(DataType, workspace.find_type(uid, DataType))
        if data_type is not None:
            assert data_type.primitive_type == geom_primitive_type
            return data_type
        return cls(workspace, uid=uid, primitive_type=geom_primitive_type)

    @classmethod
    def for_x_data(cls, workspace: "workspace.Workspace") -> DataType:
        return cls._for_geometric_data(
            workspace, GeometricDataConstants.x_datatype_uid()
        )

    @classmethod
    def for_y_data(cls, workspace: "workspace.Workspace") -> DataType:
        return cls._for_geometric_data(
            workspace, GeometricDataConstants.y_datatype_uid()
        )

    @classmethod
    def for_z_data(cls, workspace: "workspace.Workspace") -> DataType:
        return cls._for_geometric_data(
            workspace, GeometricDataConstants.z_datatype_uid()
        )
