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

import json
from typing import List, Optional

from .data import Data
from .primitive_type_enum import PrimitiveTypeEnum


class TextData(Data):
    @classmethod
    def primitive_type(cls) -> PrimitiveTypeEnum:
        return PrimitiveTypeEnum.TEXT

    @property
    def values(self) -> Optional[str]:
        """
        :obj:`str` Text value.
        """
        if (getattr(self, "_values", None) is None) and self.existing_h5_entity:
            self._values = self.workspace.fetch_values(self.uid)

        return self._values

    @values.setter
    def values(self, values):
        self.modified_attributes = "values"
        self._values = values

    def __call__(self):
        return self.values


class CommentsData(Data):
    """
    Comments added to an Object or Group.
    Stored as a list of dictionaries with the following keys:

        .. code-block:: python

            comments = [
                {
                    "Author": "username",
                    "Date": "2020-05-21T10:12:15",
                    "Text": "A text comment."
                },
            ]

        where "Date" can be generated from datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    """

    @classmethod
    def primitive_type(cls) -> PrimitiveTypeEnum:
        return PrimitiveTypeEnum.TEXT

    @property
    def values(self) -> Optional[List[dict]]:
        """
        :obj:`list` List of comments
        """
        if (getattr(self, "_values", None) is None) and self.existing_h5_entity:
            comment_str = self.workspace.fetch_values(self.uid)

            if comment_str is not None:
                self._values = json.loads(comment_str[0])["Comments"]

        return self._values

    @values.setter
    def values(self, values):
        self.modified_attributes = "values"

        if values is not None:
            for value in values:
                assert isinstance(value, dict), (
                    f"Error setting CommentsData with expected input of type list[dict].\n"
                    f"Input {type(values)} provided."
                )
                assert list(value.keys()) == ["Author", "Date", "Text"], (
                    f"Comment dictionaries must include keys 'Author', 'Date' and 'Text'.\n"
                    f"Keys {list(value.keys())} provided."
                )

        self._values = values

    def __call__(self):
        return self.values
