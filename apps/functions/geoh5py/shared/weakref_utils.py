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

import weakref
from typing import Dict, Optional, TypeVar
from weakref import ReferenceType

K = TypeVar("K")  # pylint: disable=invalid-name
T = TypeVar("T")  # pylint: disable=invalid-name


def remove_none_referents(some_dict: Dict[K, ReferenceType]):
    """
    Removes any key from the given ``some_dict`` where the value is a reference to a deleted value
    (that is where referent of the ``weakref`` value is None).

    :param some_dict: The dictionary to be cleaned up.
    """
    dead_keys = [key for key, value in some_dict.items() if value() is None]
    for key in dead_keys:
        del some_dict[key]


def get_clean_ref(some_dict: Dict[K, ReferenceType[T]], key: K) -> Optional[T]:
    """
    Gets the referent value for the given ``key`` in a ``some_dict`` of ``weakref`` values.
    In case ``key`` points to a reference to a deleted value, remove that key from ``some_dict``
    on the fly, and returns None.
    :param some_dict: The dictionary of ``weakref`` values.
    :param key: The key
    :return: the referent value for ``key`` if found in the the dictionary, else None.
    """
    ref = some_dict.get(key, None)
    if ref is None:
        return None
    if ref() is None:
        del some_dict[key]
        return None
    return ref()


def insert_once(some_dict: Dict[K, ReferenceType], key: K, value):
    """
    TODO
    :param some_dict:
    :param key:
    :param value:
    :return:
    """
    existing_ref = some_dict.get(key, None)
    if existing_ref is not None and existing_ref() is not None:
        raise RuntimeError(f"Key '{key}' already used.")

    some_dict[key] = weakref.ref(value)
