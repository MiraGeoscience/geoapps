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

try:
    import thriftpy2
except (ModuleNotFoundError, ImportError):
    pass
else:
    from pathlib import Path
    from types import ModuleType
    from typing import Dict

    _INTERFACES_PATH = Path("interfaces")
    _INTERFACES: Dict[str, ModuleType] = {}

    def __getattr__(name):
        try:
            return _INTERFACES[name]
        except KeyError:
            interface = thriftpy2.load(str(_INTERFACES_PATH.joinpath(f"{name}.thrift")))
            _INTERFACES[name] = interface
            return interface
