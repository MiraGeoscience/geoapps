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

# pylint: skip-file

from typing import TYPE_CHECKING, List

from .. import interfaces

if TYPE_CHECKING:
    from ..interfaces.workspace import Workspace as i_Workspace
    from ..interfaces.shared import Uuid as i_Uuid
    from ..interfaces.shared import VersionString as i_VersionString


class WorkspaceHandler:
    @staticmethod
    def get_api_version() -> i_VersionString:
        version = interfaces.shared.VersionString()
        version.value = interfaces.api.API_VERSION
        return version

    def create_geoh5(self, file_path: str) -> i_Workspace:
        # TODO
        pass

    # pylint: disable=unused-argument
    @staticmethod
    def open_geoh5(file_path: str) -> i_Workspace:
        # TODO
        return interfaces.workspace.Workspace()

    def save(self, file_path: str, overwrite_file: bool) -> i_Workspace:
        # TODO
        pass

    def save_copy(self, file_path: str, overwrite_file: bool) -> i_Workspace:
        # TODO
        pass

    def export_objects(
        self, objects_or_groups: List[i_Uuid], file_path: str, overwrite_file: bool
    ) -> i_Workspace:
        # TODO
        pass

    def close(self,) -> None:
        # TODO
        pass

    def get_contributors(self,) -> List[str]:
        # TODO
        pass
