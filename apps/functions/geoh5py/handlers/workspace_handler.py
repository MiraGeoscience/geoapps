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
