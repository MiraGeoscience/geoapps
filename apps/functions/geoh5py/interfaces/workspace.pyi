from dataclasses import dataclass
from enum import IntEnum
from typing import *


from . import shared




class FileIOException(Exception):
    message: Optional[str] = ""

class FileFormatException(Exception):
    message: Optional[str] = ""







@dataclass
class Workspace:
    file_path: Optional[str] = ""
    version: Optional[shared.VersionNumber] = None
    distance_unit: Optional[shared.DistanceUnit] = None
    date_created: Optional[shared.DateTime] = None
    date_modified: Optional[shared.DateTime] = None




class WorkspaceService:
    def get_api_version(
        self,
    ) -> shared.VersionString:
        ...
    def create_geoh5(
        self,
        file_path: str,
    ) -> Workspace:
        ...
    def open_geoh5(
        self,
        file_path: str,
    ) -> Workspace:
        ...
    def save(
        self,
        file_path: str,
        overwrite_file: bool,
    ) -> Workspace:
        ...
    def export_objects(
        self,
        objects_or_groups: List[shared.Uuid],
        file_path: str,
        overwrite_file: bool,
    ) -> Workspace:
        ...
    def export_all(
        self,
        file_path: str,
        overwrite_file: bool,
    ) -> Workspace:
        ...
    def close(
        self,
    ) -> None:
        ...
    def get_contributors(
        self,
    ) -> List[str]:
        ...
