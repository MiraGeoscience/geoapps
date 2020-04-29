from dataclasses import dataclass
from enum import IntEnum
from typing import *


from . import shared




class InvalidGroupOperation(Exception):
    message: Optional[str] = ""




class GroupClass(IntEnum):
    UNKNOWN = 0
    CONTAINER = 1
    DRILLHOLE = 2




@dataclass
class Group:
    entity_: Optional[shared.Entity] = None
    allow_move: Optional[bool] = True

@dataclass
class GroupQuery:
    name: Optional[str] = None
    type_uid: Optional[shared.Uuid] = None
    in_group: Optional[shared.Uuid] = None
    recursive: Optional[bool] = False




class GroupsService:
    def get_root(
        self,
    ) -> Group:
        ...
    def get_type(
        self,
        group_class: int,
    ) -> shared.Uuid:
        ...
    def get_class(
        self,
        type_uid: shared.Uuid,
    ) -> int:
        ...
    def get_all(
        self,
    ) -> List[Group]:
        ...
    def find(
        self,
        query: GroupQuery,
    ) -> List[Group]:
        ...
    def set_allow_move(
        self,
        groups: List[shared.Uuid],
        allow: bool,
    ) -> None:
        ...
    def move_to_group(
        self,
        groups: List[shared.Uuid],
        destination_group: shared.Uuid,
    ) -> None:
        ...
    def create(
        self,
        type_uid: shared.Uuid,
    ) -> Group:
        ...
    def set_public(
        self,
        entities: List[shared.Uuid],
        is_public: bool,
    ) -> None:
        ...
    def set_visible(
        self,
        entities: List[shared.Uuid],
        visible: bool,
    ) -> None:
        ...
    def set_allow_delete(
        self,
        entities: List[shared.Uuid],
        allow: bool,
    ) -> None:
        ...
    def set_allow_rename(
        self,
        entities: List[shared.Uuid],
        allow: bool,
    ) -> None:
        ...
    def rename(
        self,
        entities: shared.Uuid,
        new_name: str,
    ) -> None:
        ...
