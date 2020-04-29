from dataclasses import dataclass
from enum import IntEnum
from typing import *






class InvalidUid(Exception):
    message: Optional[str] = ""

class BadEntityType(Exception):
    message: Optional[str] = ""

class BadEntityName(Exception):
    message: Optional[str] = ""







@dataclass
class VersionString:
    value: Optional[str] = ""

@dataclass
class VersionNumber:
    value: Optional[float] = 0.0

@dataclass
class Uuid:
    id: Optional[str] = ""

@dataclass
class DateTime:
    value: Optional[str] = ""

@dataclass
class DistanceUnit:
    unit: Optional[str] = ""

@dataclass
class Coord3D:
    x: Optional[float] = 0.0
    y: Optional[float] = 0.0
    z: Optional[float] = 0.0

@dataclass
class Entity:
    uid: Optional[Uuid] = None
    type_uid: Optional[Uuid] = None
    name: Optional[str] = None
    visible: Optional[bool] = False
    allow_delete: Optional[bool] = True
    allow_rename: Optional[bool] = True
    is_public: Optional[bool] = True




class EntityService:
    def set_public(
        self,
        entities: List[Uuid],
        is_public: bool,
    ) -> None:
        ...
    def set_visible(
        self,
        entities: List[Uuid],
        visible: bool,
    ) -> None:
        ...
    def set_allow_delete(
        self,
        entities: List[Uuid],
        allow: bool,
    ) -> None:
        ...
    def set_allow_rename(
        self,
        entities: List[Uuid],
        allow: bool,
    ) -> None:
        ...
    def rename(
        self,
        entities: Uuid,
        new_name: str,
    ) -> None:
        ...
