from dataclasses import dataclass
from enum import IntEnum
from typing import *


from . import shared




class InvalidDataOperation(Exception):
    message: Optional[str] = ""

class BadPrimitiveType(Exception):
    message: Optional[str] = ""




class DataAssociation(IntEnum):
    UNKNOWN = 0
    OBJECT = 1
    CELL = 2
    FACE = 3
    VERTEX = 4

class PrimitiveType(IntEnum):
    UNKNOWN = 0
    INTEGER = 1
    FLOAT = 2
    REFERENCED = 3
    TEXT = 4
    FILENAME = 5
    DATETIME = 6
    BLOB = 7




@dataclass
class Data:
    entity_: Optional[shared.Entity] = None
    association: Optional[int] = None

@dataclass
class DataUnit:
    unit: Optional[str] = ""

@dataclass
class DataType:
    uid: Optional[shared.Uuid] = None
    name: Optional[str] = None
    description: Optional[str] = ""
    units: Optional[DataUnit] = None
    primitive_type: Optional[int] = None

@dataclass
class DataSlab:
    start: Optional[int] = 0
    stride: Optional[int] = 1
    count: Optional[int] = 0
    block: Optional[int] = 1

@dataclass
class ReferencedDataEntry:
    key: Optional[int] = None
    value: Optional[str] = None

@dataclass
class ReferencedValues:
    indices: Optional[List[int]] = None
    entries: Optional[List[ReferencedDataEntry]] = None

@dataclass
class DataQuery:
    name: Optional[str] = None
    object_or_group: Optional[shared.Uuid] = None
    data_type: Optional[shared.Uuid] = None
    primitive_type: Optional[int] = None
    association: Optional[int] = None

@dataclass
class DataTypeQuery:
    name: Optional[str] = None
    primitive_type: Optional[int] = None
    units: Optional[DataUnit] = None




class DataService:
    def get_all(
        self,
    ) -> List[Data]:
        ...
    def find(
        self,
        query: DataQuery,
    ) -> List[Data]:
        ...
    def get(
        self,
        uid: shared.Uuid,
    ) -> Data:
        ...
    def get_float_values(
        self,
        data: shared.Uuid,
        slab: DataSlab,
    ) -> List[float]:
        ...
    def get_integer_values(
        self,
        data: shared.Uuid,
        slab: DataSlab,
    ) -> List[int]:
        ...
    def get_text_values(
        self,
        data: shared.Uuid,
        slab: DataSlab,
    ) -> List[str]:
        ...
    def get_referenced_values(
        self,
        data: shared.Uuid,
        slab: DataSlab,
    ) -> ReferencedValues:
        ...
    def get_datetime_values(
        self,
        data: shared.Uuid,
        slab: DataSlab,
    ) -> List[str]:
        ...
    def get_filename_values(
        self,
        data: shared.Uuid,
        slab: DataSlab,
    ) -> List[str]:
        ...
    def get_file_content(
        self,
        data: shared.Uuid,
        file_name: str,
    ) -> str:
        ...
    def get_blob_values(
        self,
        data: shared.Uuid,
        slab: DataSlab,
    ) -> List[int]:
        ...
    def get_blob_element(
        self,
        data: shared.Uuid,
        index: int,
    ) -> str:
        ...
    def get_all_types(
        self,
    ) -> List[DataType]:
        ...
    def find_types(
        self,
        query: DataTypeQuery,
    ) -> List[DataType]:
        ...
    def get_type(
        self,
        uid: shared.Uuid,
    ) -> DataType:
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
