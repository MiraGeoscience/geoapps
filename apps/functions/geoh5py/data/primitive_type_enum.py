from enum import Enum


class PrimitiveTypeEnum(Enum):
    INVALID = 0
    INTEGER = 1
    FLOAT = 2
    TEXT = 3
    REFERENCED = 4
    FILENAME = 5
    BLOB = 6
    VECTOR = 7
    DATETIME = 8
    GEOMETRIC = 9
