from enum import Enum


class DataAssociationEnum(Enum):
    """Known data association
    """

    UNKNOWN = 0
    OBJECT = 1
    CELL = 2
    VERTEX = 3
    FACE = 4
    GROUP = 5
