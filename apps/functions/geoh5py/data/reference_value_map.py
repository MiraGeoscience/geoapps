from typing import Dict, Optional


class ReferenceValue:
    """ Represents a value for ReferencedData as a string.
    """

    def __init__(self, value: str = None):
        self._value = value

    @property
    def value(self) -> Optional[str]:
        return self._value

    def __str__(self):
        # TODO: representation for None?
        return str(self._value)


class ReferenceValueMap:
    """ Maps from reference index to reference value of ReferencedData.
    """

    def __init__(self, color_map: Dict[int, ReferenceValue] = None):
        self._map = dict() if color_map is None else color_map

    def __getitem__(self, item: int) -> ReferenceValue:
        return self._map[item]

    def __len__(self):
        return len(self._map)
