from typing import Optional


class DataUnit:
    """
    Data unit
    """

    def __init__(self, unit_name: str = None):
        self._rep = unit_name

    @property
    def name(self) -> Optional[str]:
        return self._rep

    def __str__(self):
        return str(self._rep)
