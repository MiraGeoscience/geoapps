class VersionString:
    def __init__(self, value: str):
        self._value = value

    @property
    def value(self) -> str:
        return self._value

    def __str__(self):
        return self._value

    def _h5_rep(self) -> str:
        return self._value
