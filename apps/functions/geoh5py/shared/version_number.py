class VersionNumber:
    def __init__(self, number: float):
        self._value = float(number)

    @property
    def value(self) -> float:
        return self._value

    def __float__(self):
        return self._value

    def __str__(self):
        return str(self._value)
