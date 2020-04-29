from .data import Data
from .primitive_type_enum import PrimitiveTypeEnum


class TextData(Data):
    @classmethod
    def primitive_type(cls) -> PrimitiveTypeEnum:
        return PrimitiveTypeEnum.TEXT

    @property
    def values(self):
        if (getattr(self, "_values", None) is None) and self.existing_h5_entity:
            self._values = self.workspace.fetch_values(self.uid)

        return self._values

    @values.setter
    def values(self, values):
        """
        values(values)

        Assign string value

        Parameters
        ----------
        values: str
            Text in string format

        """
        self.modified_attributes = "values"
        self._values = values

    def __call__(self):
        return self.values
