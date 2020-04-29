from numpy import ndarray, ravel

from .data import Data, PrimitiveTypeEnum


class IntegerData(Data):
    @classmethod
    def primitive_type(cls) -> PrimitiveTypeEnum:
        return PrimitiveTypeEnum.INTEGER

    @property
    def values(self) -> ndarray:
        """
        :return: values: An array of float values
        """
        if (getattr(self, "_values", None) is None) and self.existing_h5_entity:
            self._values = self.workspace.fetch_values(self.uid)

        return self._values

    @values.setter
    def values(self, values):
        self.modified_attributes = "values"
        self._values = ravel(values).astype(int)

    def __call__(self):
        return self.values
