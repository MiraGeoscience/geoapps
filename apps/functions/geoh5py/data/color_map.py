import numpy as np


class ColorMap:
    """ Records colors assigned to value ranges (where Value is the start of the range).
    """

    _attribute_map = {"File name": "name"}

    def __init__(self, **kwargs):

        self._values = dict()
        self._name = "Unknown"

        for attr, item in kwargs.items():
            try:
                if attr in self._attribute_map.keys():
                    attr = self._attribute_map[attr]
                setattr(self, attr, item)
            except AttributeError:
                continue

    @property
    def values(self) -> np.ndarray:
        """
        Colormap defined by an array of RGBA and values such that
        [val_1, R_1, G_1, B_1, A_1]
        ...
        [val_i, R_i, G_i, B_i, A_i]
        where R:red, G:green, B:blue and A:alpha are integer values between [0, 255]
        and val_i are sorted data values defining the position of each RGBA color.
        """
        return self._values

    @values.setter
    def values(self, values: np.ndarray):
        names = ["Value", "Red", "Green", "Blue", "Alpha"]
        formats = ["<f8", "u1", "u1", "u1", "u1"]

        if isinstance(values.dtype, np.dtype):
            assert all(
                [name in names for name in values.dtype.names]
            ), f"Input 'values' must contain fields with types {names}"
            self._values = np.asarray(values, dtype=list(zip(names, formats)))

        else:
            assert (
                values.shape[1] == 5
            ), f"'values' must be a an array of shape (*, 5) for [value, r, g, b, a]"
            self._values = np.core.records.fromarrays(
                values.T, names=names, formats=formats
            )

    @property
    def name(self):
        """
        Name of the colormap
        """
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    def __len__(self):
        return len(self._values)
