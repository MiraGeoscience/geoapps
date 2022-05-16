#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from copy import deepcopy

from geoh5py.ui_json import InputFile

from geoapps.base.params import BaseParams

from .constants import default_ui_json, defaults, validations


class ScatterPlotParams(BaseParams):
    """
    Parameter class for scatter plot creation application.
    """

    def __init__(self, input_file=None, **kwargs):
        self._default_ui_json = deepcopy(default_ui_json)
        self._defaults = deepcopy(defaults)
        self._validations = validations
        self._objects = None
        self._data = None
        self._downsampling = None
        self._x = None
        self._x_log = None
        self._x_min = None
        self._x_max = None
        self._y = None
        self._y_log = None
        self._y_min = None
        self._y_max = None
        self._z = None
        self._z_log = None
        self._z_min = None
        self._z_max = None
        self._color = None
        self._color_min = None
        self._color_max = None
        self._color_maps = None
        self._size = None
        self._size_min = None
        self._size_max = None
        self._marker_size = None

        if input_file is None:
            ui_json = deepcopy(self._default_ui_json)
            input_file = InputFile(
                ui_json=ui_json,
                validations=self.validations,
                validation_options={"disabled": True},
            )

        super().__init__(input_file=input_file, **kwargs)

    @property
    def objects(self) -> ObjectBase | None:
        """
            Input object
        """
        return self._objects

    @objects.setter
    def objects(self, val):
        self.setter_validator("objects", val, fun=self._uuid_promoter)

    @property
    def data(self) -> Data | None:
        """
        Data
        """
        return self._data

    @data.setter
    def data(self, val):
        self.setter_validator("data", val)

    @property
    def downsampling(self) -> int | None:
        """
        downsampling
        """
        return self._downsampling

    @downsampling.setter
    def downsampling(self, val):
        self.setter_validator("downsampling", val)

    @property
    def x(self) -> Data | None:
        """
        x
        """
        return self._x

    @x.setter
    def x(self, val):
        self.setter_validator("x", val)

    @property
    def x_active(self) -> bool | None:
        """
        x active
        """
        return self._x_active

    @x_active.setter
    def x_active(self, val):
        self.setter_validator("x_active", val)

    @property
    def x_log(self) -> bool | None:
        """
        x log
        """
        return self._x_log

    @x_log.setter
    def x_log(self, val):
        self.setter_validator("x_log", val)

    @property
    def x_min(self) -> int | None:
        """
        x min
        """
        return self._x_min

    @x_min.setter
    def x_min(self, val):
        self.setter_validator("x_min", val)

    @property
    def x_max(self) -> int | None:
        """
        x max
        """
        return self._x_max

    @x_max.setter
    def x_max(self, val):
        self.setter_validator("x_max", val)

    @property
    def y(self) -> Data | None:
        """
        y
        """
        return self._y

    @y.setter
    def y(self, val):
        self.setter_validator("y", val)

    @property
    def y_log(self) -> bool | None:
        """
        y log
        """
        return self._y_log

    @y_log.setter
    def y_log(self, val):
        self.setter_validator("y_log", val)

    @property
    def y_min(self) -> int | None:
        """
        x min
        """
        return self._y_min

    @y_min.setter
    def y_min(self, val):
        self.setter_validator("y_min", val)

    @property
    def y_max(self) -> int | None:
        """
        y max
        """
        return self._y_max

    @y_max.setter
    def y_max(self, val):
        self.setter_validator("y_max", val)

    @property
    def z(self) -> Data | None:
        """
        z
        """
        return self._z

    @z.setter
    def z(self, val):
        self.setter_validator("z", val)

    @property
    def z_log(self) -> bool | None:
        """
        z log
        """
        return self._z_log

    @z_log.setter
    def z_log(self, val):
        self.setter_validator("z_log", val)

    @property
    def z_min(self) -> int | None:
        """
        z min
        """
        return self._z_min

    @z_min.setter
    def z_min(self, val):
        self.setter_validator("z_min", val)

    @property
    def z_max(self) -> int | None:
        """
        z max
        """
        return self._z_max

    @z_max.setter
    def z_max(self, val):
        self.setter_validator("z_max", val)

    @property
    def color(self) -> Data | None:
        """
        color
        """
        return self._color

    @color.setter
    def color(self, val):
        self.setter_validator("color", val)

    @property
    def color_log(self) -> bool | None:
        """
        color log
        """
        return self._color_log

    @color_log.setter
    def color_log(self, val):
        self.setter_validator("color_log", val)

    @property
    def color_min(self) -> int | None:
        """
        color min
        """
        return self._color_min

    @color_min.setter
    def color_min(self, val):
        self.setter_validator("color_min", val)

    @property
    def color_max(self) -> int | None:
        """
        color max
        """
        return self._color_max

    @color_max.setter
    def color_max(self, val):
        self.setter_validator("color_max", val)

    @property
    def size(self) -> Data | None:
        """
        size
        """
        return self._size

    @size.setter
    def size(self, val):
        self.setter_validator("size", val)

    @property
    def size_log(self) -> bool | None:
        """
        size log
        """
        return self._size_log

    @size_log.setter
    def size_log(self, val):
        self.setter_validator("size_log", val)

    @property
    def size_min(self) -> int | None:
        """
        size min
        """
        return self._size_min

    @size_min.setter
    def size_min(self, val):
        self.setter_validator("size_min", val)

    @property
    def size_max(self) -> int | None:
        """
        size max
        """
        return self._size_max

    @size_max.setter
    def size_max(self, val):
        self.setter_validator("size_max", val)

