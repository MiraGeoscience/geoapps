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
        self._x = None
        self._x_active = None
        self._x_log = True
        self._y = None
        self._y_active = None
        self._y_log = None
        self._z = None
        self._z_log = None
        self._z_active = None
        self._color = None
        self._color_active = None
        self._color_log = None
        self._color_maps = None
        self._size = None
        self._size_active = None
        self._size_log = None

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
    def y(self) -> Data | None:
        """
        y
        """
        return self._y

    @y.setter
    def y(self, val):
        self.setter_validator("y", val)

    @property
    def y_active(self) -> bool | None:
        """
        y active
        """
        return self._y_active

    @y_active.setter
    def y_active(self, val):
        self.setter_validator("y_active", val)

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
    def z(self) -> Data | None:
        """
        z
        """
        return self._z

    @z.setter
    def z(self, val):
        self.setter_validator("z", val)

    @property
    def z_active(self) -> bool | None:
        """
        z active
        """
        return self._z_active

    @z_active.setter
    def z_active(self, val):
        self.setter_validator("z_active", val)

    @property
    def z_log(self) -> bool | None:
        """
        z log
        """
        return self._z_log

    @z_log.setter
    def z_log(self, val):
        self.setter_validator("z_log", val)

