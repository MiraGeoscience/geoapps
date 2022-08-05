#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from copy import deepcopy

from geoh5py.data import Data
from geoh5py.objects import ObjectBase
from geoh5py.ui_json import InputFile

from geoapps.driver_base.params import BaseParams

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
        self._downsampling = None
        self._x = None
        self._x_log = None
        self._x_min = None
        self._x_max = None
        self._x_thresh = None
        self._y = None
        self._y_log = None
        self._y_min = None
        self._y_max = None
        self._y_thresh = None
        self._z = None
        self._z_log = None
        self._z_min = None
        self._z_max = None
        self._z_thresh = None
        self._color = None
        self._color_log = None
        self._color_min = None
        self._color_max = None
        self._color_maps = None
        self._color_thresh = None
        self._size = None
        self._size_log = None
        self._size_min = None
        self._size_max = None
        self._size_thresh = None
        self._size_markers = None
        self._monitoring_directory = None

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
    def downsampling(self) -> int | None:
        """
        Percent to downsample the data
        """
        return self._downsampling

    @downsampling.setter
    def downsampling(self, val):
        self.setter_validator("downsampling", val)

    @property
    def x(self) -> Data | None:
        """
        X-axis data
        """
        return self._x

    @x.setter
    def x(self, val):
        self.setter_validator("x", val)

    @property
    def x_log(self) -> bool | None:
        """
        Plot x-axis logarithmically
        """
        return self._x_log

    @x_log.setter
    def x_log(self, val):
        self.setter_validator("x_log", val)

    @property
    def x_min(self) -> float | None:
        """
        Minimum value for x data
        """
        return self._x_min

    @x_min.setter
    def x_min(self, val):
        self.setter_validator("x_min", val)

    @property
    def x_max(self) -> float | None:
        """
        Max value for x data
        """
        return self._x_max

    @x_max.setter
    def x_max(self, val):
        self.setter_validator("x_max", val)

    @property
    def x_thresh(self) -> float | None:
        """
        Threshold for x log
        """
        return self._x_thresh

    @x_thresh.setter
    def x_thresh(self, val):
        self.setter_validator("x_thresh", val)

    @property
    def y(self) -> Data | None:
        """
        Y-axis data
        """
        return self._y

    @y.setter
    def y(self, val):
        self.setter_validator("y", val)

    @property
    def y_log(self) -> bool | None:
        """
        Plot y-axis logarithmically
        """
        return self._y_log

    @y_log.setter
    def y_log(self, val):
        self.setter_validator("y_log", val)

    @property
    def y_min(self) -> float | None:
        """
        Minimum value for y data
        """
        return self._y_min

    @y_min.setter
    def y_min(self, val):
        self.setter_validator("y_min", val)

    @property
    def y_max(self) -> float | None:
        """
        Max value for y data
        """
        return self._y_max

    @y_max.setter
    def y_max(self, val):
        self.setter_validator("y_max", val)

    @property
    def y_thresh(self) -> float | None:
        """
        Threshold for y log
        """
        return self._y_thresh

    @y_thresh.setter
    def y_thresh(self, val):
        self.setter_validator("y_thresh", val)

    @property
    def z(self) -> Data | None:
        """
        Z-axis data
        """
        return self._z

    @z.setter
    def z(self, val):
        self.setter_validator("z", val)

    @property
    def z_log(self) -> bool | None:
        """
        Plot z-axis logarithmically
        """
        return self._z_log

    @z_log.setter
    def z_log(self, val):
        self.setter_validator("z_log", val)

    @property
    def z_min(self) -> float | None:
        """
        Minimum value for z data
        """
        return self._z_min

    @z_min.setter
    def z_min(self, val):
        self.setter_validator("z_min", val)

    @property
    def z_max(self) -> float | None:
        """
        Max value for z data
        """
        return self._z_max

    @z_max.setter
    def z_max(self, val):
        self.setter_validator("z_max", val)

    @property
    def z_thresh(self) -> float | None:
        """
        Threshold for z log
        """
        return self._z_thresh

    @z_thresh.setter
    def z_thresh(self, val):
        self.setter_validator("z_thresh", val)

    @property
    def color(self) -> Data | None:
        """
        Color data
        """
        return self._color

    @color.setter
    def color(self, val):
        self.setter_validator("color", val)

    @property
    def color_log(self) -> bool | None:
        """
        Plot color data logarithmically
        """
        return self._color_log

    @color_log.setter
    def color_log(self, val):
        self.setter_validator("color_log", val)

    @property
    def color_min(self) -> float | None:
        """
        Minimum value for color data
        """
        return self._color_min

    @color_min.setter
    def color_min(self, val):
        self.setter_validator("color_min", val)

    @property
    def color_max(self) -> float | None:
        """
        Max value for color data
        """
        return self._color_max

    @color_max.setter
    def color_max(self, val):
        self.setter_validator("color_max", val)

    @property
    def color_thresh(self) -> float | None:
        """
        Threshold for color log
        """
        return self._color_thresh

    @color_thresh.setter
    def color_thresh(self, val):
        self.setter_validator("color_thresh", val)

    @property
    def color_maps(self) -> list | None:
        """
        Color map choices
        """
        return self._color_maps

    @color_maps.setter
    def color_maps(self, val):
        self.setter_validator("color_maps", val)

    @property
    def size(self) -> Data | None:
        """
        Size data
        """
        return self._size

    @size.setter
    def size(self, val):
        self.setter_validator("size", val)

    @property
    def size_log(self) -> bool | None:
        """
        Plot size data logarithmically
        """
        return self._size_log

    @size_log.setter
    def size_log(self, val):
        self.setter_validator("size_log", val)

    @property
    def size_min(self) -> float | None:
        """
        Minimum value for size data
        """
        return self._size_min

    @size_min.setter
    def size_min(self, val):
        self.setter_validator("size_min", val)

    @property
    def size_max(self) -> float | None:
        """
        Max value for size data
        """
        return self._size_max

    @size_max.setter
    def size_max(self, val):
        self.setter_validator("size_max", val)

    @property
    def size_thresh(self) -> float | None:
        """
        Threshold for size log
        """
        return self._size_thresh

    @size_thresh.setter
    def size_thresh(self, val):
        self.setter_validator("size_thresh", val)

    @property
    def size_markers(self) -> int | None:
        """
        Size of markers
        """
        return self._size_markers

    @size_markers.setter
    def size_markers(self, val):
        self.setter_validator("size_markers", val)

    @property
    def monitoring_directory(self) -> str | None:
        """
        Output path.
        """
        return self._monitoring_directory

    @monitoring_directory.setter
    def monitoring_directory(self, val):
        self.setter_validator("monitoring_directory", val)
