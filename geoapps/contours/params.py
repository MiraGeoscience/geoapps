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

from geoapps.contours.constants import default_ui_json, defaults, validations
from geoapps.driver_base.params import BaseParams


class ContoursParams(BaseParams):
    """
    Parameter class for contours application.
    """

    def __init__(self, input_file=None, **kwargs):
        self._default_ui_json = deepcopy(default_ui_json)
        self._defaults = deepcopy(defaults)
        self._validations = validations
        self._objects = None
        self._data = None
        self._interval_min = None
        self._interval_max = None
        self._interval_spacing = None
        self._fixed_contours = None
        self._window_azimuth = None
        self._window_center_x = None
        self._window_center_y = None
        self._window_width = None
        self._window_height = None
        self._resolution = None
        self._export_as = None
        self._z_value = None
        self._ga_group_name = None

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
        Input object.
        """
        return self._objects

    @objects.setter
    def objects(self, val):
        self.setter_validator("objects", val, fun=self._uuid_promoter)

    @property
    def data(self) -> Data | None:
        """
        Input data.
        """
        return self._data

    @data.setter
    def data(self, val):
        self.setter_validator("data", val)

    @property
    def interval_min(self) -> float | None:
        """
        Minimum value for contours.
        """
        return self._interval_min

    @interval_min.setter
    def interval_min(self, val):
        self.setter_validator("interval_min", val)

    @property
    def interval_max(self) -> float | None:
        """
        Maximum value for contours.
        """
        return self._interval_max

    @interval_max.setter
    def interval_max(self, val):
        self.setter_validator("interval_max", val)

    @property
    def interval_spacing(self) -> float | None:
        """
        Step size for contours.
        """
        return self._interval_spacing

    @interval_spacing.setter
    def interval_spacing(self, val):
        self.setter_validator("interval_spacing", val)

    @property
    def fixed_contours(self) -> str | None:
        """
        String defining list of fixed contours.
        """
        return self._fixed_contours

    @fixed_contours.setter
    def fixed_contours(self, val):
        self.setter_validator("fixed_contours", val)

    @property
    def window_azimuth(self) -> float | None:
        """
        Rotation angle of the selection box.
        """
        return self._window_azimuth

    @window_azimuth.setter
    def window_azimuth(self, val):
        self.setter_validator("window_azimuth", val)

    @property
    def window_center_x(self) -> float | None:
        """
        Easting position of the selection box.
        """
        return self._window_center_x

    @window_center_x.setter
    def window_center_x(self, val):
        self.setter_validator("window_center_x", val)

    @property
    def window_center_y(self) -> float | None:
        """
        Northing position of the selection box.
        """
        return self._window_center_y

    @window_center_y.setter
    def window_center_y(self, val):
        self.setter_validator("window_center_y", val)

    @property
    def window_width(self) -> float | None:
        """
        Width (m) of the selection box.
        """
        return self._window_width

    @window_width.setter
    def window_width(self, val):
        self.setter_validator("window_width", val)

    @property
    def window_height(self) -> float | None:
        """
        Height (m) of the selection box.
        """
        return self._window_height

    @window_height.setter
    def window_height(self, val):
        self.setter_validator("window_height", val)

    @property
    def resolution(self) -> float | None:
        """
        Minimum data separation (m).
        """
        return self._resolution

    @resolution.setter
    def resolution(self, val):
        self.setter_validator("resolution", val)

    @property
    def export_as(self) -> str | None:
        """
        Name to save surface as.
        """
        return self._export_as

    @export_as.setter
    def export_as(self, val):
        self.setter_validator("export_as", val)

    @property
    def z_value(self) -> bool | None:
        """ """
        return self._z_value

    @z_value.setter
    def z_value(self, val):
        self.setter_validator("z_value", val)

    @property
    def ga_group_name(self) -> str | None:
        """
        Name to save geoh5 file as.
        """
        return self._ga_group_name

    @ga_group_name.setter
    def ga_group_name(self, val):
        self.setter_validator("ga_group_name", val)
