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
from geoapps.iso_surfaces.constants import default_ui_json, defaults, validations


class IsoSurfacesParams(BaseParams):
    """
    Parameter class for iso surfaces creation application.
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
        self._contours = None
        self._max_distance = None
        self._resolution = None
        self._export_as = None

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
        Input data
        """
        return self._data

    @data.setter
    def data(self, val):
        self.setter_validator("data", val)

    @property
    def interval_min(self):
        return self._interval_min

    @interval_min.setter
    def interval_min(self, val):
        self._interval_min = val

    @property
    def interval_max(self):
        return self._interval_max

    @interval_max.setter
    def interval_max(self, val):
        self._interval_max = val

    @property
    def interval_spacing(self):
        return self._interval_spacing

    @property
    def fixed_contours(self):
        return self._fixed_contours

    @fixed_contours.setter
    def fixed_contours(self, val):
        self._fixed_contours = val

    @interval_spacing.setter
    def interval_spacing(self, val):
        self._interval_spacing = val

    @property
    def contours(self) -> str | None:
        """
        String defining sets of contours.
        Contours can be defined over an interval `50:200:10` and/or at a fix value `215`.
        Any combination of the above can be used:
        50:200:10, 215 => Contours between values 50 and 200 every 10, with a contour at 215.
        """
        if self._contours is None:

            contour_list = [self.interval_min, self.interval_max, self.interval_spacing]
            if None not in contour_list:
                contour_str = ':'.join([str(k) for k in contour_list])
            if self.fixed_contours is not None:
                contour_str += f", {','.join([str(k) for k in self.fixed_contours])}"
            return contour_str

        else:
            return self._contours

    @contours.setter
    def contours(self, val):
        self.setter_validator("contours", val)

    @property
    def max_distance(self) -> float | None:
        """
        Maximum distance from input data to generate iso surface.
        """
        return self._max_distance

    @max_distance.setter
    def max_distance(self, val):
        self.setter_validator("max_distance", val)

    @property
    def resolution(self) -> float | None:
        """
        Grid size used to generate the iso surface.
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
