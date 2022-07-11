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
from geoapps.interpolation.constants import default_ui_json, defaults, validations


class DataInterpolationParams(BaseParams):
    """
    Parameter class for data interpolation application.
    """

    def __init__(self, input_file=None, **kwargs):
        self._default_ui_json = deepcopy(default_ui_json)
        self._defaults = deepcopy(defaults)
        self._validations = validations
        self._objects = None
        self._data = None
        self._method = None
        self._skew_angle = None
        self._skew_factor = None
        self._space = None
        self._max_distance = None
        self._xy_extent = None
        self._topo_object = None
        self._topo_data = None
        self._max_depth = None
        self._no_data_value = None
        self._out_object = None
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
        Object to interpolate from.
        """
        return self._objects

    @objects.setter
    def objects(self, val):
        self.setter_validator("objects", val, fun=self._uuid_promoter)

    @property
    def data(self) -> Data | None:
        """
        Data to interpolate from.
        """
        return self._data

    @data.setter
    def data(self, val):
        self.setter_validator("data", val)

    @property
    def method(self) -> str | None:
        """
        Method of interpolation: "nearest" or "inverse distance".
        """
        return self._method

    @method.setter
    def method(self, val):
        self.setter_validator("method", val)

    @property
    def skew_angle(self) -> float | None:
        """
        Skew angle for inverse distance interpolation.
        """
        return self._skew_angle

    @skew_angle.setter
    def skew_angle(self, val):
        self.setter_validator("skew_angle", val)

    @property
    def skew_factor(self) -> float | None:
        """
        Skew factor for inverse distance interpolation.
        """
        return self._skew_factor

    @skew_factor.setter
    def skew_factor(self, val):
        self.setter_validator("skew_factor", val)

    @property
    def space(self) -> str | None:
        """
        Scaling: "log" or "linear".
        """
        return self._space

    @space.setter
    def space(self, val):
        self.setter_validator("space", val)

    @property
    def max_distance(self) -> float | None:
        """
        Max distance for horizontal extent.
        """
        return self._max_distance

    @max_distance.setter
    def max_distance(self, val):
        self.setter_validator("max_distance", val)

    @property
    def xy_extent(self) -> ObjectBase | None:
        """
        Object hull.
        """
        return self._xy_extent

    @xy_extent.setter
    def xy_extent(self, val):
        self.setter_validator("xy_extent", val, fun=self._uuid_promoter)

    @property
    def topo_object(self) -> ObjectBase | None:
        """
        Object defining topography.
        """
        return self._topo_object

    @topo_object.setter
    def topo_object(self, val):
        self.setter_validator("topo_object", val, fun=self._uuid_promoter)

    @property
    def topo_data(self) -> Data | None:
        """
        Data defining topography.
        """
        return self._topo_data

    @topo_data.setter
    def topo_data(self, val):
        self.setter_validator("topo_data", val)

    @property
    def max_depth(self) -> float | None:
        """
        Max depth for vertical extent.
        """
        return self._max_depth

    @max_depth.setter
    def max_depth(self, val):
        self.setter_validator("max_depth", val)

    @property
    def no_data_value(self) -> float | None:
        """
        Value to replace nans.
        """
        return self._no_data_value

    @no_data_value.setter
    def no_data_value(self, val):
        self.setter_validator("no_data_value", val)

    @property
    def out_object(self) -> ObjectBase | None:
        """
        Object to interpolate to.
        """
        return self._out_object

    @out_object.setter
    def out_object(self, val):
        self.setter_validator("out_object", val, fun=self._uuid_promoter)

    @property
    def ga_group_name(self) -> str | None:
        """
        Output label.
        """
        return self._ga_group_name

    @ga_group_name.setter
    def ga_group_name(self, val):
        self.setter_validator("ga_group_name", val)
