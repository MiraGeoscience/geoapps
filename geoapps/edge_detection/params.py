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
from geoapps.edge_detection.constants import default_ui_json, defaults, validations


class EdgeDetectionParams(BaseParams):
    """
    Parameter class for edge detection application.
    """

    def __init__(self, input_file=None, **kwargs):
        self._default_ui_json = deepcopy(default_ui_json)
        self._defaults = deepcopy(defaults)
        self._validations = validations
        self._objects = None
        self._data = None
        self._line_length = None
        self._line_gap = None
        self._sigma = None
        self._threshold = None
        self._window_size = None
        self._window_azimuth = None
        self._window_center_x = None
        self._window_center_y = None
        self._window_width = None
        self._window_height = None
        self._colorbar = None
        self._zoom_extent = None
        self._export_as = None
        self._ga_group_name = None
        self._resolution = None

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
    def line_length(self) -> int | None:
        """
        Minimum accepted pixel length of detected lines. (Hough)
        """
        return self._line_length

    @line_length.setter
    def line_length(self, val):
        self.setter_validator("line_length", val)

    @property
    def line_gap(self) -> int | None:
        """
        Maximum gap between pixels to still form a line. (Hough)
        """
        return self._line_gap

    @line_gap.setter
    def line_gap(self, val):
        self.setter_validator("line_gap", val)

    @property
    def sigma(self) -> float | None:
        """
        Standard deviation of the Gaussian filter. (Canny)
        """
        return self._sigma

    @sigma.setter
    def sigma(self, val):
        self.setter_validator("sigma", val)

    @property
    def threshold(self) -> int | None:
        """
        Value threshold. (Hough)
        """
        return self._threshold

    @threshold.setter
    def threshold(self, val):
        self.setter_validator("threshold", val)

    @property
    def window_size(self) -> int | None:
        """
        Window size.
        """
        return self._window_size

    @window_size.setter
    def window_size(self, val):
        self.setter_validator("window_size", val)

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
    def colorbar(self) -> bool | None:
        """
        Display the colorbar.
        """
        return self._colorbar

    @colorbar.setter
    def colorbar(self, val):
        self.setter_validator("colorbar", val)

    @property
    def zoom_extent(self) -> bool | None:
        """
        Set plotting limits to the selection box.
        """
        return self._zoom_extent

    @zoom_extent.setter
    def zoom_extent(self, val):
        self.setter_validator("zoom_extent", val)

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
    def ga_group_name(self) -> str | None:
        """
        Name to save geoh5 file as.
        """
        return self._ga_group_name

    @ga_group_name.setter
    def ga_group_name(self, val):
        self.setter_validator("ga_group_name", val)

    @property
    def resolution(self) -> float | None:
        """
        Minimum data separation (m).
        """
        return self._resolution

    @resolution.setter
    def resolution(self, val):
        self.setter_validator("resolution", val)

    def edge_args(self):
        return (
            self.objects,
            self.data,
            self.sigma,
            self.line_length,
            self.threshold,
            self.line_gap,
            self.window_size,
            self.window_center_x,
            self.window_center_y,
            self.window_width,
            self.window_height,
            self.window_azimuth,
            self.resolution,
        )
