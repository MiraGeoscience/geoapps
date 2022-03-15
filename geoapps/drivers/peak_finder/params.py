#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from copy import deepcopy

from geoh5py.ui_json import InputFile

from geoapps.drivers.base_params import BaseParams

from .constants import default_ui_json, defaults, template_dict, validations


class PeakFinderParams(BaseParams):
    """
    Parameter class for peak finder application.
    """

    def __init__(self, input_file=None, **kwargs):
        self._default_ui_json = deepcopy(default_ui_json)
        self._defaults = deepcopy(defaults)
        self._free_parameter_keys: list = ["data", "color"]
        self._free_parameter_identifier: str = "group"
        self._validations = validations
        self._objects = None
        self._data = None
        self._flip_sign = None
        self._line_field = None
        self._tem_checkbox = None
        self._system = None
        self._smoothing = None
        self._min_amplitude = None
        self._min_value = None
        self._min_width = None
        self._max_migration = None
        self._min_channels = None
        self._ga_group_name = None
        self._structural_markers = None
        self._line_id = None
        self._group_auto = None
        self._center = None
        self._width = None
        self._template_data = None
        self._template_color = None
        self._plot_result = True

        if input_file is None:
            free_param_dict = {}
            for key, value in kwargs.items():
                if (
                    self._free_parameter_identifier in key.lower()
                    and "data" in key.lower()
                ):
                    group = key.replace("data", "").rstrip()
                    free_param_dict[group] = deepcopy(template_dict)

            ui_json = deepcopy(self._default_ui_json)
            for group, forms in free_param_dict.items():
                for key, form in forms.items():
                    form["group"] = group
                    ui_json[f"{group} {key}"] = form
                    self._defaults[f"{group} {key}"] = form["value"]

            input_file = InputFile(
                ui_json=ui_json,
                validations=self.validations,
                validation_options={"disabled": True},
            )

        super().__init__(input_file=input_file, **kwargs)

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, val):
        self.setter_validator("center", val)

    @property
    def conda_environment(self):
        return self._conda_environment

    @conda_environment.setter
    def conda_environment(self, val):
        self.setter_validator("conda_environment", val)

    @property
    def conda_environment_boolean(self):
        return self._conda_environment_boolean

    @conda_environment_boolean.setter
    def conda_environment_boolean(self, val):
        self.setter_validator("conda_environment_boolean", val)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, val):
        self.setter_validator("data", val, fun=self._uuid_promoter)

    @property
    def flip_sign(self):
        return self._flip_sign

    @flip_sign.setter
    def flip_sign(self, val):
        self.setter_validator("flip_sign", val)

    @property
    def ga_group_name(self):
        return self._ga_group_name

    @ga_group_name.setter
    def ga_group_name(self, val):
        self.setter_validator("ga_group_name", val)

    @property
    def group_auto(self):
        return self._group_auto

    @group_auto.setter
    def group_auto(self, val):
        self.setter_validator("group_auto", val)

    @property
    def line_field(self):
        return self._line_field

    @line_field.setter
    def line_field(self, val):
        self.setter_validator("line_field", val, fun=self._uuid_promoter)

    @property
    def line_id(self):
        return self._line_id

    @line_id.setter
    def line_id(self, val):
        self.setter_validator("line_id", val)

    @property
    def max_migration(self):
        return self._max_migration

    @max_migration.setter
    def max_migration(self, val):
        self.setter_validator("max_migration", val)

    @property
    def min_amplitude(self):
        return self._min_amplitude

    @min_amplitude.setter
    def min_amplitude(self, val):
        self.setter_validator("min_amplitude", val)

    @property
    def min_channels(self):
        return self._min_channels

    @min_channels.setter
    def min_channels(self, val):
        self.setter_validator("min_channels", val)

    @property
    def min_value(self):
        return self._min_value

    @min_value.setter
    def min_value(self, val):
        self.setter_validator("min_value", val)

    @property
    def min_width(self):
        return self._min_width

    @min_width.setter
    def min_width(self, val):
        self.setter_validator("min_width", val)

    @property
    def monitoring_directory(self):
        return self._monitoring_directory

    @monitoring_directory.setter
    def monitoring_directory(self, val):
        self.setter_validator("monitoring_directory", val)

    @property
    def objects(self):
        return self._objects

    @objects.setter
    def objects(self, val):
        self.setter_validator("objects", val, fun=self._uuid_promoter)

    @property
    def plot_result(self):
        return self._plot_result

    @plot_result.setter
    def plot_result(self, val):
        self._plot_result = val

    @property
    def smoothing(self):
        return self._smoothing

    @smoothing.setter
    def smoothing(self, val):
        self.setter_validator("smoothing", val)

    @property
    def structural_markers(self):
        return self._structural_markers

    @structural_markers.setter
    def structural_markers(self, val):
        self.setter_validator("structural_markers", val)

    @property
    def system(self):
        return self._system

    @system.setter
    def system(self, val):
        self.setter_validator("system", val)

    @property
    def tem_checkbox(self):
        return self._tem_checkbox

    @tem_checkbox.setter
    def tem_checkbox(self, val):
        self.setter_validator("tem_checkbox", val)

    @property
    def template_data(self):
        return self._template_data

    @template_data.setter
    def template_data(self, val):
        self.setter_validator("template_data", val)

    @property
    def template_color(self):
        return self._template_color

    @template_color.setter
    def template_color(self, val):
        self.setter_validator("template_color", val)

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, val):
        self.setter_validator("title", val)

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, val):
        self.setter_validator("width", val)
