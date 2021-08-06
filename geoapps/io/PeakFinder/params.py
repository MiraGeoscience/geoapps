#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from typing import Any
from uuid import UUID

from geoh5py.workspace import Workspace

from ..input_file import InputFile
from ..params import Params
from ..validators import InputFreeformValidator
from .constants import default_ui_json, defaults, required_parameters, validations


class PeakFinderParams(Params):

    defaults = defaults
    _default_ui_json = default_ui_json
    _required_parameters = required_parameters
    _validations = validations
    param_names = list(default_ui_json.keys())
    _free_param_keys: list[str] = ["data", "color"]

    def __init__(self, validate=True, **kwargs):

        self.validator: InputFreeformValidator = InputFreeformValidator(
            required_parameters, validations, free_params_keys=self._free_param_keys
        )

        self._title = None
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
        self._free_params_dict = None

        super().__init__(validate, **kwargs)

    def _set_defaults(self) -> None:
        """Wraps Params._set_defaults"""
        return super()._set_defaults(self.default_ui_json)

    def default(self, param) -> Any:
        """Wraps Params.default."""
        return super().default(self.default_ui_json, param)

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, val):
        self.setter_validator("title", val)

    @property
    def geoh5(self):
        return self._geoh5

    @geoh5.setter
    def geoh5(self, val):
        self.setter_validator(
            "geoh5", val, promote_type=str, fun=lambda x: Workspace(x)
        )

    @property
    def objects(self):
        return self._objects

    @objects.setter
    def objects(self, val):
        self.setter_validator("objects", val, promote_type=str, fun=lambda x: UUID(x))

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, val):
        self.setter_validator("data", val, promote_type=str, fun=lambda x: UUID(x))

    @property
    def flip_sign(self):
        return self._flip_sign

    @flip_sign.setter
    def flip_sign(self, val):
        self.setter_validator("flip_sign", val)

    @property
    def line_field(self):
        return self._line_field

    @line_field.setter
    def line_field(self, val):
        self.setter_validator(
            "line_field", val, promote_type=str, fun=lambda x: UUID(x)
        )

    @property
    def tem_checkbox(self):
        return self._tem_checkbox

    @tem_checkbox.setter
    def tem_checkbox(self, val):
        self.setter_validator("tem_checkbox", val)

    @property
    def system(self):
        return self._system

    @system.setter
    def system(self, val):
        self.setter_validator("system", val)

    @property
    def smoothing(self):
        return self._smoothing

    @smoothing.setter
    def smoothing(self, val):
        self.setter_validator("smoothing", val)

    @property
    def min_amplitude(self):
        return self._min_amplitude

    @min_amplitude.setter
    def min_amplitude(self, val):
        self.setter_validator("min_amplitude", val)

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
    def max_migration(self):
        return self._max_migration

    @max_migration.setter
    def max_migration(self, val):
        self.setter_validator("max_migration", val)

    @property
    def min_channels(self):
        return self._min_channels

    @min_channels.setter
    def min_channels(self, val):
        self.setter_validator("min_channels", val)

    @property
    def ga_group_name(self):
        return self._ga_group_name

    @ga_group_name.setter
    def ga_group_name(self, val):
        self.setter_validator("ga_group_name", val)

    @property
    def structural_markers(self):
        return self._structural_markers

    @structural_markers.setter
    def structural_markers(self, val):
        self.setter_validator("structural_markers", val)

    @property
    def line_id(self):
        return self._line_id

    @line_id.setter
    def line_id(self, val):
        self.setter_validator("line_id", val)

    @property
    def group_auto(self):
        return self._group_auto

    @group_auto.setter
    def group_auto(self, val):
        self.setter_validator("group_auto", val)

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, val):
        self.setter_validator("center", val)

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, val):
        self.setter_validator("width", val)

    @property
    def run_command(self):
        return self._run_command

    @run_command.setter
    def run_command(self, val):
        self.setter_validator("run_command", val)

    @property
    def run_command_boolean(self):
        return self._run_command_boolean

    @run_command_boolean.setter
    def run_command_boolean(self, val):
        self.setter_validator("run_command_boolean", val)

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
    def workspace_geoh5(self):
        return self._workspace_geoh5

    @workspace_geoh5.setter
    def workspace_geoh5(self, val):
        self.setter_validator(
            "workspace_geoh5", val, promote_type=str, fun=lambda x: Workspace(x)
        )

    @property
    def workspace(self):
        return self._workspace

    @workspace.setter
    def workspace(self, val):
        self.setter_validator(
            "workspace", val, promote_type=str, fun=lambda x: Workspace(x)
        )

    @property
    def monitoring_directory(self):
        return self._monitoring_directory

    @monitoring_directory.setter
    def monitoring_directory(self, val):
        self.setter_validator("monitoring_directory", val)
