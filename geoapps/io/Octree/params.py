#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from typing import Any
from uuid import UUID

from ..input_file import InputFile
from ..params import Params
from . import default_ui_json, defaults, required_parameters, validations
from .validators import OctreeValidator


class OctreeParams(Params):

    defaults = defaults
    _default_ui_json = default_ui_json
    _required_parameters = required_parameters
    _validations = validations
    param_names = list(default_ui_json.keys())

    def __init__(self, validate=True, **kwargs):

        self.validator: OctreeValidator = OctreeValidator(
            required_parameters, validations
        )
        self.title = None
        self.geoh5 = None
        self.objects = None
        self.u_cell_size = None
        self.v_cell_size = None
        self.w_cell_size = None
        self.horizontal_padding = None
        self.vertical_padding = None
        self.depth_core = None
        self.ga_group_name = None
        self.run_command = None
        self.run_command_boolean = None
        self.monitoring_directory = None
        self.workspace_geoh5 = None
        self.conda_environment = None
        self.conda_environment_boolean = None
        self.workspace = None
        self._refinements = None

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
        self.setter_validator("geoh5", val)

    @property
    def objects(self):
        return self._objects

    @objects.setter
    def objects(self, val):
        self.setter_validator("objects", val, promote_type=str, fun=lambda x: UUID(x))

    @property
    def u_cell_size(self):
        return self._u_cell_size

    @u_cell_size.setter
    def u_cell_size(self, val):
        self.setter_validator("u_cell_size", val)

    @property
    def v_cell_size(self):
        return self._v_cell_size

    @v_cell_size.setter
    def v_cell_size(self, val):
        self.setter_validator("v_cell_size", val)

    @property
    def w_cell_size(self):
        return self._w_cell_size

    @w_cell_size.setter
    def w_cell_size(self, val):
        self.setter_validator("w_cell_size", val)

    @property
    def horizontal_padding(self):
        return self._horizontal_padding

    @horizontal_padding.setter
    def horizontal_padding(self, val):
        self.setter_validator("horizontal_padding", val)

    @property
    def vertical_padding(self):
        return self._vertical_padding

    @vertical_padding.setter
    def vertical_padding(self, val):
        self.setter_validator("vertical_padding", val)

    @property
    def depth_core(self):
        return self._depth_core

    @depth_core.setter
    def depth_core(self, val):
        self.setter_validator("depth_core", val)

    @property
    def ga_group_name(self):
        return self._ga_group_name

    @ga_group_name.setter
    def ga_group_name(self, val):
        self.setter_validator("ga_group_name", val)

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
    def monitoring_directory(self):
        return self._monitoring_directory

    @monitoring_directory.setter
    def monitoring_directory(self, val):
        self.setter_validator("monitoring_directory", val)

    @property
    def workspace_geoh5(self):
        return self._workspace_geoh5

    @workspace_geoh5.setter
    def workspace_geoh5(self, val):
        self.setter_validator("workspace_geoh5", val)

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
    def workspace(self):
        return self._workspace

    @workspace.setter
    def workspace(self, val):
        self.setter_validator("workspace", val)

    @property
    def refinements(self):
        if getattr(self, "_refinements", None) is None:
            self._refinements = self.validator.refinements

        return self._refinements
