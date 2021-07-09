#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from typing import Any, Dict
from uuid import UUID

from geoapps.io.Octree import (
    OctreeValidator,
    default_ui_json,
    required_parameters,
    validations,
)

from ..input_file import InputFile
from ..params import Params


class OctreeParams(Params):

    _default_ui_json = default_ui_json

    def __init__(self, **kwargs):

        self.validations: Dict[str, Any] = validations
        self.validator: OctreeValidator = OctreeValidator(
            required_parameters, validations
        )
        self.geoh5 = None
        self.objects = None
        self.u_cell_size = None
        self.v_cell_size = None
        self.w_cell_size = None
        self.horizontal_padding = None
        self.vertical_padding = None
        self.depth_core = None
        self.ga_group_name = None
        self.monitoring_directory = None
        self.workspace_geoh5 = None
        self.run_command = None
        self.run_command_boolean = None
        self.conda_environment = None
        self.conda_environment_boolean = None
        self._refinements = None
        self._input_file = InputFile()

        super().__init__(**kwargs)

    def _set_defaults(self) -> None:
        """ Wraps Params._set_defaults """
        return super()._set_defaults(self.default_ui_json)

    def default(self, param) -> Any:
        """ Wraps Params.default. """
        return super().default(self.default_ui_json, param)

    @property
    def objects(self):
        return self._objects

    @objects.setter
    def objects(self, val):
        self.setter_validator("objects", val, fun=lambda x: UUID(x))

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
    def refinements(self):
        if getattr(self, "_refinements", None) is None:
            self._refinements = self.validator.refinements

        return self._refinements

    def _init_params(self, inputfile: InputFile) -> None:
        """ Wraps Params._init_params. """
        super()._init_params(inputfile, required_parameters, validations)
