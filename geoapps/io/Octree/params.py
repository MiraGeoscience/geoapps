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
from ..validators import InputFreeformValidator
from . import default_ui_json, defaults, required_parameters, validations


class OctreeParams(Params):

    _required_parameters = required_parameters
    _validations = validations
    default_ui_json = default_ui_json
    param_names = list(default_ui_json.keys())
    _free_param_keys = ["object", "levels", "type", "distance"]
    _free_param_identifier = "refinement"

    def __init__(self, input_file=None, validate=True, **kwargs):
        super().__init__(input_file, validate, **kwargs)

        free_params_dict = {}
        for k, v in kwargs.items():
            if self._free_param_identifier in k.lower():
                for param in self._free_param_keys:
                    if param not in v.keys():
                        raise ValueError(
                            f"Provided free parameter {k} should have a key argument {param}"
                        )
                free_params_dict[k] = v

        if any(free_params_dict):
            self._free_params_dict = free_params_dict

        self._initialize(kwargs)

    def _initialize(self, kwargs):

        if self.input_file is not None:
            self.input_file.data.update(kwargs)
        elif kwargs:
            self.input_file = InputFile.from_dict(kwargs)
        else:
            self.input_file = InputFile()

        self.apply_defaults()
        self.workspace = self.input_file.data["geoh5"]
        self.validator: InputFreeformValidator = InputFreeformValidator(
            required_parameters, validations, self.workspace,
            free_params_keys=self._free_param_keys, input=self.input_file.data
        )
        self.update(self.input_file.data)

    def apply_defaults(self):
        validate = self.validate
        self.validate = False
        self.defaults = defaults
        self.param_names = list(self.defaults.keys())
        self.update(self.defaults)
        self.validate = validate


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
    def objects(self):
        return self._objects

    @objects.setter
    def objects(self, val):
        self.setter_validator(
            "objects", val, fun=lambda x: UUID(x) if isinstance(val, str) else x
        )

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
