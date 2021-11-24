#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from typing import Any
from uuid import UUID
from copy import deepcopy

from geoh5py.shared import Entity

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
    _free_param_dict = {}

    def __init__(self, input_file=None, default=True, validate=True, **kwargs):

        self.validate = False
        self.default_ui_json = deepcopy(default_ui_json)
        self.title = None
        self.objects = None
        self.u_cell_size = None
        self.v_cell_size = None
        self.w_cell_size = None
        self.horizontal_padding = None
        self.vertical_padding = None
        self.depth_core = None
        self.ga_group_name = None

        super().__init__(input_file, default, validate, **kwargs)

        self._initialize(kwargs)

    def _initialize(self, params_dict):

        if self.input_file:
            params_dict = dict(self.input_file.data, **params_dict)

        # Use forward_only state to determine defaults and default_ui_json.
        self.defaults = defaults
        self.param_names = list(self.defaults.keys())

        # Superimpose params_dict onto defaults.
        if self.default:
            params_dict = dict(self.defaults, **params_dict)

        # Validate.
        if self.validate:
            self.workspace = params_dict["geoh5"]
            self.associations = self.get_associations(params_dict)
            self.validator: InputFreeformValidator = InputFreeformValidator(
                required_parameters,
                validations,
                self.workspace,
                free_params_keys=self._free_param_keys,
            )
            self.validator.validate_chunk(params_dict, self.associations)

        # Set params attributes from validated input.
        self.update(params_dict)

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

    def update(self, params_dict: Dict[str, Any]):
        """Update parameters with dictionary contents."""

        # Pull out workspace data for validations and forward_only for defaults.

        if "geoh5" in params_dict.keys():
            if params_dict["geoh5"] is not None:
                setattr(self, "workspace", params_dict["geoh5"])

        free_param_dict = {}
        for key, value in params_dict.items():

            if key == "workspace":
                continue  # ignores deprecated workspace name

            if "Template" in key:
                continue

            # Update default_ui_json and store free_param_groups for app
            if (
                self._free_param_identifier in key.lower()
                and key not in self.default_ui_json
            ):
                for param in self._free_param_keys:
                    if param in key.lower():
                        group = key.lower().replace(param, "").rstrip()

                        if group not in free_param_dict:
                            free_param_dict[group] = {}

                        free_param_dict[group][param] = value
                        self.default_ui_json[key] = self.default_ui_json[
                            f"Template {param.capitalize()}"
                        ]
                        self.default_ui_json[key]["group"] = group
                        break

            if isinstance(value, dict):
                field = "value"
                if "isValue" in value.keys():
                    if not value["isValue"]:
                        field = "property"
                setattr(self, key, value[field])
            else:
                if isinstance(value, Entity):
                    setattr(self, key, value.uid)
                else:
                    setattr(self, key, value)

        # Clear Template

        for key in list(self.default_ui_json.keys()):
            if "Template" in key:
                del self.default_ui_json[key]

        self._free_param_dict = free_param_dict
        self.param_names = list(self.default_ui_json.keys())
