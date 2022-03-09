#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from copy import deepcopy

from geoh5py.shared import Entity
from geoh5py.ui_json import InputFile

from ..params import Params
from ..validation import InputFreeformValidation
from . import default_ui_json, defaults, template_dict, validations


class OctreeParams(Params):
    _default_ui_json = deepcopy(default_ui_json)
    _defaults = deepcopy(defaults)
    _free_parameter_keys = ["object", "levels", "type", "distance"]
    _free_parameter_identifier = "refinement"
    _validations = validations

    def __init__(self, input_file=None, **kwargs):
        self._objects = None
        self._u_cell_size = None
        self._v_cell_size = None
        self._w_cell_size = None
        self._horizontal_padding = None
        self._vertical_padding = None
        self._depth_core = None
        self._ga_group_name = None

        if input_file is None:
            free_param_dict = {}
            for key, value in kwargs.items():
                if (
                    self._free_parameter_identifier in key.lower()
                    and "object" in key.lower()
                ):
                    group = key.replace("object", "").rstrip()
                    free_param_dict[group] = deepcopy(template_dict)

            # Add at least one refinements
            if len(free_param_dict) == 0:
                free_param_dict["Refinenement A"] = deepcopy(template_dict)

            ui_json = deepcopy(self._default_ui_json)
            for group, forms in free_param_dict.items():
                for key, form in forms.items():
                    form["group"] = group
                    ui_json[f"{group} {key}"] = form
                    self._defaults[f"{group} {key}"] = form["value"]

            input_file = InputFile(
                ui_json=ui_json,
                data=self._defaults,
                validations=self.validations,
                validation_options={"disabled": True},
            )

        super().__init__(input_file=input_file, **kwargs)

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
        self.setter_validator("objects", val, fun=self._uuid_promoter)

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
