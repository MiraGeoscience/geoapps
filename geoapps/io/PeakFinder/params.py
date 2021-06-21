#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from typing import Any, Dict
from uuid import UUID

from ..input_file import InputFile
from ..params import Params
from ..validators import InputValidator
from .constants import default_ui_json, required_parameters, validations


class PeakFinderParams(Params):

    _default_ui_json = default_ui_json

    def __init__(self, **kwargs):

        self.validations: Dict[str, Any] = validations
        self.validator: InputValidator = InputValidator(
            required_parameters, validations
        )
        self.geoh5 = None
        self.objects = None
        self.data = None
        self.tem_checkbox = None
        self.line_field = None
        self.line_id = None
        self.smoothing = None
        self.center = None
        self.width = None
        self.ga_group_name = None
        self.monitoring_directory = None
        self.run_command = None
        self.conda_environment = None
        self._input_file = InputFile()
        self._input_file.input_from_dict(
            self.default_ui_json,
            required_parameters=required_parameters,
            validations=validations,
        )

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
    def data(self):
        return self._data

    @data.setter
    def data(self, val):
        self.setter_validator("data", val)

    @property
    def line_field(self):
        return self._line_field

    @line_field.setter
    def line_field(self, val):
        self.setter_validator(
            "line_field", val, fun=lambda x: UUID(x) if isinstance(x, str) else x
        )

    @property
    def line_id(self):
        return self._line_id

    @line_id.setter
    def line_id(self, val):
        self.setter_validator("line_id", val)

    @property
    def smoothing(self):
        return self._smoothing

    @smoothing.setter
    def smoothing(self, val):
        self.setter_validator("smoothing", val)

    @property
    def tem_checkbox(self):
        return self._tem_checkbox

    @tem_checkbox.setter
    def tem_checkbox(self, val):
        self.setter_validator("tem_checkbox", val)

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
    def ga_group_name(self):
        return self._ga_group_name

    @ga_group_name.setter
    def ga_group_name(self, val):
        self.setter_validator("ga_group_name", val)

    def _init_params(self, inputfile: InputFile) -> None:
        """ Wraps Params._init_params. """
        super()._init_params(inputfile, required_parameters, validations)
