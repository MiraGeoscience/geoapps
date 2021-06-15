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
        self.lines = None
        self.smoothing = None
        self.center = None
        self.width = None
        self.ga_group_name = None
        self.monitoring_directory = None
        self.run_command = None
        self.conda_environment = None
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
        if val is None:
            self._objects = val
            return

        p = "objects"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._objects = UUID(val)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, val):
        if val is None:
            self._data = val
            return
        p = "data"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._data = UUID(val) if isinstance(val, str) else val

    @property
    def lines(self):
        return self._lines

    @lines.setter
    def lines(self, val):
        if val is None:
            self._lines = val
            return
        p = "lines"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._lines = UUID(val) if isinstance(val, str) else val

    @property
    def smoothing(self):
        return self._smoothing

    @smoothing.setter
    def smoothing(self, val):
        if val is None:
            self._smoothing = val
            return
        p = "smoothing"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._smoothing = val

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, val):
        if val is None:
            self._center = val
            return
        p = "center"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._center = val

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, val):
        if val is None:
            self._width = val
            return
        p = "width"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._width = val

    @property
    def ga_group_name(self):
        return self._ga_group_name

    @ga_group_name.setter
    def ga_group_name(self, val):
        if val is None:
            self._ga_group_name = val
            return
        p = "ga_group_name"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._ga_group_name = val

    def _init_params(self, inputfile: InputFile) -> None:
        """ Wraps Params._init_params. """
        super()._init_params(inputfile, required_parameters, validations)
