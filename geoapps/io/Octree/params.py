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
    def __init__(self):
        super().__init__()

        self.validations: Dict[str, Any] = validations
        self.validator: OctreeValidator = OctreeValidator(
            required_parameters, validations
        )
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
        self.geoh5 = None
        self.run_command = None
        self.run_command_boolean = None
        self.conda_environment = None
        self.conda_environment_boolean = None
        self._refinements = None
        self._input_file = InputFile()
        self._default_ui_json = default_ui_json

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
    def u_cell_size(self):
        return self._u_cell_size

    @u_cell_size.setter
    def u_cell_size(self, val):
        if val is None:
            self._u_cell_size = val
            return
        p = "u_cell_size"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._u_cell_size = val

    @property
    def v_cell_size(self):
        return self._v_cell_size

    @v_cell_size.setter
    def v_cell_size(self, val):
        if val is None:
            self._v_cell_size = val
            return
        p = "v_cell_size"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._v_cell_size = val

    @property
    def w_cell_size(self):
        return self._w_cell_size

    @w_cell_size.setter
    def w_cell_size(self, val):
        if val is None:
            self._w_cell_size = val
            return
        p = "w_cell_size"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._w_cell_size = val

    @property
    def horizontal_padding(self):
        return self._horizontal_padding

    @horizontal_padding.setter
    def horizontal_padding(self, val):
        if val is None:
            self._horizontal_padding = val
            return
        p = "horizontal_padding"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._horizontal_padding = val

    @property
    def vertical_padding(self):
        return self._vertical_padding

    @vertical_padding.setter
    def vertical_padding(self, val):
        if val is None:
            self._vertical_padding = val
            return
        p = "vertical_padding"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._vertical_padding = val

    @property
    def depth_core(self):
        return self._depth_core

    @depth_core.setter
    def depth_core(self, val):
        if val is None:
            self._depth_core = val
            return
        p = "depth_core"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._depth_core = val

    @property
    def refinements(self):
        if getattr(self, "_refinements", None) is None:
            self._refinements = self.validator.refinements

        return self._refinements

    def _init_params(self, inputfile: InputFile) -> None:
        """ Wraps Params._init_params. """
        super()._init_params(inputfile, required_parameters, validations)
