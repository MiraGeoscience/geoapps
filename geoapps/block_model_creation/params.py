#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from copy import deepcopy

from geoh5py.objects import ObjectBase
from geoh5py.ui_json import InputFile

from geoapps.block_model_creation.constants import (
    default_ui_json,
    defaults,
    validations,
)
from geoapps.driver_base.params import BaseParams


class BlockModelParams(BaseParams):
    """
    Parameter class for block model creation application.
    """

    def __init__(self, input_file=None, **kwargs):
        self._default_ui_json = deepcopy(default_ui_json)
        self._defaults = deepcopy(defaults)
        self._validations = validations
        self._objects = None
        self._cell_size_x = None
        self._cell_size_y = None
        self._cell_size_z = None
        self._horizontal_padding = None
        self._bottom_padding = None
        self._depth_core = None
        self._expansion_fact = None
        self._new_grid = None
        self._live_link = None
        self._output_path = None

        if input_file is None:
            ui_json = deepcopy(self._default_ui_json)
            input_file = InputFile(
                ui_json=ui_json,
                validations=self.validations,
                validation_options={"disabled": True},
            )
        super().__init__(input_file=input_file, **kwargs)

    @property
    def objects(self) -> ObjectBase | None:
        """
        Input object.
        """
        return self._objects

    @objects.setter
    def objects(self, val):
        self.setter_validator("objects", val, fun=self._uuid_promoter)

    @property
    def cell_size_x(self) -> float | None:
        """
        x cell size for 3D grid.
        """
        return self._cell_size_x

    @cell_size_x.setter
    def cell_size_x(self, val):
        self.setter_validator("cell_size_x", val)

    @property
    def cell_size_y(self) -> float | None:
        """
        y cell size for 3D grid.
        """
        return self._cell_size_y

    @cell_size_y.setter
    def cell_size_y(self, val):
        self.setter_validator("cell_size_y", val)

    @property
    def cell_size_z(self) -> float | None:
        """
        z cell size for 3D grid.
        """
        return self._cell_size_z

    @cell_size_z.setter
    def cell_size_z(self, val):
        self.setter_validator("cell_size_z", val)

    @property
    def horizontal_padding(self) -> float | None:
        """
        Horizontal padding distance for 3D grid.
        """
        return self._horizontal_padding

    @horizontal_padding.setter
    def horizontal_padding(self, val):
        self.setter_validator("horizontal_padding", val)

    @property
    def bottom_padding(self) -> float | None:
        """
        Bottom padding distance for 3D grid.
        """
        return self._bottom_padding

    @bottom_padding.setter
    def bottom_padding(self, val):
        self.setter_validator("bottom_padding", val)

    @property
    def depth_core(self) -> float | None:
        """
        Core depth for 3D grid.
        """
        return self._depth_core

    @depth_core.setter
    def depth_core(self, val):
        self.setter_validator("depth_core", val)

    @property
    def expansion_fact(self) -> float | None:
        """
        Expansion factor for 3D grid.
        """
        return self._expansion_fact

    @expansion_fact.setter
    def expansion_fact(self, val):
        self.setter_validator("expansion_fact", val)

    @property
    def new_grid(self) -> str | None:
        """
        Name of 3D grid.
        """
        return self._new_grid

    @new_grid.setter
    def new_grid(self, val):
        self.setter_validator("new_grid", val)

    @property
    def monitoring_directory(self) -> str | None:
        """
        Output path.
        """
        return self._monitoring_directory

    @monitoring_directory.setter
    def monitoring_directory(self, val):
        self.setter_validator("monitoring_directory", val)
