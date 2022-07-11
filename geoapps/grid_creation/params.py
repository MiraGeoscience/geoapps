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

from geoapps.driver_base.params import BaseParams
from geoapps.grid_creation.constants import default_ui_json, defaults, validations


class GridCreationParams(BaseParams):
    """
    Parameter class for block model creation application.
    """

    def __init__(self, input_file=None, **kwargs):
        self._default_ui_json = deepcopy(default_ui_json)
        self._defaults = deepcopy(defaults)
        self._validations = validations
        self._objects = None
        self._xy_reference = None
        self._core_cell_size = None
        self._padding_distance = None
        self._depth_core = None
        self._expansion_fact = None
        self._new_grid = None
        self._live_link = None

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
    def xy_reference(self) -> ObjectBase | None:
        """
        Lateral extent object for 3D grid.
        """
        return self._xy_reference

    @xy_reference.setter
    def xy_reference(self, val):
        self.setter_validator("xy_reference", val, fun=self._uuid_promoter)

    @property
    def core_cell_size(self) -> str | None:
        """
        Core cell size for 3D grid.
        """
        return self._core_cell_size

    @core_cell_size.setter
    def core_cell_size(self, val):
        self.setter_validator("core_cell_size", val)

    @property
    def padding_distance(self) -> str | None:
        """
        Padding distance for 3D grid.
        """
        return self._padding_distance

    @padding_distance.setter
    def padding_distance(self, val):
        self.setter_validator("padding_distance", val)

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
    def live_link(self) -> bool | None:
        """
        Live link.
        """
        return self._live_link

    @live_link.setter
    def live_link(self, val):
        self.setter_validator("live_link", val)
