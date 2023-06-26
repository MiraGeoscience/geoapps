#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from geoh5py.groups.simpeg_group import SimPEGGroup

from geoapps.inversion.params import InversionBaseParams


class BaseJointParams(InversionBaseParams):
    """
    Parameter class for gravity->density inversion.
    """

    _physical_property = ""

    def __init__(self, input_file=None, forward_only=False, **kwargs):
        self._group_a = None
        self._group_b = None
        self._group_c = None

        super().__init__(input_file=input_file, forward_only=forward_only, **kwargs)

    @property
    def group_a(self):
        """First SimPEGGroup inversion."""
        return self._group_a

    @group_a.setter
    def group_a(self, val: SimPEGGroup):
        self.setter_validator("group_a", val, fun=self._uuid_promoter)

    @property
    def group_b(self):
        """Second SimPEGGroup inversion."""
        return self._group_b

    @group_b.setter
    def group_b(self, val: SimPEGGroup):
        self.setter_validator("group_b", val, fun=self._uuid_promoter)

    @property
    def group_c(self):
        """Third SimPEGGroup inversion."""
        return self._group_c

    @group_c.setter
    def group_c(self, val: SimPEGGroup):
        self.setter_validator("group_c", val, fun=self._uuid_promoter)
