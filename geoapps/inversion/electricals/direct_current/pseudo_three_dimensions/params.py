#  Copyright (c) 2024 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from copy import deepcopy

from geoapps.inversion.electricals.direct_current.pseudo_three_dimensions.constants import (
    default_ui_json,
    forward_defaults,
    inversion_defaults,
    validations,
)
from geoapps.inversion.electricals.params import BasePseudo3DParams


class DirectCurrentPseudo3DParams(BasePseudo3DParams):
    """
    Parameter class for electrical->conductivity inversion.
    """

    _physical_property = "conductivity"
    _inversion_type = "direct current 3d"

    def __init__(self, input_file=None, forward_only=False, **kwargs):
        self._default_ui_json = deepcopy(default_ui_json)
        self._forward_defaults = deepcopy(forward_defaults)
        self._inversion_defaults = deepcopy(inversion_defaults)
        self._validations = validations
        self._potential_channel_bool = None
        self._potential_channel = None
        self._potential_uncertainty = None

        super().__init__(input_file=input_file, forward_only=forward_only, **kwargs)

    @property
    def potential_channel_bool(self):
        return self._potential_channel_bool

    @potential_channel_bool.setter
    def potential_channel_bool(self, val):
        self.setter_validator("potential_channel_bool", val)

    @property
    def potential_channel(self):
        return self._potential_channel

    @potential_channel.setter
    def potential_channel(self, val):
        self.setter_validator("potential_channel", val, fun=self._uuid_promoter)

    @property
    def potential_uncertainty(self):
        return self._potential_uncertainty

    @potential_uncertainty.setter
    def potential_uncertainty(self, val):
        self.setter_validator("potential_uncertainty", val, fun=self._uuid_promoter)
