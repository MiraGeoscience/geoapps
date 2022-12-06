#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from copy import deepcopy

from geoapps.inversion import InversionBaseParams

from .constants import (
    default_ui_json,
    forward_defaults,
    forward_ui_json,
    inversion_defaults,
    inversion_ui_json,
    validations,
)


class DirectCurrent2DParams(InversionBaseParams):
    """
    Parameter class for electrical->conductivity inversion.
    """

    _directive_list = [
        "UpdateSensitivityWeights",
        "Update_IRLS",
        "BetaEstimate_ByEig",
        "UpdatePreconditioner",
        "SaveIterationsGeoH5",
    ]

    def __init__(self, input_file=None, forward_only=False, **kwargs):
        self._default_ui_json = deepcopy(default_ui_json)
        self._forward_defaults = deepcopy(forward_defaults)
        self._forward_ui_json = deepcopy(forward_ui_json)
        self._inversion_defaults = deepcopy(inversion_defaults)
        self._inversion_ui_json = deepcopy(inversion_ui_json)
        self._inversion_type = "direct current 2d"
        self._validations = validations
        self._potential_channel_bool = None
        self._potential_channel = None
        self._potential_uncertainty = None
        self._line_object = None
        self._line_id = None

        super().__init__(input_file=input_file, forward_only=forward_only, **kwargs)

    @property
    def inversion_type(self):
        return self._inversion_type

    @inversion_type.setter
    def inversion_type(self, val):
        self.setter_validator("inversion_type", val)

    @property
    def line_object(self):
        return self._line_object

    @line_object.setter
    def line_object(self, val):
        self._line_object = val

    @property
    def line_id(self):
        return self._line_id

    @line_id.setter
    def line_id(self, val):
        self._line_id = val

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
