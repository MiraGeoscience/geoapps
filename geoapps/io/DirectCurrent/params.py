#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from copy import deepcopy

from geoapps.io.Inversion import InversionParams

from .constants import (
    default_ui_json,
    forward_defaults,
    forward_ui_json,
    inversion_defaults,
    inversion_ui_json,
    required_parameters,
    validations,
)


class DirectCurrentParams(InversionParams):

    _required_parameters = required_parameters
    _validations = validations
    _validators = None
    _forward_defaults = forward_defaults
    _inversion_defaults = inversion_defaults
    forward_ui_json = forward_ui_json
    inversion_ui_json = inversion_ui_json
    _directive_list = [
        "UpdateSensitivityWeights",
        "Update_IRLS",
        "BetaEstimate_ByEig",
        "UpdatePreconditioner",
        "SaveIterationsGeoH5",
    ]

    def __init__(
        self, input_file=None, default=True, validate=True, validator_opts={}, **kwargs
    ):

        self.validate = False
        self.default_ui_json = deepcopy(default_ui_json)
        self.inversion_type = "direct current"
        self.potential_channel_bool = None
        self.potential_channel = None
        self.potential_uncertainty = None
        self.out_group = None

        super().__init__(input_file, default, validate, validator_opts, **kwargs)

    @property
    def inversion_type(self):
        return self._inversion_type

    @inversion_type.setter
    def inversion_type(self, val):
        self.setter_validator("inversion_type", val)

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
