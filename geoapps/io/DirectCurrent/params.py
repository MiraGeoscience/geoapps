#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from uuid import UUID

from geoapps.io.Inversion import InversionParams

from ..validators import InputValidator
from .constants import (
    default_ui_json,
    forward_defaults,
    inversion_defaults,
    required_parameters,
    validations,
)


class DirectCurrentParams(InversionParams):

    _required_parameters = required_parameters
    _validations = validations
    _directive_list = [
        "UpdateSensitivityWeights",
        "Update_IRLS",
        "BetaEstimate_ByEig",
        "UpdatePreconditioner",
        "SaveIterationsGeoH5",
    ]

    def __init__(self, forward=False, **kwargs):

        self.validator: InputValidator = InputValidator(
            required_parameters, validations
        )
        self.inversion_type = "direct_current"
        self.potential_channel_bool = None
        self.potential_channel = None
        self.potential_uncertainty = None
        self.out_group = None

        self.defaults = forward_defaults if forward else inversion_defaults
        self.default_ui_json = {k: default_ui_json[k] for k in self.defaults}
        self.param_names = list(self.default_ui_json.keys())

        for k, v in self.default_ui_json.items():
            if isinstance(v, dict):
                field = "value"
                if "isValue" in v.keys():
                    if not v["isValue"]:
                        field = "property"
                self.default_ui_json[k][field] = self.defaults[k]
            else:
                self.default_ui_json[k] = self.defaults[k]

        super().__init__(**kwargs)

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
        if val is None:
            self._potential_channel_bool = val
            return
        p = "potential_channel_bool"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._potential_channel_bool = val

    @property
    def potential_channel(self):
        return self._potential_channel

    @potential_channel.setter
    def potential_channel(self, val):
        self.setter_validator(
            "potential_channel", val, fun=lambda x: UUID(x) if isinstance(x, str) else x
        )

    @property
    def potential_uncertainty(self):
        return self._potential_uncertainty

    @potential_uncertainty.setter
    def potential_uncertainty(self, val):
        self.setter_validator(
            "potential_uncertainty",
            val,
            fun=lambda x: UUID(x) if isinstance(x, str) else x,
        )
