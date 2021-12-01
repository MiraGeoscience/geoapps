#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from copy import deepcopy
from uuid import UUID

from geoapps.io.Inversion import InversionParams

from ..validators import InputValidator
from .constants import (
    default_ui_json,
    forward_defaults,
    forward_ui_json,
    inversion_defaults,
    inversion_ui_json,
    required_parameters,
    validations,
)


class InducedPolarizationParams(InversionParams):

    _required_parameters = required_parameters
    _validations = validations
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

    def __init__(self, input_file=None, default=True, validate=True, **kwargs):

        self.validate = False
        self.default_ui_json = deepcopy(default_ui_json)
        self.inversion_type = "induced polarization"
        self.chargeability_channel_bool = None
        self.chargeability_channel = None
        self.chargeability_uncertainty = None
        self.conductivity_model_object = None
        self.conductivity_model = None
        self.out_group = None

        super().__init__(input_file, default, validate, **kwargs)

    @property
    def inversion_type(self):
        return self._inversion_type

    @inversion_type.setter
    def inversion_type(self, val):
        self.setter_validator("inversion_type", val)

    @property
    def chargeability_channel_bool(self):
        return self._chargeability_channel_bool

    @chargeability_channel_bool.setter
    def chargeability_channel_bool(self, val):
        if val is None:
            self._chargeability_channel_bool = val
            return
        p = "chargeability_channel_bool"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._chargeability_channel_bool = val

    @property
    def chargeability_channel(self):
        return self._chargeability_channel

    @chargeability_channel.setter
    def chargeability_channel(self, val):
        self.setter_validator(
            "chargeability_channel",
            val,
            fun=lambda x: UUID(x) if isinstance(x, str) else x,
        )

    @property
    def chargeability_uncertainty(self):
        return self._chargeability_uncertainty

    @chargeability_uncertainty.setter
    def chargeability_uncertainty(self, val):
        self.setter_validator(
            "chargeability_uncertainty",
            val,
            fun=lambda x: UUID(x) if isinstance(x, str) else x,
        )

    @property
    def conductivity_model_object(self):
        return self._conductivity_model_object

    @conductivity_model_object.setter
    def conductivity_model_object(self, val):
        self.setter_validator(
            "conductivity_model_object",
            val,
            fun=lambda x: UUID(x) if isinstance(x, str) else x,
        )

    @property
    def conductivity_model(self):
        return self._conductivity_model

    @conductivity_model.setter
    def conductivity_model(self, val):
        self.setter_validator(
            "conductivity_model",
            val,
            fun=lambda x: UUID(x) if isinstance(x, str) else x,
        )
