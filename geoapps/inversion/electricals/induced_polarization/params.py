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


class InducedPolarizationParams(InversionBaseParams):
    """
    Parameter class for electrical-induced polarization (IP) inversion.
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
        self._inversion_type = "induced polarization"
        self._validations = validations
        self.chargeability_channel_bool = None
        self.chargeability_channel = None
        self.chargeability_uncertainty = None
        self.conductivity_model_object = None
        self.conductivity_model = None

        super().__init__(input_file=input_file, forward_only=forward_only, **kwargs)

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
        self.setter_validator("chargeability_channel_bool", val)

    @property
    def chargeability_channel(self):
        return self._chargeability_channel

    @chargeability_channel.setter
    def chargeability_channel(self, val):
        self.setter_validator("chargeability_channel", val, fun=self._uuid_promoter)

    @property
    def chargeability_uncertainty(self):
        return self._chargeability_uncertainty

    @chargeability_uncertainty.setter
    def chargeability_uncertainty(self, val):
        self.setter_validator("chargeability_uncertainty", val, fun=self._uuid_promoter)

    @property
    def conductivity_model_object(self):
        return self._conductivity_model_object

    @conductivity_model_object.setter
    def conductivity_model_object(self, val):
        self.setter_validator("conductivity_model_object", val, fun=self._uuid_promoter)

    @property
    def conductivity_model(self):
        return self._conductivity_model

    @conductivity_model.setter
    def conductivity_model(self, val):
        self.setter_validator("conductivity_model", val, fun=self._uuid_promoter)
