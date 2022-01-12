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


class MagnetotelluricsParams(InversionParams):

    _required_parameters = required_parameters
    _validations = validations
    _forward_defaults = forward_defaults
    _inversion_defaults = inversion_defaults
    forward_ui_json = forward_ui_json
    inversion_ui_json = inversion_ui_json
    _directive_list = [
        "VectorInversion",
        "Update_IRLS",
        "UpdateSensitivityWeights",
        "BetaEstimate_ByEig",
        "UpdatePreconditioner",
        "SaveIterationsGeoH5",
    ]

    def __init__(
        self, input_file=None, default=True, validate=True, validator_opts={}, **kwargs
    ):

        self.validate = False
        self.default_ui_json = deepcopy(default_ui_json)
        self.inversion_type: str = "magnetotellurics"
        self._zxx_channel_bool = None
        self._zxx_channel = None
        self._zxx_uncertainty = None
        self._zxy_channel_bool = None
        self._zxy_channel = None
        self._zxy_uncertainty = None
        self._zyx_channel_bool = None
        self._zyx_channel = None
        self._zyx_uncertainty = None
        self._zyy_channel_bool = None
        self._zyy_channel = None
        self._zyy_uncertainty = None

        super().__init__(input_file, default, validate, validator_opts, **kwargs)

    @property
    def zxx_channel_bool(self):
        return self._zxx_channel_bool

    @zxx_channel_bool.setter
    def zxx_channel_bool(self, val):
        self.setter_validator("zxx_channel_bool", val)

    @property
    def zxx_channel(self):
        return self._zxx_channel

    @zxx_channel.setter
    def zxx_channel(self, val):
        self.setter_validator("zxx_channel", val, fun=self._uuid_promoter)

    @property
    def zxx_uncertainty(self):
        return self._zxx_uncertainty

    @zxx_uncertainty.setter
    def zxx_uncertainty(self, val):
        self.setter_validator("zxx_uncertainty", val, fun=self._uuid_promoter)

    @property
    def zxy_channel_bool(self):
        return self._zxy_channel_bool

    @zxy_channel_bool.setter
    def zxy_channel_bool(self, val):
        self.setter_validator("zxy_channel_bool", val)

    @property
    def zxy_channel(self):
        return self._zxy_channel

    @zxy_channel.setter
    def zxy_channel(self, val):
        self.setter_validator("zxy_channel", val, fun=self._uuid_promoter)

    @property
    def zxy_uncertainty(self):
        return self._zxy_uncertainty

    @zxy_uncertainty.setter
    def zxy_uncertainty(self, val):
        self.setter_validator("zxy_uncertainty", val, fun=self._uuid_promoter)

    @property
    def zyx_channel_bool(self):
        return self._zyx_channel_bool

    @zyx_channel_bool.setter
    def zyx_channel_bool(self, val):
        self.setter_validator("zyx_channel_bool", val)

    @property
    def zyx_channel(self):
        return self._zyx_channel

    @zyx_channel.setter
    def zyx_channel(self, val):
        self.setter_validator("zyx_channel", val, fun=self._uuid_promoter)

    @property
    def zyx_uncertainty(self):
        return self._zyx_uncertainty

    @zyx_uncertainty.setter
    def zyx_uncertainty(self, val):
        self.setter_validator("zyx_uncertainty", val, fun=self._uuid_promoter)

    @property
    def zyy_channel_bool(self):
        return self._zyy_channel_bool

    @zyy_channel_bool.setter
    def zyy_channel_bool(self, val):
        self.setter_validator("zyy_channel_bool", val)

    @property
    def zyy_channel(self):
        return self._zyy_channel

    @zyy_channel.setter
    def zyy_channel(self, val):
        self.setter_validator("zyy_channel", val, fun=self._uuid_promoter)

    @property
    def zyy_uncertainty(self):
        return self._zyy_uncertainty

    @zyy_uncertainty.setter
    def zyy_uncertainty(self, val):
        self.setter_validator("zyy_uncertainty", val, fun=self._uuid_promoter)
